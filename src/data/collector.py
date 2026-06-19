"""
src/data/collector.py
=====================
F1DataCollector — object-oriented FastF1 wrapper.

Responsibilities
----------------
1. Iterate over every race event in the requested seasons.
2. Download five datasets per race:
       • Race results   → data/raw/races/
       • Lap times      → data/raw/laps/
       • Pit stops      → data/raw/pit_stops/
       • Tire compounds → embedded inside laps (also saved separately)
       • Weather        → data/raw/weather/
3. Persist each dataset as a Parquet file with a deterministic filename.
4. Skip races already downloaded (idempotent pipeline).
5. Retry transient network failures with exponential back-off.
6. Emit structured logs for every action so failures are easy to diagnose.

Usage
-----
    from src.data.collector import F1DataCollector

    collector = F1DataCollector(seasons=[2023, 2024])
    collector.run()
    summary = collector.get_summary()
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

# Add project root to sys.path so config is importable when this script
# is executed directly (e.g. python src/data/collector.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    FASTF1_CACHE_DIR,
    LOGS_DIR,
    RAW_RACES_DIR,
    RAW_LAPS_DIR,
    RAW_PITS_DIR,
    RAW_WEATHER_DIR,
    PARQUET_COMPRESSION,
    MAX_RETRIES,
    RETRY_WAIT_MIN,
    RETRY_WAIT_MAX,
    LOG_LEVEL,
    LOG_FORMAT,
    SESSION_TYPE,
)


# ── Logging setup ─────────────────────────────────────────────────────────────

def _configure_logging(log_dir: Path, level: str = LOG_LEVEL) -> None:
    """
    Configure Loguru with two handlers:
      • Colour output to stdout for interactive use.
      • Plain text file rotated at 10 MB, retained for 7 days.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()  # Remove Loguru's default handler

    # Console
    logger.add(
        sys.stdout,
        level=level,
        format=LOG_FORMAT,
        colorize=True,
    )

    # File
    logger.add(
        log_dir / "collector_{time:YYYY-MM-DD}.log",
        level=level,
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RaceIdentifier:
    """Minimal info needed to uniquely identify and retrieve a race session."""
    season:       int
    round_number: int
    event_name:   str   # e.g. "British Grand Prix"
    country:      str   # e.g. "United Kingdom"
    circuit:      str   # e.g. "Silverstone"


@dataclass
class CollectionResult:
    """Outcome record for a single race download attempt."""
    race:         RaceIdentifier
    races_path:   Optional[Path] = None
    laps_path:    Optional[Path] = None
    pits_path:    Optional[Path] = None
    weather_path: Optional[Path] = None
    skipped:      bool = False   # True if all files already existed
    success:      bool = False
    error:        str  = ""


@dataclass
class CollectionSummary:
    """Aggregate statistics across all processed races."""
    total_races:   int       = 0
    successful:    int       = 0
    skipped:       int       = 0
    failed:        int       = 0
    failed_events: list[str] = field(default_factory=list)


# ── Main collector class ───────────────────────────────────────────────────────

class F1DataCollector:
    """
    Downloads and persists F1 race data for the requested seasons.

    Parameters
    ----------
    seasons : list[int]
        Calendar years to collect, e.g. [2023, 2024].
    session_type : str
        FastF1 session identifier — "R" for Race (default).
    overwrite : bool
        If True, re-download even if the Parquet file already exists.
        Default False keeps the pipeline idempotent.
    cache_dir : Path
        Directory FastF1 uses to cache raw session data. Prevents
        repeated HTTP requests on subsequent runs.
    """

    def __init__(
        self,
        seasons:      list[int],
        session_type: str  = SESSION_TYPE,
        overwrite:    bool = False,
        cache_dir:    Path = FASTF1_CACHE_DIR,
    ) -> None:
        self.seasons      = seasons
        self.session_type = session_type
        self.overwrite    = overwrite
        self.cache_dir    = cache_dir
        self._results: list[CollectionResult] = []

        # Create every output directory up-front so writers never fail on mkdir.
        for directory in [RAW_RACES_DIR, RAW_LAPS_DIR, RAW_PITS_DIR,
                          RAW_WEATHER_DIR, cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        _configure_logging(LOGS_DIR)

        # Enable FastF1 local cache — dramatically speeds up re-runs.
        fastf1.Cache.enable_cache(str(self.cache_dir))
        logger.info(f"FastF1 cache  : {self.cache_dir}")
        logger.info(f"Seasons       : {self.seasons}")
        logger.info(f"Session type  : {self.session_type}")
        logger.info(f"Overwrite     : {self.overwrite}")

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self) -> CollectionSummary:
        """
        Main entry point. Iterates every season → race and collects all
        datasets. Returns an aggregate CollectionSummary.
        """
        logger.info("=" * 60)
        logger.info("F1 Data Collection Pipeline — START")
        logger.info("=" * 60)

        for season in self.seasons:
            self._collect_season(season)

        summary = self.get_summary()
        self._log_summary(summary)
        return summary

    def get_summary(self) -> CollectionSummary:
        """Build a CollectionSummary from stored per-race results."""
        summary = CollectionSummary(total_races=len(self._results))
        for r in self._results:
            if r.skipped:
                summary.skipped += 1
            elif r.success:
                summary.successful += 1
            else:
                summary.failed += 1
                summary.failed_events.append(
                    f"{r.race.season} R{r.race.round_number} {r.race.event_name}"
                )
        return summary

    # ── Season-level iteration ────────────────────────────────────────────────

    def _collect_season(self, season: int) -> None:
        """Fetch the event schedule for a season and process each race."""
        logger.info(f"── Season {season} ──────────────────────────────────")

        try:
            schedule = fastf1.get_event_schedule(season, include_testing=False)
        except Exception as exc:
            logger.error(f"Failed to fetch schedule for {season}: {exc}")
            return

        # RoundNumber > 0 filters out pre-season testing events.
        races = schedule[schedule["RoundNumber"] > 0].copy()
        logger.info(f"Season {season}: {len(races)} race events found")

        for _, event in races.iterrows():
            race_id = RaceIdentifier(
                season       = season,
                round_number = int(event["RoundNumber"]),
                event_name   = str(event["EventName"]),
                country      = str(event.get("Country", "Unknown")),
                circuit      = str(event.get("Location", "Unknown")),
            )
            result = self._collect_race(race_id)
            self._results.append(result)

            # Brief pause between races — polite to FastF1 / Ergast servers.
            time.sleep(1)

    # ── Race-level collection ─────────────────────────────────────────────────

    def _collect_race(self, race: RaceIdentifier) -> CollectionResult:
        """
        Collect all datasets for a single race.

        Returns a CollectionResult regardless of success/failure so the
        summary can always be built without crashing the pipeline.
        """
        label  = f"{race.season} R{race.round_number:02d} — {race.event_name}"
        result = CollectionResult(race=race)

        logger.info(f"Processing: {label}")

        # Build deterministic file paths once — used for skip-check and writing.
        slug         = self._make_slug(race)
        races_path   = RAW_RACES_DIR   / f"{slug}_results.parquet"
        laps_path    = RAW_LAPS_DIR    / f"{slug}_laps.parquet"
        pits_path    = RAW_PITS_DIR    / f"{slug}_pitstops.parquet"
        weather_path = RAW_WEATHER_DIR / f"{slug}_weather.parquet"

        all_exist = all(
            p.exists() for p in [races_path, laps_path, pits_path, weather_path]
        )

        if all_exist and not self.overwrite:
            logger.info(f"  SKIP — all files already exist for {label}")
            result.skipped      = True
            result.races_path   = races_path
            result.laps_path    = laps_path
            result.pits_path    = pits_path
            result.weather_path = weather_path
            return result

        # ── Load session ──────────────────────────────────────────────────────
        try:
            session = self._load_session_with_retry(race.season, race.round_number)
        except RetryError as exc:
            logger.error(
                f"  FAIL — could not load session after {MAX_RETRIES} retries: {exc}"
            )
            result.error = str(exc)
            return result
        except Exception as exc:
            logger.error(f"  FAIL — unexpected error loading session: {exc}")
            result.error = str(exc)
            return result

        # ── Extract and save each dataset ─────────────────────────────────────
        try:
            result.races_path   = self._save_race_results(session, race, races_path)
            result.laps_path    = self._save_laps(session, race, laps_path)
            result.pits_path    = self._save_pit_stops(session, race, pits_path)
            result.weather_path = self._save_weather(session, race, weather_path)
            result.success      = True
            logger.success(f"  OK  — {label}")

        except Exception as exc:
            logger.error(f"  FAIL — error extracting data: {exc}", exc_info=True)
            result.error = str(exc)

        return result

    # ── FastF1 session loader with retry ──────────────────────────────────────

    @retry(
        retry      = retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        stop       = stop_after_attempt(MAX_RETRIES),
        wait       = wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        before_sleep = before_sleep_log(logger, "WARNING"),
        reraise    = True,
    )
    def _load_session_with_retry(
        self, season: int, round_number: int
    ) -> fastf1.core.Session:
        """
        Load a FastF1 session and download all required data.

        The @retry decorator catches transient network errors and retries
        up to MAX_RETRIES times with exponential back-off between attempts.

        telemetry=False — raw telemetry is very large; collected separately
                          if needed in a dedicated telemetry pipeline.
        """
        session = fastf1.get_session(season, round_number, self.session_type)
        session.load(
            laps      = True,
            telemetry = False,
            weather   = True,
            messages  = False,
        )
        logger.debug(f"  Loaded: {session.event['EventName']} {season}")
        return session

    # ── Dataset extractors ────────────────────────────────────────────────────

    def _save_race_results(
        self,
        session:     fastf1.core.Session,
        race:        RaceIdentifier,
        output_path: Path,
    ) -> Path:
        """
        Extract the final race classification.

        Columns: position, grid_position, driver_code, full_name,
                 team_name, status, points, race_time.
        """
        results = session.results.copy()

        if results.empty:
            logger.warning(f"  Race results empty for {race.event_name} {race.season}")
            return output_path

        results = self._add_race_metadata(results, race)

        column_map = {
            "DriverNumber":       "driver_number",
            "BroadcastName":      "broadcast_name",
            "Abbreviation":       "driver_code",
            "DriverId":           "driver_id",
            "TeamName":           "team_name",
            "TeamColor":          "team_color",
            "FirstName":          "first_name",
            "LastName":           "last_name",
            "FullName":           "full_name",
            "HeadshotUrl":        "headshot_url",
            "CountryCode":        "country_code",
            "Position":           "finish_position",
            "ClassifiedPosition": "classified_position",
            "GridPosition":       "grid_position",
            "Q1":                 "q1_time",
            "Q2":                 "q2_time",
            "Q3":                 "q3_time",
            "Time":               "race_time",
            "Status":             "status",
            "Points":             "points",
        }
        results.rename(columns=column_map, inplace=True)

        # Cast positions to numeric — "DNF", "DSQ" etc. become NaN.
        for col in ["finish_position", "grid_position"]:
            if col in results.columns:
                results[col] = pd.to_numeric(results[col], errors="coerce")

        self._write_parquet(results, output_path)
        logger.debug(f"  Saved results  → {output_path.name}  ({len(results)} rows)")
        return output_path

    def _save_laps(
        self,
        session:     fastf1.core.Session,
        race:        RaceIdentifier,
        output_path: Path,
    ) -> Path:
        """
        Extract per-driver per-lap data including tire compound and stint info.

        This is the richest dataset — it drives tire degradation and strategy
        type calculations in the feature engineering phase.

        Key columns
        -----------
        lap_number, lap_time_s, sector1_s, sector2_s, sector3_s,
        compound, tyre_life, stint_number, pit_in_time_s, pit_out_time_s,
        is_personal_best, track_status, driver_code, team_name.
        """
        laps = session.laps.copy()

        if laps.empty:
            logger.warning(f"  Laps empty for {race.event_name} {race.season}")
            return output_path

        laps = self._add_race_metadata(laps, race)

        # Convert all timedelta columns to total seconds (float).
        # Parquet does not support pandas Timedelta natively and downstream
        # arithmetic is far simpler with plain floats.
        timedelta_cols = [
            "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
            "PitInTime", "PitOutTime", "Time", "LapStartTime",
        ]
        for col in timedelta_cols:
            if col in laps.columns:
                laps[col] = (
                    pd.to_timedelta(laps[col], errors="coerce").dt.total_seconds()
                )

        column_map = {
            "Time":           "session_time_s",
            "Driver":         "driver_code",
            "DriverNumber":   "driver_number",
            "LapTime":        "lap_time_s",
            "LapNumber":      "lap_number",
            "Stint":          "stint_number",
            "PitOutTime":     "pit_out_time_s",
            "PitInTime":      "pit_in_time_s",
            "Sector1Time":    "sector1_s",
            "Sector2Time":    "sector2_s",
            "Sector3Time":    "sector3_s",
            "SpeedI1":        "speed_i1",
            "SpeedI2":        "speed_i2",
            "SpeedFL":        "speed_fl",
            "SpeedST":        "speed_st",
            "IsPersonalBest": "is_personal_best",
            "Compound":       "compound",
            "TyreLife":       "tyre_life",
            "FreshTyre":      "fresh_tyre",
            "LapStartTime":   "lap_start_time_s",
            "Team":           "team_name",
            "TrackStatus":    "track_status",
            "IsAccurate":     "is_accurate",
        }
        laps.rename(
            columns={k: v for k, v in column_map.items() if k in laps.columns},
            inplace=True,
        )

        self._write_parquet(laps, output_path)
        logger.debug(f"  Saved laps     → {output_path.name}  ({len(laps)} rows)")
        return output_path

    def _save_pit_stops(
        self,
        session:     fastf1.core.Session,
        race:        RaceIdentifier,
        output_path: Path,
    ) -> Path:
        """
        Derive pit stop events from lap data.

        FastF1 does not expose a dedicated pit stop table. Instead, stops
        are inferred from laps where PitInTime is not NaN.

        pit_duration_s is calculated as:
            next lap's PitOutTime  −  current lap's PitInTime

        This measures stationary box time only, excluding pit lane travel,
        making it comparable across circuits with different pit lane lengths.

        Columns produced
        ----------------
        driver_code, driver_number, team_name, lap_number, stint_number,
        pit_in_time_s, pit_out_time_s, pit_duration_s, stop_number,
        compound_before, compound_after.
        """
        laps = session.laps.copy()

        if laps.empty:
            logger.warning(f"  No laps data to derive pit stops: {race.event_name}")
            return output_path

        for col in ["PitInTime", "PitOutTime", "LapTime"]:
            if col in laps.columns:
                laps[col] = (
                    pd.to_timedelta(laps[col], errors="coerce").dt.total_seconds()
                )

        laps_sorted = laps.sort_values(["DriverNumber", "LapNumber"]).copy()

        # Shift PitOutTime up by one row within each driver so we can match
        # the pit-in on lap N with the pit-out on lap N+1.
        laps_sorted["next_pit_out"] = (
            laps_sorted.groupby("DriverNumber")["PitOutTime"].shift(-1)
        )

        # Also capture what compound was fitted after the stop.
        laps_sorted["next_compound"] = (
            laps_sorted.groupby("DriverNumber")["Compound"].shift(-1)
        )

        pit_stops = laps_sorted[laps_sorted["PitInTime"].notna()].copy()

        if pit_stops.empty:
            logger.warning(
                f"  No pit stops detected for {race.event_name} {race.season}"
            )
            # Write an empty frame with the expected schema so downstream
            # code can always pd.read_parquet() without KeyError.
            empty = pd.DataFrame(columns=[
                "season", "round_number", "event_name", "circuit",
                "driver_code", "driver_number", "team_name",
                "lap_number", "stint_number", "pit_in_time_s",
                "pit_out_time_s", "pit_duration_s", "stop_number",
                "compound_before", "compound_after",
            ])
            self._write_parquet(empty, output_path)
            return output_path

        pit_stops["pit_duration_s"] = (
            pit_stops["next_pit_out"] - pit_stops["PitInTime"]
        )

        # Sequential stop counter per driver (1st stop, 2nd stop, …).
        pit_stops["stop_number"] = (
            pit_stops.groupby("DriverNumber").cumcount() + 1
        )

        pit_stops = self._add_race_metadata(pit_stops, race)

        column_map = {
            "Driver":       "driver_code",
            "DriverNumber": "driver_number",
            "Team":         "team_name",
            "LapNumber":    "lap_number",
            "Stint":        "stint_number",
            "PitInTime":    "pit_in_time_s",
            "next_pit_out": "pit_out_time_s",
            "Compound":     "compound_before",
            "next_compound":"compound_after",
        }
        pit_stops.rename(
            columns={k: v for k, v in column_map.items() if k in pit_stops.columns},
            inplace=True,
        )

        keep = [
            "season", "round_number", "event_name", "circuit",
            "driver_code", "driver_number", "team_name",
            "lap_number", "stint_number",
            "pit_in_time_s", "pit_out_time_s", "pit_duration_s",
            "stop_number", "compound_before", "compound_after",
        ]
        pit_stops = pit_stops[[c for c in keep if c in pit_stops.columns]]

        self._write_parquet(pit_stops, output_path)
        logger.debug(
            f"  Saved pit stops → {output_path.name}  ({len(pit_stops)} stops)"
        )
        return output_path

    def _save_weather(
        self,
        session:     fastf1.core.Session,
        race:        RaceIdentifier,
        output_path: Path,
    ) -> Path:
        """
        Extract session weather data.

        FastF1 samples weather at irregular intervals (~60 s average).

        The `rainfall` boolean column is the key flag used in the cleaning
        phase to classify a race as wet or dry.

        Columns: session_time_s, air_temp_c, humidity_pct, pressure_mbar,
                 rainfall, track_temp_c, wind_direction_deg, wind_speed_ms.
        """
        weather = session.weather_data

        if weather is None or weather.empty:
            logger.warning(f"  No weather data for {race.event_name} {race.season}")
            return output_path

        weather = weather.copy()

        if "Time" in weather.columns:
            weather["Time"] = (
                pd.to_timedelta(weather["Time"], errors="coerce").dt.total_seconds()
            )

        weather = self._add_race_metadata(weather, race)

        column_map = {
            "Time":          "session_time_s",
            "AirTemp":       "air_temp_c",
            "Humidity":      "humidity_pct",
            "Pressure":      "pressure_mbar",
            "Rainfall":      "rainfall",
            "TrackTemp":     "track_temp_c",
            "WindDirection": "wind_direction_deg",
            "WindSpeed":     "wind_speed_ms",
        }
        weather.rename(
            columns={k: v for k, v in column_map.items() if k in weather.columns},
            inplace=True,
        )

        self._write_parquet(weather, output_path)
        logger.debug(f"  Saved weather  → {output_path.name}  ({len(weather)} rows)")
        return output_path

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_slug(race: RaceIdentifier) -> str:
        """
        Build a deterministic, filesystem-safe filename prefix.

        Format : {season}_R{round:02d}_{circuit}
        Example: 2023_R07_Silverstone
        """
        safe_chars   = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_-"
        )
        circuit_slug = race.circuit.replace(" ", "_").replace("/", "-")
        circuit_slug = "".join(c for c in circuit_slug if c in safe_chars)
        return f"{race.season}_R{race.round_number:02d}_{circuit_slug}"

    @staticmethod
    def _add_race_metadata(df: pd.DataFrame, race: RaceIdentifier) -> pd.DataFrame:
        """
        Prepend four identifying columns to every dataset so any Parquet
        file is self-describing when loaded in isolation.
        """
        df = df.copy()
        df.insert(0, "season",       race.season)
        df.insert(1, "round_number", race.round_number)
        df.insert(2, "event_name",   race.event_name)
        df.insert(3, "circuit",      race.circuit)
        return df

    @staticmethod
    def _write_parquet(df: pd.DataFrame, path: Path) -> None:
        """Write a DataFrame to Parquet using pyarrow + snappy compression."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            path,
            engine      = "pyarrow",
            compression = PARQUET_COMPRESSION,
            index       = False,
        )

    @staticmethod
    def _log_summary(summary: CollectionSummary) -> None:
        logger.info("=" * 60)
        logger.info("Collection complete")
        logger.info(f"  Total  : {summary.total_races}")
        logger.info(f"  OK     : {summary.successful}")
        logger.info(f"  Skipped: {summary.skipped}")
        logger.info(f"  Failed : {summary.failed}")
        if summary.failed_events:
            logger.warning("  Failed events:")
            for e in summary.failed_events:
                logger.warning(f"    • {e}")
        logger.info("=" * 60)