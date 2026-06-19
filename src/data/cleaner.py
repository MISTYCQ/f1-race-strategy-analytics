"""
src/data/cleaner.py
===================
F1DataCleaner — reads raw Parquet files, cleans every dataset, flags
outliers, detects wet races, measures data quality, and writes processed
Parquet files plus two CSV reports.

Pipeline
--------
1. Load   — read all Parquet files from data/raw/
2. Clean  — per-dataset cleaning (missing values, types, categories)
3. Flag   — outlier indicator columns (never delete rows)
4. Enrich — join wet_race flag onto results and laps
5. Report — quality_report.csv + data_dictionary.csv
6. Save   — write processed Parquet to data/processed/

Assumptions (documented here and in quality_report.csv)
---------------------------------------------------------
A1  Lap times < 60 s or > 300 s are flagged as outliers (pit/SC laps
    inflate times; sub-60 s is physically impossible on any F1 circuit).
A2  Pit stop durations < 1.5 s or > 60 s are flagged as suspicious.
    <1.5 s = data artifact; >60 s = drive-through or extended stop.
A3  Missing compound on inlaps / outlaps (is_accurate == False) is filled
    with "UNKNOWN" — these laps are excluded from degradation analysis.
A4  Missing compound on accurate laps is forward-filled within each
    driver-stint group, then back-filled as a fallback.
A5  Missing pit_duration_s where stop is otherwise valid is left as NaN
    and flagged with pit_duration_missing = True.
A6  A race is marked wet_race = True if ANY weather sample recorded
    rainfall = True during that session.
A7  Grid position 0 is remapped to NaN (used by FastF1 for pit-lane
    starts; retains the row but signals missing true grid data).
A8  Duplicate lap rows (same driver + lap_number within a race) are
    logged and de-duplicated — the first occurrence is kept.

Usage
-----
    from src.data.cleaner import F1DataCleaner

    cleaner = F1DataCleaner()
    cleaner.run()
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
RAW_RESULTS_DIR  = PROJECT_ROOT / "data" / "raw" / "races"
RAW_LAPS_DIR     = PROJECT_ROOT / "data" / "raw" / "laps"
RAW_PITSTOPS_DIR = PROJECT_ROOT / "data" / "raw" / "pit_stops"
RAW_WEATHER_DIR  = PROJECT_ROOT / "data" / "raw" / "weather"

PROCESSED_DIR    = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR      = PROJECT_ROOT / "reports"
LOGS_DIR         = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Domain thresholds (change here only — referenced throughout the module)
# ---------------------------------------------------------------------------
LAP_TIME_MIN_S:       float = 60.0    # A1
LAP_TIME_MAX_S:       float = 300.0   # A1
PIT_DURATION_MIN_S:   float = 1.5     # A2
PIT_DURATION_MAX_S:   float = 60.0    # A2
VALID_DRY_COMPOUNDS   = {"SOFT", "MEDIUM", "HARD"}
VALID_ALL_COMPOUNDS   = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"}
VALID_STATUS_CODES    = {
    "Finished", "Accident", "Collision", "Engine", "Gearbox",
    "Hydraulics", "Electrical", "Suspension", "Brakes", "Mechanical",
    "Overheating", "Retired", "Disqualified", "Withdrew", "+1 Lap",
    "+2 Laps", "+3 Laps", "+4 Laps", "+5 Laps",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(name: str = "f1_cleaner") -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt_console = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )
    fmt_file = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    if log.handlers:
        return log

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)
    log.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        LOGS_DIR / f"cleaner_{time.strftime('%Y-%m-%d')}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    log.addHandler(fh)

    return log


log = _build_logger()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DatasetQuality:
    """Quality metrics for one dataset (one Parquet file)."""
    name:               str
    row_count:          int               = 0
    col_count:          int               = 0
    duplicate_rows:     int               = 0
    missing_pct:        dict[str, float]  = field(default_factory=dict)
    invalid_counts:     dict[str, int]    = field(default_factory=dict)
    assumptions_applied: list[str]        = field(default_factory=list)


@dataclass
class CleaningReport:
    """Aggregate report produced after a full cleaning run."""
    datasets:       list[DatasetQuality] = field(default_factory=list)
    wet_race_count: int                  = 0
    dry_race_count: int                  = 0
    total_races:    int                  = 0


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _read_parquet_dir(directory: Path, label: str) -> pd.DataFrame:
    """
    Read all Parquet files in *directory* into a single concatenated DataFrame.

    Returns an empty DataFrame (not raises) if the directory is empty or
    missing, so the pipeline can report the gap rather than crash.
    """
    if not directory.exists():
        log.warning("Directory not found: %s", directory)
        return pd.DataFrame()

    files = sorted(directory.glob("*.parquet"))
    if not files:
        log.warning("No Parquet files found in %s", directory)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f, engine="pyarrow"))
        except Exception as exc:
            log.error("Failed to read %s: %s", f.name, exc)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    log.info("Loaded %-12s  files=%3d  rows=%7d  cols=%d",
             label, len(frames), len(df), len(df.columns))
    return df


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write *df* to *path* as snappy-compressed Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    log.info("Saved  %-50s  rows=%d", path.relative_to(PROJECT_ROOT), len(df))


def _missing_pct(df: pd.DataFrame) -> dict[str, float]:
    """Return {column: pct_missing} for columns with any missing values."""
    pct = (df.isna().sum() / max(len(df), 1) * 100).round(2)
    return {col: float(val) for col, val in pct.items() if val > 0}


def _race_key(df: pd.DataFrame) -> pd.Series:
    """Composite race key as a string: '{season}_{round_number}'."""
    return df["season"].astype(str) + "_" + df["round_number"].astype(str)


# ---------------------------------------------------------------------------
# Data dictionary definitions
# ---------------------------------------------------------------------------

# Each entry: (dataset, column, description)
_DICT_ENTRIES: list[tuple[str, str, str]] = [
    # ── Shared metadata ───────────────────────────────────────────────────
    ("all",      "season",            "F1 calendar year (2023 or 2024)"),
    ("all",      "round_number",      "Round number within the season (1-based)"),
    ("all",      "event_name",        "Official FIA event name e.g. 'British Grand Prix'"),
    ("all",      "circuit",           "Circuit location e.g. 'Silverstone'"),
    # ── Results ──────────────────────────────────────────────────────────
    ("results",  "driver_number",     "FIA car number as string"),
    ("results",  "driver_code",       "Three-letter driver abbreviation e.g. 'HAM'"),
    ("results",  "full_name",         "Driver full name"),
    ("results",  "team_name",         "Constructor name e.g. 'Mercedes'"),
    ("results",  "grid_position",     "Starting grid position (NaN = pit lane start, see A7)"),
    ("results",  "finish_position",   "Classified finish position (NaN = DNF/DSQ)"),
    ("results",  "classified_position","FIA classified position string including NC/R/D codes"),
    ("results",  "status",            "FIA result status string"),
    ("results",  "points",            "Championship points scored"),
    ("results",  "wet_race",          "True if any rainfall recorded during the race (see A6)"),
    ("results",  "position_change",   "grid_position − finish_position (positive = gained)"),
    # ── Laps ─────────────────────────────────────────────────────────────
    ("laps",     "driver_code",       "Three-letter driver abbreviation"),
    ("laps",     "driver_number",     "FIA car number as string"),
    ("laps",     "team_name",         "Constructor name"),
    ("laps",     "lap_number",        "Lap number within the race (1-based)"),
    ("laps",     "lap_time_s",        "Lap time in seconds (NaN for inlap/outlap)"),
    ("laps",     "sector1_s",         "Sector 1 time in seconds"),
    ("laps",     "sector2_s",         "Sector 2 time in seconds"),
    ("laps",     "sector3_s",         "Sector 3 time in seconds"),
    ("laps",     "compound",          "Tyre compound: SOFT/MEDIUM/HARD/INTERMEDIATE/WET/UNKNOWN"),
    ("laps",     "tyre_life",         "Laps completed on current tyre set at end of this lap"),
    ("laps",     "fresh_tyre",        "True if tyre set is new (not a used set)"),
    ("laps",     "stint_number",      "Stint index within race (1 = first stint)"),
    ("laps",     "pit_in_time_s",     "Session time (s) at which driver entered pit lane"),
    ("laps",     "pit_out_time_s",    "Session time (s) at which driver exited pit lane"),
    ("laps",     "position",          "On-track position at end of lap"),
    ("laps",     "is_personal_best",  "True if this is driver's fastest lap in the session"),
    ("laps",     "track_status",      "Track status codes active during lap (1=green,2=yellow,…)"),
    ("laps",     "is_accurate",       "FastF1 timing accuracy flag; False on inlaps/outlaps"),
    ("laps",     "wet_race",          "True if any rainfall recorded during the race"),
    ("laps",     "lap_time_outlier",  "True if lap_time_s outside [60 s, 300 s] (see A1)"),
    # ── Pit stops ─────────────────────────────────────────────────────────
    ("pitstops", "driver_code",         "Three-letter driver abbreviation"),
    ("pitstops", "driver_number",       "FIA car number as string"),
    ("pitstops", "team_name",           "Constructor name"),
    ("pitstops", "stop_number",         "Sequential stop index for this driver (1, 2, 3 …)"),
    ("pitstops", "lap_number",          "Lap on which the driver pitted in"),
    ("pitstops", "stint_number",        "Stint number before this stop"),
    ("pitstops", "pit_in_time_s",       "Session time (s) at pit entry"),
    ("pitstops", "pit_out_time_s",      "Session time (s) at pit exit"),
    ("pitstops", "pit_duration_s",      "Stationary pit box time in seconds (see A2, A5)"),
    ("pitstops", "compound_fitted",     "Tyre compound fitted during this stop"),
    ("pitstops", "total_stops",         "Total number of stops made by this driver in this race"),
    ("pitstops", "pit_duration_outlier","True if pit_duration_s outside [1.5 s, 60 s] (see A2)"),
    ("pitstops", "pit_duration_missing","True if pit_duration_s could not be calculated (see A5)"),
    ("pitstops", "strategy_type",       "1-stop / 2-stop / 3-stop / 4+-stop derived from total_stops"),
    # ── Weather ──────────────────────────────────────────────────────────
    ("weather",  "session_time_s",    "Seconds elapsed since session start"),
    ("weather",  "air_temp_c",        "Ambient air temperature in °C"),
    ("weather",  "track_temp_c",      "Track surface temperature in °C"),
    ("weather",  "humidity_pct",      "Relative humidity in percent"),
    ("weather",  "pressure_mbar",     "Atmospheric pressure in millibar"),
    ("weather",  "wind_speed_ms",     "Wind speed in m/s"),
    ("weather",  "wind_direction_deg","Wind direction in degrees (0 = North)"),
    ("weather",  "rainfall",          "Boolean: True if precipitation recorded at this sample"),
]


# ---------------------------------------------------------------------------
# F1DataCleaner
# ---------------------------------------------------------------------------

class F1DataCleaner:
    """
    Reads raw Parquet files, cleans every dataset, detects wet races,
    flags outliers, measures data quality, and writes processed outputs.

    Parameters
    ----------
    raw_results_dir  : directory containing raw race result Parquet files
    raw_laps_dir     : directory containing raw lap Parquet files
    raw_pitstops_dir : directory containing raw pit stop Parquet files
    raw_weather_dir  : directory containing raw weather Parquet files
    processed_dir    : output directory for cleaned Parquet files
    reports_dir      : output directory for CSV reports
    """

    def __init__(
        self,
        raw_results_dir:  Path = RAW_RESULTS_DIR,
        raw_laps_dir:     Path = RAW_LAPS_DIR,
        raw_pitstops_dir: Path = RAW_PITSTOPS_DIR,
        raw_weather_dir:  Path = RAW_WEATHER_DIR,
        processed_dir:    Path = PROCESSED_DIR,
        reports_dir:      Path = REPORTS_DIR,
    ) -> None:
        self.raw_results_dir  = raw_results_dir
        self.raw_laps_dir     = raw_laps_dir
        self.raw_pitstops_dir = raw_pitstops_dir
        self.raw_weather_dir  = raw_weather_dir
        self.processed_dir    = processed_dir
        self.reports_dir      = reports_dir

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Populated during run()
        self._report = CleaningReport()

        log.info("F1DataCleaner initialised")
        log.info("  processed_dir : %s", self.processed_dir)
        log.info("  reports_dir   : %s", self.reports_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> CleaningReport:
        """
        Execute the full cleaning pipeline end-to-end.

        Steps
        -----
        1. Load all raw datasets.
        2. Build wet_race lookup from weather data.
        3. Clean each dataset individually.
        4. Join wet_race onto results and laps.
        5. Save processed Parquet files.
        6. Write quality_report.csv and data_dictionary.csv.

        Returns
        -------
        CleaningReport with per-dataset quality metrics.
        """
        log.info("=" * 60)
        log.info("F1DataCleaner pipeline — START")
        log.info("=" * 60)

        # ── 1. Load ───────────────────────────────────────────────────
        raw_results  = _read_parquet_dir(self.raw_results_dir,  "results")
        raw_laps     = _read_parquet_dir(self.raw_laps_dir,     "laps")
        raw_pitstops = _read_parquet_dir(self.raw_pitstops_dir, "pitstops")
        raw_weather  = _read_parquet_dir(self.raw_weather_dir,  "weather")

        # ── 2. Wet race lookup ────────────────────────────────────────
        wet_races = self._build_wet_race_lookup(raw_weather)

        # ── 3 + 4. Clean and enrich ───────────────────────────────────
        clean_results  = self._clean_results(raw_results, wet_races)
        clean_laps     = self._clean_laps(raw_laps, wet_races)
        clean_pitstops = self._clean_pitstops(raw_pitstops)
        clean_weather  = self._clean_weather(raw_weather)

        # ── 5. Save processed files ───────────────────────────────────
        _write_parquet(clean_results,  self.processed_dir / "results.parquet")
        _write_parquet(clean_laps,     self.processed_dir / "laps.parquet")
        _write_parquet(clean_pitstops, self.processed_dir / "pitstops.parquet")
        _write_parquet(clean_weather,  self.processed_dir / "weather.parquet")

        # ── 6. Reports ────────────────────────────────────────────────
        self._write_quality_report()
        self._write_data_dictionary(
            {
                "results":  clean_results,
                "laps":     clean_laps,
                "pitstops": clean_pitstops,
                "weather":  clean_weather,
            }
        )

        log.info("=" * 60)
        log.info("F1DataCleaner pipeline — COMPLETE")
        log.info("=" * 60)
        return self._report

    # ------------------------------------------------------------------
    # Wet race detection
    # ------------------------------------------------------------------

    def _build_wet_race_lookup(self, weather: pd.DataFrame) -> dict[str, bool]:
        """
        Build a {race_key: wet_race} mapping from weather data.

        A race is wet if ANY weather sample during the session recorded
        rainfall = True  (assumption A6).

        race_key format: '{season}_{round_number}'
        """
        if weather.empty or "rainfall" not in weather.columns:
            log.warning("Weather data missing or has no 'rainfall' column.")
            return {}

        weather = weather.copy()
        weather["race_key"] = _race_key(weather)

        # Coerce rainfall to bool in case it arrived as int or object
        weather["rainfall"] = weather["rainfall"].astype(bool)

        lookup = (
            weather.groupby("race_key")["rainfall"]
            .any()
            .to_dict()
        )

        wet  = sum(v for v in lookup.values())
        dry  = len(lookup) - wet
        self._report.wet_race_count = wet
        self._report.dry_race_count = dry
        self._report.total_races    = len(lookup)

        log.info("Wet race detection: %d wet / %d dry out of %d races",
                 wet, dry, len(lookup))
        return lookup

    def _apply_wet_race(self, df: pd.DataFrame, lookup: dict[str, bool]) -> pd.DataFrame:
        """Join the wet_race boolean onto *df* using the race key."""
        if not lookup:
            df["wet_race"] = False
            return df
        df = df.copy()
        df["race_key"] = _race_key(df)
        df["wet_race"] = df["race_key"].map(lookup).fillna(False).astype(bool)
        df.drop(columns=["race_key"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # Results cleaning
    # ------------------------------------------------------------------

    def _clean_results(
        self,
        df: pd.DataFrame,
        wet_races: dict[str, bool],
    ) -> pd.DataFrame:
        """
        Clean the race results dataset.

        Operations
        ----------
        - De-duplicate rows (A8).
        - Cast finish_position and grid_position to float (allows NaN).
        - Remap grid_position == 0 to NaN (A7 — pit lane start).
        - Cast points to float.
        - Validate status codes; unknown values logged, retained.
        - Derive position_change = grid_position − finish_position.
        - Join wet_race flag.
        """
        if df.empty:
            log.warning("Results dataset is empty — skipping.")
            return df

        quality = DatasetQuality(name="results", row_count=len(df), col_count=len(df.columns))
        df = df.copy()

        # ── De-duplicate ──────────────────────────────────────────────
        dupe_cols = ["season", "round_number", "driver_code"]
        dupe_cols_present = [c for c in dupe_cols if c in df.columns]
        dupes = df.duplicated(subset=dupe_cols_present, keep="first").sum()
        if dupes:
            log.warning("Results: dropping %d duplicate rows", dupes)
            df.drop_duplicates(subset=dupe_cols_present, keep="first", inplace=True)
        quality.duplicate_rows = int(dupes)
        quality.assumptions_applied.append("A8: duplicates removed, first occurrence kept")

        # ── Numeric types ─────────────────────────────────────────────
        for col in ["finish_position", "grid_position", "points"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Grid position 0 → NaN (A7) ────────────────────────────────
        if "grid_position" in df.columns:
            pit_lane_starts = (df["grid_position"] == 0).sum()
            if pit_lane_starts:
                log.info("Results: %d pit-lane starts (grid_position 0 → NaN)", pit_lane_starts)
                df.loc[df["grid_position"] == 0, "grid_position"] = np.nan
            quality.invalid_counts["grid_position_pit_lane"] = int(pit_lane_starts)
            quality.assumptions_applied.append("A7: grid_position=0 remapped to NaN (pit lane start)")

        # ── Status validation ─────────────────────────────────────────
        if "status" in df.columns:
            df["status"] = df["status"].astype(str).str.strip()
            unknown_status = ~df["status"].isin(VALID_STATUS_CODES)
            if unknown_status.any():
                unique_unknown = df.loc[unknown_status, "status"].unique().tolist()
                log.debug("Results: %d rows with unrecognised status: %s",
                          unknown_status.sum(), unique_unknown[:10])
            quality.invalid_counts["unknown_status"] = int(unknown_status.sum())

        # ── Derive position_change ─────────────────────────────────────
        if all(c in df.columns for c in ["grid_position", "finish_position"]):
            df["position_change"] = df["grid_position"] - df["finish_position"]

        # ── Wet race flag ─────────────────────────────────────────────
        df = self._apply_wet_race(df, wet_races)

        # ── Quality metrics ───────────────────────────────────────────
        quality.row_count    = len(df)
        quality.missing_pct  = _missing_pct(df)
        self._report.datasets.append(quality)

        log.info("Results cleaned: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Laps cleaning
    # ------------------------------------------------------------------

    def _clean_laps(
        self,
        df: pd.DataFrame,
        wet_races: dict[str, bool],
    ) -> pd.DataFrame:
        """
        Clean the laps dataset.

        Operations
        ----------
        - De-duplicate on (season, round_number, driver_code, lap_number) (A8).
        - Cast numeric columns.
        - Compound cleaning:
            * Fill missing compound on inaccurate laps with "UNKNOWN" (A3).
            * Forward-fill then back-fill within driver+stint groups (A4).
            * Upper-case and validate against VALID_ALL_COMPOUNDS.
        - Standardise tyre_life: coerce to int, clip to [0, 70].
        - Outlier flag: lap_time_outlier for times outside [60, 300] s (A1).
        - Wet race flag.
        """
        if df.empty:
            log.warning("Laps dataset is empty — skipping.")
            return df

        quality = DatasetQuality(name="laps", row_count=len(df), col_count=len(df.columns))
        df = df.copy()

        # ── De-duplicate ──────────────────────────────────────────────
        dupe_cols = [c for c in ["season", "round_number", "driver_code", "lap_number"]
                     if c in df.columns]
        dupes = df.duplicated(subset=dupe_cols, keep="first").sum()
        if dupes:
            log.warning("Laps: dropping %d duplicate rows", dupes)
            df.drop_duplicates(subset=dupe_cols, keep="first", inplace=True)
        quality.duplicate_rows = int(dupes)
        quality.assumptions_applied.append("A8: duplicates removed, first occurrence kept")

        # ── Numeric types ─────────────────────────────────────────────
        numeric_cols = [
            "lap_number", "lap_time_s", "sector1_s", "sector2_s", "sector3_s",
            "tyre_life", "stint_number", "position",
            "pit_in_time_s", "pit_out_time_s",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Compound cleaning ─────────────────────────────────────────
        if "compound" in df.columns:
            # Normalise to upper-case string
            df["compound"] = df["compound"].astype(str).str.strip().str.upper()
            df["compound"].replace("NAN", np.nan, inplace=True)
            df["compound"].replace("NONE", np.nan, inplace=True)

            missing_before = df["compound"].isna().sum()

            # A3: inaccurate laps (in/out laps) → UNKNOWN
            if "is_accurate" in df.columns:
                mask_inaccurate = (~df["is_accurate"].astype(bool)) & df["compound"].isna()
                df.loc[mask_inaccurate, "compound"] = "UNKNOWN"
                quality.assumptions_applied.append(
                    "A3: missing compound on inaccurate laps filled with UNKNOWN"
                )

            # A4: ffill then bfill within driver + stint
            group_cols = [c for c in ["season", "round_number", "driver_code", "stint_number"]
                          if c in df.columns]
            if group_cols:
                df["compound"] = (
                    df.sort_values(group_cols + (["lap_number"] if "lap_number" in df.columns else []))
                    .groupby(group_cols)["compound"]
                    .transform(lambda s: s.ffill().bfill())
                )
            quality.assumptions_applied.append(
                "A4: missing compound ffilled then bfilled within driver-stint groups"
            )

            # Any still-missing → UNKNOWN
            df["compound"].fillna("UNKNOWN", inplace=True)

            missing_after = (df["compound"] == "UNKNOWN").sum() - (
                df["compound"].isna().sum()
            )
            log.info(
                "Laps: compound — %d missing before, %d UNKNOWN after fill",
                missing_before, missing_after,
            )

            # Validate
            invalid_compound = ~df["compound"].isin(VALID_ALL_COMPOUNDS)
            if invalid_compound.any():
                unique_bad = df.loc[invalid_compound, "compound"].unique().tolist()
                log.warning("Laps: %d rows with invalid compound values: %s",
                            invalid_compound.sum(), unique_bad[:10])
                df.loc[invalid_compound, "compound"] = "UNKNOWN"
            quality.invalid_counts["invalid_compound"] = int(invalid_compound.sum())

            # Cast to pandas Categorical (reduces memory ~5x on large frames)
            df["compound"] = pd.Categorical(
                df["compound"], categories=sorted(VALID_ALL_COMPOUNDS)
            )

        # ── Tyre life ─────────────────────────────────────────────────
        if "tyre_life" in df.columns:
            df["tyre_life"] = pd.to_numeric(df["tyre_life"], errors="coerce")
            df["tyre_life"] = df["tyre_life"].clip(lower=0, upper=70)

        # ── Outlier flag: lap time (A1) ───────────────────────────────
        if "lap_time_s" in df.columns:
            lap_time_mask = (
                df["lap_time_s"].notna()
                & ~df["lap_time_s"].between(LAP_TIME_MIN_S, LAP_TIME_MAX_S)
            )
            df["lap_time_outlier"] = lap_time_mask
            n_outliers = lap_time_mask.sum()
            log.info(
                "Laps: %d lap time outliers flagged (outside %.0f–%.0f s)",
                n_outliers, LAP_TIME_MIN_S, LAP_TIME_MAX_S,
            )
            quality.invalid_counts["lap_time_outlier"] = int(n_outliers)
            quality.assumptions_applied.append(
                f"A1: lap_time_outlier=True for lap_time_s outside "
                f"[{LAP_TIME_MIN_S}, {LAP_TIME_MAX_S}] s (rows retained)"
            )

        # ── Wet race flag ─────────────────────────────────────────────
        df = self._apply_wet_race(df, wet_races)

        # ── Quality metrics ───────────────────────────────────────────
        quality.row_count   = len(df)
        quality.missing_pct = _missing_pct(df)
        self._report.datasets.append(quality)

        log.info("Laps cleaned: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Pit stops cleaning
    # ------------------------------------------------------------------

    def _clean_pitstops(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the pit stops dataset.

        Operations
        ----------
        - De-duplicate on (season, round_number, driver_code, stop_number) (A8).
        - Cast numeric columns.
        - Flag missing pit_duration_s with pit_duration_missing (A5).
        - Flag outlier pit_duration_s with pit_duration_outlier (A2).
        - Validate compound_fitted; unknown values → "UNKNOWN".
        - Derive strategy_type from total_stops.
        """
        if df.empty:
            log.warning("Pit stops dataset is empty — skipping.")
            return df

        quality = DatasetQuality(name="pitstops", row_count=len(df), col_count=len(df.columns))
        df = df.copy()

        # ── De-duplicate ──────────────────────────────────────────────
        dupe_cols = [c for c in
                     ["season", "round_number", "driver_code", "stop_number"]
                     if c in df.columns]
        dupes = df.duplicated(subset=dupe_cols, keep="first").sum()
        if dupes:
            log.warning("Pit stops: dropping %d duplicate rows", dupes)
            df.drop_duplicates(subset=dupe_cols, keep="first", inplace=True)
        quality.duplicate_rows = int(dupes)

        # ── Numeric types ─────────────────────────────────────────────
        for col in ["lap_number", "stint_number", "stop_number", "total_stops",
                    "pit_in_time_s", "pit_out_time_s", "pit_duration_s"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Missing pit duration flag (A5) ────────────────────────────
        if "pit_duration_s" in df.columns:
            missing_mask = df["pit_duration_s"].isna()
            df["pit_duration_missing"] = missing_mask
            n_missing = missing_mask.sum()
            log.info("Pit stops: %d rows with missing pit_duration_s", n_missing)
            quality.invalid_counts["pit_duration_missing"] = int(n_missing)
            quality.assumptions_applied.append(
                "A5: missing pit_duration_s retained as NaN, flagged with pit_duration_missing"
            )

            # ── Outlier flag (A2) ──────────────────────────────────────
            outlier_mask = (
                df["pit_duration_s"].notna()
                & ~df["pit_duration_s"].between(PIT_DURATION_MIN_S, PIT_DURATION_MAX_S)
            )
            df["pit_duration_outlier"] = outlier_mask
            n_outliers = outlier_mask.sum()
            log.info(
                "Pit stops: %d duration outliers flagged (outside %.1f–%.0f s)",
                n_outliers, PIT_DURATION_MIN_S, PIT_DURATION_MAX_S,
            )
            quality.invalid_counts["pit_duration_outlier"] = int(n_outliers)
            quality.assumptions_applied.append(
                f"A2: pit_duration_outlier=True for pit_duration_s outside "
                f"[{PIT_DURATION_MIN_S}, {PIT_DURATION_MAX_S}] s (rows retained)"
            )

        # ── Compound fitted validation ────────────────────────────────
        if "compound_fitted" in df.columns:
            df["compound_fitted"] = (
                df["compound_fitted"].astype(str).str.strip().str.upper()
            )
            df["compound_fitted"].replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN"}, inplace=True)
            invalid = ~df["compound_fitted"].isin(VALID_ALL_COMPOUNDS)
            if invalid.any():
                df.loc[invalid, "compound_fitted"] = "UNKNOWN"
            df["compound_fitted"] = pd.Categorical(
                df["compound_fitted"], categories=sorted(VALID_ALL_COMPOUNDS)
            )

        # ── Strategy type ─────────────────────────────────────────────
        if "total_stops" in df.columns:
            def _strategy_label(n: float) -> str:
                if pd.isna(n):
                    return "Unknown"
                n = int(n)
                if n == 1:
                    return "1-stop"
                if n == 2:
                    return "2-stop"
                if n == 3:
                    return "3-stop"
                return "4+-stop"

            df["strategy_type"] = df["total_stops"].apply(_strategy_label)
            df["strategy_type"] = pd.Categorical(
                df["strategy_type"],
                categories=["1-stop", "2-stop", "3-stop", "4+-stop", "Unknown"],
            )

        # ── Quality metrics ───────────────────────────────────────────
        quality.row_count   = len(df)
        quality.missing_pct = _missing_pct(df)
        self._report.datasets.append(quality)

        log.info("Pit stops cleaned: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Weather cleaning
    # ------------------------------------------------------------------

    def _clean_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the weather dataset.

        Operations
        ----------
        - Cast all sensor columns to float.
        - Coerce rainfall to bool.
        - Clip physically impossible values:
            air_temp_c       → [−10, 60] °C
            track_temp_c     → [0, 80] °C
            humidity_pct     → [0, 100] %
            wind_speed_ms    → [0, 100] m/s
            pressure_mbar    → [800, 1100] mbar
        - Log clipped counts as quality signals (rows not deleted).
        """
        if df.empty:
            log.warning("Weather dataset is empty — skipping.")
            return df

        quality = DatasetQuality(name="weather", row_count=len(df), col_count=len(df.columns))
        df = df.copy()

        # ── Numeric types ─────────────────────────────────────────────
        float_cols = [
            "session_time_s", "air_temp_c", "track_temp_c",
            "humidity_pct", "pressure_mbar",
            "wind_speed_ms", "wind_direction_deg",
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Rainfall → bool ───────────────────────────────────────────
        if "rainfall" in df.columns:
            df["rainfall"] = df["rainfall"].astype(bool)

        # ── Physical range clips ──────────────────────────────────────
        clips: dict[str, tuple[float, float]] = {
            "air_temp_c":       (-10.0, 60.0),
            "track_temp_c":     (  0.0, 80.0),
            "humidity_pct":     (  0.0, 100.0),
            "wind_speed_ms":    (  0.0, 100.0),
            "pressure_mbar":    (800.0, 1100.0),
        }
        for col, (lo, hi) in clips.items():
            if col not in df.columns:
                continue
            out_of_range = df[col].notna() & ~df[col].between(lo, hi)
            n = out_of_range.sum()
            if n:
                log.warning(
                    "Weather: %d out-of-range values in %s clipped to [%.0f, %.0f]",
                    n, col, lo, hi,
                )
                quality.invalid_counts[f"{col}_clipped"] = int(n)
            df[col] = df[col].clip(lower=lo, upper=hi)

        # ── Quality metrics ───────────────────────────────────────────
        quality.missing_pct = _missing_pct(df)
        quality.row_count   = len(df)
        self._report.datasets.append(quality)

        log.info("Weather cleaned: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Quality report
    # ------------------------------------------------------------------

    def _write_quality_report(self) -> None:
        """
        Write reports/quality_report.csv.

        Schema
        ------
        dataset, metric, column_or_key, value, note
        """
        rows: list[dict] = []

        # Wet/dry race summary
        rows.append({
            "dataset": "weather",
            "metric":  "wet_race_count",
            "column_or_key": "rainfall",
            "value":   self._report.wet_race_count,
            "note":    "Races with at least one rainfall=True weather sample (A6)",
        })
        rows.append({
            "dataset": "weather",
            "metric":  "dry_race_count",
            "column_or_key": "rainfall",
            "value":   self._report.dry_race_count,
            "note":    "Races with no rainfall recorded",
        })

        for dq in self._report.datasets:
            # Row / column count
            rows.append({"dataset": dq.name, "metric": "row_count",
                         "column_or_key": "", "value": dq.row_count, "note": ""})
            rows.append({"dataset": dq.name, "metric": "col_count",
                         "column_or_key": "", "value": dq.col_count, "note": ""})
            rows.append({"dataset": dq.name, "metric": "duplicate_rows",
                         "column_or_key": "", "value": dq.duplicate_rows, "note": ""})

            # Missing value percentages
            for col, pct in dq.missing_pct.items():
                rows.append({
                    "dataset": dq.name,
                    "metric":  "missing_pct",
                    "column_or_key": col,
                    "value":   pct,
                    "note":    f"{pct:.2f}% of rows have no value",
                })

            # Invalid / flagged counts
            for key, count in dq.invalid_counts.items():
                rows.append({
                    "dataset": dq.name,
                    "metric":  "invalid_count",
                    "column_or_key": key,
                    "value":   count,
                    "note":    "",
                })

            # Assumptions applied
            for assumption in dq.assumptions_applied:
                rows.append({
                    "dataset": dq.name,
                    "metric":  "assumption_applied",
                    "column_or_key": "",
                    "value":   "",
                    "note":    assumption,
                })

        out_path = self.reports_dir / "quality_report.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        log.info("Quality report → %s  (%d rows)", out_path.relative_to(PROJECT_ROOT), len(rows))

    # ------------------------------------------------------------------
    # Data dictionary
    # ------------------------------------------------------------------

    def _write_data_dictionary(self, cleaned: dict[str, pd.DataFrame]) -> None:
        """
        Write reports/data_dictionary.csv.

        Schema
        ------
        dataset, column_name, data_type, nullable, description
        """
        # Build a lookup of actual dtypes from the cleaned frames
        dtype_lookup: dict[tuple[str, str], str] = {}
        for ds_name, frame in cleaned.items():
            for col in frame.columns:
                dtype_lookup[(ds_name, col)] = str(frame[col].dtype)

        rows: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for dataset, column, description in _DICT_ENTRIES:
            # "all" entries apply to every dataset
            targets = list(cleaned.keys()) if dataset == "all" else [dataset]
            for ds in targets:
                key = (ds, column)
                if key in seen:
                    continue
                seen.add(key)
                frame  = cleaned.get(ds, pd.DataFrame())
                dtype  = dtype_lookup.get(key, "N/A")
                exists = column in (frame.columns if not frame.empty else [])
                rows.append({
                    "dataset":     ds,
                    "column_name": column,
                    "data_type":   dtype,
                    "nullable":    "Yes" if exists and frame[column].isna().any() else "No",
                    "description": description,
                })

        # Add any columns present in the data but not in the manual list
        for ds_name, frame in cleaned.items():
            for col in frame.columns:
                if (ds_name, col) not in seen:
                    rows.append({
                        "dataset":     ds_name,
                        "column_name": col,
                        "data_type":   str(frame[col].dtype),
                        "nullable":    "Yes" if frame[col].isna().any() else "No",
                        "description": "(auto-detected — add description to _DICT_ENTRIES)",
                    })
                    seen.add((ds_name, col))

        out_path = self.reports_dir / "data_dictionary.csv"
        (
            pd.DataFrame(rows)
            .sort_values(["dataset", "column_name"])
            .to_csv(out_path, index=False)
        )
        log.info(
            "Data dictionary → %s  (%d entries)",
            out_path.relative_to(PROJECT_ROOT), len(rows),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean raw F1 Parquet files and write processed outputs."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Output directory for cleaned Parquet files",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Output directory for quality_report.csv and data_dictionary.csv",
    )
    args = parser.parse_args()

    cleaner = F1DataCleaner(
        processed_dir = args.processed_dir,
        reports_dir   = args.reports_dir,
    )
    report = cleaner.run()

    print(f"\n  Datasets cleaned : {len(report.datasets)}")
    print(f"  Wet races        : {report.wet_race_count}")
    print(f"  Dry races        : {report.dry_race_count}")
    print(f"  Reports written to: {args.reports_dir}\n")