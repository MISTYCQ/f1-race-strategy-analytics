"""
src/features/feature_engineering.py
=====================================
F1FeatureEngineer — reads cleaned Parquet files and produces two
feature-level datasets used by EDA, SQL analysis, and Power BI.

Output datasets
---------------
data/features/driver_features.parquet
    One row per driver per race.  Contains every strategy, performance,
    tire, and context feature for that driver's race.

data/features/race_features.parquet
    One row per race.  Contains aggregate race-level metrics: dominant
    strategy, average pit counts, circuit classification, etc.

Business meaning of every feature
----------------------------------
STRATEGY FEATURES
  strategy_type         Which pit strategy the driver used (0-stop through
                        3+-stop).  The primary dimension for answering
                        "which strategy wins?".  A 1-stop minimises total
                        pit time but risks tyre degradation; a 2-stop
                        keeps fresher rubber at the cost of extra pit time.

  avg_stint_length      Mean number of laps between pit stops.  Longer
                        stints indicate a conservative strategy or a tyre
                        compound lasting well on that circuit.  Short stints
                        suggest aggressive early undercuts or tyre failures.

  total_pit_time_s      Sum of all stationary pit box times for the driver
                        in seconds.  Directly subtracts from race time —
                        every extra second spent in the pits is a second
                        lost on track.  Benchmarks crew performance vs
                        competitors.

  first_pit_lap         The lap on which the driver made their first pit
                        stop.  Early first stops (laps 1–15) often signal
                        an undercut attempt or damage; late first stops
                        (laps 25+) indicate an overcut or long-first-stint
                        strategy.

  last_pit_lap          The lap of the final pit stop.  A late last stop
                        means the driver is running fresher rubber at race
                        end, often a competitive advantage in the final
                        stint.

PERFORMANCE FEATURES
  position_gain         Grid position minus finish position.  Positive =
                        gained places; negative = lost places.  The single
                        most direct measure of whether a strategy (and
                        driver execution) improved the race result relative
                        to starting slot.

  avg_race_pace_s       Mean lap time in seconds across all accurate,
                        non-outlier laps.  Lower is faster.  Isolates
                        outright car and driver speed independent of
                        strategy.  Used to separate "the strategy worked"
                        from "the car was simply faster".

  fastest_lap_s         The driver's single quickest lap of the race in
                        seconds.  Indicates peak performance capability.
                        In modern F1 one bonus championship point is
                        awarded for the overall fastest lap, so teams
                        sometimes pit late specifically to chase it.

  pace_consistency_s    Standard deviation of lap times across accurate
                        laps.  Lower = more consistent.  A highly
                        consistent driver manages tyre degradation smoothly;
                        a high std dev suggests thermal degradation spikes,
                        traffic disruption, or driving errors.

  best_stint_pace_s     The median lap time of the driver's quickest
                        individual stint.  Reveals the maximum pace the car
                        can sustain on a given compound without the noise of
                        in/out laps or safety car periods.

TIRE FEATURES
  tire_degradation_rate  Slope of lap time vs tyre age (seconds per lap)
                         calculated by OLS linear regression within each
                         stint, then averaged across all dry-compound stints.
                         A positive slope means lap times are increasing as
                         the tyre wears — the steeper the slope, the faster
                         the tyre is falling off.  Crucial for deciding
                         optimal stint length and compound selection.

  compound_usage         Comma-separated ordered list of compounds used,
                         e.g. "MEDIUM,HARD".  Encodes the full strategic
                         narrative of the race for qualitative analysis and
                         categorical grouping.

  avg_tyre_life          Mean number of laps a tyre set was used before
                         being retired.  Longer average tyre life reduces
                         pit stop frequency and total pit time.  Comparing
                         across circuits reveals which venues are hardest on
                         rubber.

  longest_stint          Maximum number of laps on a single tyre set during
                         the race.  A very long longest stint often indicates
                         a deliberate overcut strategy or a virtual safety
                         car period that negated the need to pit.

  num_compounds_used     Count of distinct dry tyre compounds used.  FIA
                         rules mandate at least two different compounds in
                         dry races; drivers using three compounds may be
                         responding to unexpected degradation or exploiting
                         a strategic gap.

ADVANCED STRATEGY METRICS
  undercut_attempt       Boolean: True if the driver pitted before their
                         direct on-track rival within the same lap window
                         (±3 laps) and emerged ahead after the stop.
                         Undercuts exploit the performance delta of a fresh
                         tyre to manufacture a position gain in the pit
                         phase rather than on track.

  overcut_attempt        Boolean: True if the driver stayed out longer than
                         their rival and maintained or gained position by
                         running quicker laps on a less-degraded tyre while
                         the rival was in the pits.  The overcut is riskier
                         because it requires the tyre to remain competitive
                         for additional laps.

  pit_stop_efficiency    Ratio of this driver's average pit stop duration to
                         the field average for the same race.  A value < 1.0
                         means the crew is faster than average; > 1.0 means
                         slower.  Isolates crew execution from strategic
                         timing decisions.

RACE CONTEXT FEATURES
  wet_race               Boolean flag: True if any rainfall was recorded
                         during the session.  Wet races invalidate dry-tyre
                         degradation models and require completely different
                         strategy logic, so all degradation analyses should
                         filter on wet_race = False.

  circuit_type           Categorical: "street" or "permanent".  Street
                         circuits (Monaco, Singapore, Baku, Miami, Las
                         Vegas, Jeddah) have no overtaking, making strategy
                         the primary differentiator.  Permanent circuits
                         allow on-track overtaking, so strategy and pace
                         interact more.

  safety_car_deployed    Boolean: True if a safety car period was detected
                         during the race (track_status contains "4" or "5").
                         Safety car periods compress the field and often
                         trigger a wave of pit stops, making strategy
                         comparison before/after the SC period essential.

Assumptions
-----------
B1  Degradation rate is computed only on accurate laps with dry compounds
    and tyre_life >= 3 (first two laps of a stint skipped as warm-up).
B2  Undercut/overcut detection uses a ±3-lap window; drivers more than
    3 laps apart in pit timing are not considered rivals for that stop.
B3  Circuit type classification is based on a hard-coded lookup of known
    street circuits.  New circuits default to "permanent".
B4  Safety car detection uses FastF1 track_status codes: "4" = Safety Car,
    "5" = Virtual Safety Car.
B5  Pace metrics use only laps where is_accurate = True and
    lap_time_outlier = False to exclude in/out laps and SC laps.

Usage
-----
    from src.features.feature_engineering import F1FeatureEngineer

    engineer = F1FeatureEngineer()
    driver_features, race_features = engineer.run()
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
LOGS_DIR      = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Street circuits — no meaningful overtaking opportunities, strategy-dominant.
STREET_CIRCUITS: set[str] = {
    "Monaco", "Singapore", "Baku", "Jeddah", "Miami",
    "Las Vegas", "Melbourne",   # Albert Park has some street-circuit traits
}

# FastF1 track_status codes that indicate SC / VSC deployment.
# "4" = Safety Car on track, "5" = Virtual Safety Car.
SAFETY_CAR_STATUS_CODES: set[str] = {"4", "5"}

# Minimum tyre age before including a lap in degradation regression (B1).
MIN_TYRE_AGE_FOR_DEG: int = 3

# Undercut / overcut detection window in laps (B2).
STRATEGY_RIVAL_LAP_WINDOW: int = 3

# Compounds considered "dry" for degradation analysis.
DRY_COMPOUNDS: set[str] = {"SOFT", "MEDIUM", "HARD"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(name: str = "f1_feature_engineer") -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt_c = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )
    fmt_f = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    if log.handlers:
        return log

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_c)
    log.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        LOGS_DIR / f"features_{time.strftime('%Y-%m-%d')}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_f)
    log.addHandler(fh)

    return log


log = _build_logger()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _read(path: Path, label: str) -> pd.DataFrame:
    """Read a Parquet file and log its shape."""
    if not path.exists():
        log.warning("File not found — returning empty frame: %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path, engine="pyarrow")
    log.info("Loaded %-14s  rows=%7d  cols=%d", label, len(df), len(df.columns))
    return df


def _write(df: pd.DataFrame, path: Path, label: str) -> None:
    """Write a DataFrame to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    log.info("Saved  %-14s  rows=%7d  → %s", label, len(df),
             path.relative_to(PROJECT_ROOT))


def _race_key(df: pd.DataFrame) -> pd.Series:
    """Return a composite '{season}_{round_number}' Series."""
    return df["season"].astype(str) + "_" + df["round_number"].astype(str)


def _driver_race_key(df: pd.DataFrame) -> pd.Series:
    """Return a composite '{season}_{round_number}_{driver_code}' Series."""
    return (
        df["season"].astype(str)
        + "_"
        + df["round_number"].astype(str)
        + "_"
        + df["driver_code"].astype(str)
    )


def _ols_slope(x: pd.Series, y: pd.Series) -> float:
    """
    Return the OLS slope (β₁) of y ~ x using the closed-form formula.

    Falls back to NaN if fewer than 2 valid points exist or if the
    computation raises any exception (e.g. singular matrix).
    """
    try:
        mask = x.notna() & y.notna()
        xv   = x[mask].values.astype(float)
        yv   = y[mask].values.astype(float)
        if len(xv) < 2:
            return np.nan
        x_mean = xv.mean()
        y_mean = yv.mean()
        denom  = ((xv - x_mean) ** 2).sum()
        if denom == 0:
            return np.nan
        return float(((xv - x_mean) * (yv - y_mean)).sum() / denom)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# F1FeatureEngineer
# ---------------------------------------------------------------------------

class F1FeatureEngineer:
    """
    Reads cleaned Parquet files and constructs two feature datasets.

    Parameters
    ----------
    processed_dir : directory containing cleaned Parquet files
    features_dir  : output directory for feature Parquet files
    """

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DIR,
        features_dir:  Path = FEATURES_DIR,
    ) -> None:
        self.processed_dir = processed_dir
        self.features_dir  = features_dir
        self.features_dir.mkdir(parents=True, exist_ok=True)

        log.info("F1FeatureEngineer initialised")
        log.info("  processed_dir : %s", self.processed_dir)
        log.info("  features_dir  : %s", self.features_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the full feature engineering pipeline.

        Returns
        -------
        (driver_features, race_features)
        """
        log.info("=" * 60)
        log.info("F1FeatureEngineer pipeline — START")
        log.info("=" * 60)

        # ── Load cleaned datasets ─────────────────────────────────────
        results  = _read(self.processed_dir / "results.parquet",  "results")
        laps     = _read(self.processed_dir / "laps.parquet",     "laps")
        pitstops = _read(self.processed_dir / "pitstops.parquet", "pitstops")
        weather  = _read(self.processed_dir / "weather.parquet",  "weather")

        if results.empty or laps.empty:
            log.error("results or laps dataset is empty — cannot build features.")
            return pd.DataFrame(), pd.DataFrame()

        # ── Accurate lap filter (B5) — reused across multiple features ─
        accurate_laps = self._accurate_laps(laps)

        # ── Build feature blocks ──────────────────────────────────────
        log.info("── Building strategy features")
        strategy    = self._strategy_features(pitstops, laps)

        log.info("── Building performance features")
        performance = self._performance_features(results, accurate_laps)

        log.info("── Building tire features")
        tire        = self._tire_features(laps, pitstops)

        log.info("── Building advanced strategy metrics")
        advanced    = self._advanced_strategy_features(pitstops, laps, results)

        log.info("── Building race context features")
        context     = self._race_context_features(results, laps, weather)

        # ── Merge all driver-level blocks ─────────────────────────────
        log.info("── Merging driver feature blocks")
        driver_features = self._merge_driver_features(
            results, strategy, performance, tire, advanced, context
        )

        # ── Build race-level aggregate features ───────────────────────
        log.info("── Building race-level features")
        race_features = self._race_level_features(driver_features, context)

        # ── Save ──────────────────────────────────────────────────────
        _write(driver_features, self.features_dir / "driver_features.parquet",
               "driver_features")
        _write(race_features,   self.features_dir / "race_features.parquet",
               "race_features")

        log.info("=" * 60)
        log.info("F1FeatureEngineer pipeline — COMPLETE")
        log.info("  driver_features rows : %d", len(driver_features))
        log.info("  race_features rows   : %d", len(race_features))
        log.info("=" * 60)

        return driver_features, race_features

    # ------------------------------------------------------------------
    # Accurate lap filter
    # ------------------------------------------------------------------

    def _accurate_laps(self, laps: pd.DataFrame) -> pd.DataFrame:
        """
        Return only laps that are suitable for pace analysis (B5).

        Filters
        -------
        - is_accurate = True   (FastF1 timing quality flag)
        - lap_time_outlier = False  (our outlier flag from cleaner)
        - lap_time_s is not NaN
        """
        if laps.empty:
            return laps

        mask = pd.Series(True, index=laps.index)

        if "is_accurate" in laps.columns:
            mask &= laps["is_accurate"].astype(bool)

        if "lap_time_outlier" in laps.columns:
            mask &= ~laps["lap_time_outlier"].astype(bool)

        if "lap_time_s" in laps.columns:
            mask &= laps["lap_time_s"].notna()

        filtered = laps[mask].copy()
        log.info(
            "Accurate laps: %d / %d (%.1f%%)",
            len(filtered), len(laps), 100 * len(filtered) / max(len(laps), 1),
        )
        return filtered

    # ------------------------------------------------------------------
    # 1. Strategy features
    # ------------------------------------------------------------------

    def _strategy_features(
        self,
        pitstops: pd.DataFrame,
        laps:     pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute strategy features at driver × race granularity.

        Returns DataFrame indexed on (season, round_number, driver_code).
        """
        if pitstops.empty:
            log.warning("Pit stops empty — strategy features will be NaN.")
            return pd.DataFrame()

        grp_cols = ["season", "round_number", "driver_code"]

        # ── Stop count → strategy type ────────────────────────────────
        stop_counts = (
            pitstops.groupby(grp_cols)["stop_number"]
            .max()
            .reset_index()
            .rename(columns={"stop_number": "total_stops"})
        )
        stop_counts["strategy_type"] = stop_counts["total_stops"].apply(
            self._label_strategy
        )

        # ── Total pit time ────────────────────────────────────────────
        total_pit = (
            pitstops.groupby(grp_cols)["pit_duration_s"]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={"pit_duration_s": "total_pit_time_s"})
        )

        # ── First and last pit lap ────────────────────────────────────
        first_pit = (
            pitstops.groupby(grp_cols)["lap_number"]
            .min()
            .reset_index()
            .rename(columns={"lap_number": "first_pit_lap"})
        )
        last_pit = (
            pitstops.groupby(grp_cols)["lap_number"]
            .max()
            .reset_index()
            .rename(columns={"lap_number": "last_pit_lap"})
        )

        # ── Average stint length ──────────────────────────────────────
        if not laps.empty and "stint_number" in laps.columns:
            stint_lengths = (
                laps.groupby(grp_cols + ["stint_number"])["lap_number"]
                .count()
                .reset_index()
                .rename(columns={"lap_number": "stint_laps"})
            )
            avg_stint = (
                stint_lengths.groupby(grp_cols)["stint_laps"]
                .mean()
                .reset_index()
                .rename(columns={"stint_laps": "avg_stint_length"})
            )
        else:
            avg_stint = pd.DataFrame(columns=grp_cols + ["avg_stint_length"])

        # ── Merge strategy blocks ─────────────────────────────────────
        result = stop_counts.copy()
        for df_part in [total_pit, first_pit, last_pit, avg_stint]:
            if not df_part.empty:
                result = result.merge(df_part, on=grp_cols, how="left")

        log.info("Strategy features: %d driver-race rows", len(result))
        return result

    @staticmethod
    def _label_strategy(total_stops: float) -> str:
        """Map total pit stop count to a human-readable strategy label."""
        if pd.isna(total_stops):
            return "Unknown"
        n = int(total_stops)
        if n == 0:
            return "0-stop"
        if n == 1:
            return "1-stop"
        if n == 2:
            return "2-stop"
        return "3+-stop"

    # ------------------------------------------------------------------
    # 2. Performance features
    # ------------------------------------------------------------------

    def _performance_features(
        self,
        results:       pd.DataFrame,
        accurate_laps: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute pace and position performance features.

        Returns DataFrame at driver × race granularity.
        """
        grp_cols = ["season", "round_number", "driver_code"]

        # ── Position gain (from results) ──────────────────────────────
        perf = results[
            [c for c in
             ["season", "round_number", "driver_code",
              "grid_position", "finish_position", "points"]
             if c in results.columns]
        ].copy()

        if "grid_position" in perf.columns and "finish_position" in perf.columns:
            perf["position_gain"] = perf["grid_position"] - perf["finish_position"]

        if accurate_laps.empty:
            log.warning("No accurate laps — pace features will be NaN.")
            return perf

        # ── Average race pace ─────────────────────────────────────────
        avg_pace = (
            accurate_laps.groupby(grp_cols)["lap_time_s"]
            .mean()
            .reset_index()
            .rename(columns={"lap_time_s": "avg_race_pace_s"})
        )

        # ── Fastest lap ───────────────────────────────────────────────
        fastest = (
            accurate_laps.groupby(grp_cols)["lap_time_s"]
            .min()
            .reset_index()
            .rename(columns={"lap_time_s": "fastest_lap_s"})
        )

        # ── Pace consistency (std dev) ────────────────────────────────
        consistency = (
            accurate_laps.groupby(grp_cols)["lap_time_s"]
            .std()
            .reset_index()
            .rename(columns={"lap_time_s": "pace_consistency_s"})
        )

        # ── Best stint pace (median lap time of fastest stint) ────────
        best_stint_pace = self._compute_best_stint_pace(accurate_laps, grp_cols)

        # ── Merge ─────────────────────────────────────────────────────
        for df_part in [avg_pace, fastest, consistency, best_stint_pace]:
            if not df_part.empty:
                perf = perf.merge(df_part, on=grp_cols, how="left")

        log.info("Performance features: %d driver-race rows", len(perf))
        return perf

    @staticmethod
    def _compute_best_stint_pace(
        accurate_laps: pd.DataFrame,
        grp_cols:      list[str],
    ) -> pd.DataFrame:
        """
        For each driver-race, find the stint with the lowest median lap
        time and return that median as best_stint_pace_s.
        """
        if "stint_number" not in accurate_laps.columns:
            return pd.DataFrame(columns=grp_cols + ["best_stint_pace_s"])

        stint_pace = (
            accurate_laps.groupby(grp_cols + ["stint_number"])["lap_time_s"]
            .median()
            .reset_index()
            .rename(columns={"lap_time_s": "stint_median_pace"})
        )
        best = (
            stint_pace.groupby(grp_cols)["stint_median_pace"]
            .min()
            .reset_index()
            .rename(columns={"stint_median_pace": "best_stint_pace_s"})
        )
        return best

    # ------------------------------------------------------------------
    # 3. Tire features
    # ------------------------------------------------------------------

    def _tire_features(
        self,
        laps:     pd.DataFrame,
        pitstops: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute tyre-specific features per driver per race.
        """
        grp_cols = ["season", "round_number", "driver_code"]

        if laps.empty:
            return pd.DataFrame(columns=grp_cols)

        # ── Compound usage — ordered list ──────────────────────────────
        compound_usage = self._compute_compound_usage(laps, grp_cols)

        # ── Average tyre life ─────────────────────────────────────────
        # tyre_life at the last lap of each stint = laps on that set.
        if "tyre_life" in laps.columns and "stint_number" in laps.columns:
            stint_last = (
                laps.sort_values(grp_cols + ["stint_number", "lap_number"])
                .groupby(grp_cols + ["stint_number"])
                .last()
                .reset_index()
            )
            avg_tyre_life = (
                stint_last.groupby(grp_cols)["tyre_life"]
                .mean()
                .reset_index()
                .rename(columns={"tyre_life": "avg_tyre_life"})
            )
            longest_stint = (
                stint_last.groupby(grp_cols)["tyre_life"]
                .max()
                .reset_index()
                .rename(columns={"tyre_life": "longest_stint"})
            )
        else:
            avg_tyre_life = pd.DataFrame(columns=grp_cols + ["avg_tyre_life"])
            longest_stint = pd.DataFrame(columns=grp_cols + ["longest_stint"])

        # ── Number of distinct dry compounds used ─────────────────────
        num_compounds = self._compute_num_compounds(laps, grp_cols)

        # ── Tire degradation rate ─────────────────────────────────────
        deg_rate = self._compute_degradation_rate(laps, grp_cols)

        # ── Merge ─────────────────────────────────────────────────────
        result = compound_usage.copy()
        for df_part in [avg_tyre_life, longest_stint, num_compounds, deg_rate]:
            if not df_part.empty:
                result = result.merge(df_part, on=grp_cols, how="left")

        log.info("Tire features: %d driver-race rows", len(result))
        return result

    @staticmethod
    def _compute_compound_usage(
        laps:     pd.DataFrame,
        grp_cols: list[str],
    ) -> pd.DataFrame:
        """
        Build an ordered compound string per driver-race.

        We take the first observation of each stint's compound (sorted by
        lap number) and join them: e.g. "MEDIUM,HARD".
        """
        if "compound" not in laps.columns or "stint_number" not in laps.columns:
            return pd.DataFrame(columns=grp_cols + ["compound_usage"])

        # Coerce Categorical → str
        laps = laps.copy()
        laps["compound"] = laps["compound"].astype(str)

        stint_compounds = (
            laps.sort_values(grp_cols + ["stint_number", "lap_number"])
            .groupby(grp_cols + ["stint_number"])["compound"]
            .first()
            .reset_index()
        )

        def _join_compounds(group: pd.DataFrame) -> str:
            compounds = (
                group.sort_values("stint_number")["compound"]
                .str.upper()
                .tolist()
            )
            return ",".join(c for c in compounds if c not in ("NAN", "UNKNOWN"))

        usage = (
            stint_compounds.groupby(grp_cols)
            .apply(_join_compounds)
            .reset_index()
            .rename(columns={0: "compound_usage"})
        )
        return usage

    @staticmethod
    def _compute_num_compounds(
        laps:     pd.DataFrame,
        grp_cols: list[str],
    ) -> pd.DataFrame:
        """Count distinct dry compounds used per driver per race."""
        if "compound" not in laps.columns:
            return pd.DataFrame(columns=grp_cols + ["num_compounds_used"])

        laps = laps.copy()
        laps["compound"] = laps["compound"].astype(str).str.upper()
        dry = laps[laps["compound"].isin(DRY_COMPOUNDS)]

        if dry.empty:
            return pd.DataFrame(columns=grp_cols + ["num_compounds_used"])

        result = (
            dry.groupby(grp_cols)["compound"]
            .nunique()
            .reset_index()
            .rename(columns={"compound": "num_compounds_used"})
        )
        return result

    def _compute_degradation_rate(
        self,
        laps:     pd.DataFrame,
        grp_cols: list[str],
    ) -> pd.DataFrame:
        """
        Estimate tyre degradation rate per driver per race (B1).

        Method
        ------
        For each driver-race-stint (dry compounds only, tyre_life >= 3),
        fit OLS: lap_time_s ~ tyre_life.  The slope is the degradation
        rate in seconds per lap.  Slopes across all stints are averaged.

        Returns a DataFrame with column tire_degradation_rate.
        """
        if laps.empty:
            return pd.DataFrame(columns=grp_cols + ["tire_degradation_rate"])

        required = {"lap_time_s", "tyre_life", "compound", "stint_number"}
        if not required.issubset(laps.columns):
            log.warning("Degradation rate: missing columns %s", required - set(laps.columns))
            return pd.DataFrame(columns=grp_cols + ["tire_degradation_rate"])

        laps = laps.copy()
        laps["compound"] = laps["compound"].astype(str).str.upper()

        # Filter to valid degradation laps (B1)
        mask = (
            laps["compound"].isin(DRY_COMPOUNDS)
            & laps["lap_time_s"].notna()
            & laps["tyre_life"].notna()
            & (laps["tyre_life"] >= MIN_TYRE_AGE_FOR_DEG)
        )
        if "is_accurate" in laps.columns:
            mask &= laps["is_accurate"].astype(bool)
        if "lap_time_outlier" in laps.columns:
            mask &= ~laps["lap_time_outlier"].astype(bool)

        deg_laps = laps[mask]
        if deg_laps.empty:
            return pd.DataFrame(columns=grp_cols + ["tire_degradation_rate"])

        stint_cols = grp_cols + ["stint_number"]
        slopes: list[dict] = []

        for keys, group in deg_laps.groupby(stint_cols):
            slope = _ols_slope(group["tyre_life"], group["lap_time_s"])
            row   = dict(zip(stint_cols, keys if isinstance(keys, tuple) else (keys,)))
            row["stint_slope"] = slope
            slopes.append(row)

        if not slopes:
            return pd.DataFrame(columns=grp_cols + ["tire_degradation_rate"])

        slopes_df = pd.DataFrame(slopes)
        result = (
            slopes_df.groupby(grp_cols)["stint_slope"]
            .mean()
            .reset_index()
            .rename(columns={"stint_slope": "tire_degradation_rate"})
        )
        return result

    # ------------------------------------------------------------------
    # 4. Advanced strategy metrics
    # ------------------------------------------------------------------

    def _advanced_strategy_features(
        self,
        pitstops: pd.DataFrame,
        laps:     pd.DataFrame,
        results:  pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute undercut/overcut flags and pit stop efficiency.
        """
        grp_cols = ["season", "round_number", "driver_code"]

        efficiency   = self._compute_pit_efficiency(pitstops)
        undercut_ov  = self._compute_undercut_overcut(pitstops)

        result = efficiency.copy() if not efficiency.empty else pd.DataFrame(columns=grp_cols)
        if not undercut_ov.empty:
            if result.empty:
                result = undercut_ov
            else:
                result = result.merge(undercut_ov, on=grp_cols, how="outer")

        log.info("Advanced strategy features: %d driver-race rows", len(result))
        return result

    @staticmethod
    def _compute_pit_efficiency(pitstops: pd.DataFrame) -> pd.DataFrame:
        """
        Pit stop efficiency = driver avg pit duration / field avg pit duration
        per race.

        Values < 1.0 → faster than average crew.
        Values > 1.0 → slower than average crew.
        """
        grp_cols = ["season", "round_number", "driver_code"]

        if pitstops.empty or "pit_duration_s" not in pitstops.columns:
            return pd.DataFrame(columns=grp_cols + ["pit_stop_efficiency"])

        # Valid pit durations only
        valid = pitstops[pitstops["pit_duration_s"].notna()].copy()
        if valid.empty:
            return pd.DataFrame(columns=grp_cols + ["pit_stop_efficiency"])

        # Driver average
        driver_avg = (
            valid.groupby(grp_cols)["pit_duration_s"]
            .mean()
            .reset_index()
            .rename(columns={"pit_duration_s": "driver_avg_pit_s"})
        )

        # Race average (all drivers)
        race_avg = (
            valid.groupby(["season", "round_number"])["pit_duration_s"]
            .mean()
            .reset_index()
            .rename(columns={"pit_duration_s": "race_avg_pit_s"})
        )

        result = driver_avg.merge(race_avg, on=["season", "round_number"], how="left")
        result["pit_stop_efficiency"] = (
            result["driver_avg_pit_s"] / result["race_avg_pit_s"]
        ).round(4)

        return result[grp_cols + ["pit_stop_efficiency"]]

    @staticmethod
    def _compute_undercut_overcut(pitstops: pd.DataFrame) -> pd.DataFrame:
        """
        Detect undercut and overcut attempts (B2).

        Logic
        -----
        For each pit stop, look for other drivers in the same race who
        pitted within ±STRATEGY_RIVAL_LAP_WINDOW laps.

        Undercut attempt : this driver pitted BEFORE the rival
                           (rival_lap − own_lap > 0)
        Overcut attempt  : this driver pitted AFTER the rival
                           (own_lap − rival_lap > 0)

        If any such rival exists for any stop, the flag is set True.

        Note: we do not check whether the manoeuvre succeeded (i.e. whether
        position was actually gained) because finish position changes also
        depend on pit stop duration, lap pace, and traffic.  A more complex
        analysis comparing pre- and post-stop positions would require
        lap-level position data joined with pit stop data — suitable for
        the SQL / EDA phase.
        """
        grp_cols = ["season", "round_number", "driver_code"]

        if pitstops.empty or "lap_number" not in pitstops.columns:
            return pd.DataFrame(columns=grp_cols + ["undercut_attempt", "overcut_attempt"])

        records: list[dict] = []

        for (season, rnd), race_stops in pitstops.groupby(["season", "round_number"]):
            drivers = race_stops["driver_code"].unique()

            for driver in drivers:
                own_stops   = race_stops[race_stops["driver_code"] == driver]["lap_number"].values
                rival_stops = race_stops[race_stops["driver_code"] != driver]["lap_number"].values

                undercut = False
                overcut  = False

                for own_lap in own_stops:
                    for rival_lap in rival_stops:
                        gap = rival_lap - own_lap
                        if 0 < gap <= STRATEGY_RIVAL_LAP_WINDOW:
                            undercut = True
                        elif 0 < -gap <= STRATEGY_RIVAL_LAP_WINDOW:
                            overcut = True

                records.append({
                    "season":           season,
                    "round_number":     rnd,
                    "driver_code":      driver,
                    "undercut_attempt": undercut,
                    "overcut_attempt":  overcut,
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 5. Race context features
    # ------------------------------------------------------------------

    def _race_context_features(
        self,
        results: pd.DataFrame,
        laps:    pd.DataFrame,
        weather: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build race-level context: wet flag, circuit type, safety car.
        Returns one row per race (season, round_number).
        """
        race_cols = ["season", "round_number", "event_name", "circuit"]

        # ── Base from results ─────────────────────────────────────────
        available = [c for c in race_cols if c in results.columns]
        context   = results[available].drop_duplicates().copy()

        # ── Wet race flag ─────────────────────────────────────────────
        if "wet_race" in results.columns:
            wet = (
                results.groupby(["season", "round_number"])["wet_race"]
                .any()
                .reset_index()
            )
            context = context.merge(wet, on=["season", "round_number"], how="left")
        elif "wet_race" in laps.columns:
            wet = (
                laps.groupby(["season", "round_number"])["wet_race"]
                .any()
                .reset_index()
            )
            context = context.merge(wet, on=["season", "round_number"], how="left")
        else:
            context["wet_race"] = False

        # ── Circuit type ──────────────────────────────────────────────
        if "circuit" in context.columns:
            context["circuit_type"] = context["circuit"].apply(
                lambda c: "street" if str(c) in STREET_CIRCUITS else "permanent"
            )
        else:
            context["circuit_type"] = "permanent"

        # ── Safety car deployed ───────────────────────────────────────
        context["safety_car_deployed"] = False
        if not laps.empty and "track_status" in laps.columns:
            sc_laps = laps[
                laps["track_status"].astype(str).apply(
                    lambda s: any(code in s for code in SAFETY_CAR_STATUS_CODES)
                )
            ]
            if not sc_laps.empty:
                sc_races = (
                    sc_laps.groupby(["season", "round_number"])
                    .size()
                    .reset_index()[["season", "round_number"]]
                    .assign(safety_car_deployed=True)
                )
                context = context.merge(
                    sc_races, on=["season", "round_number"], how="left",
                    suffixes=("", "_new")
                )
                if "safety_car_deployed_new" in context.columns:
                    context["safety_car_deployed"] = (
                        context["safety_car_deployed_new"].fillna(False)
                    )
                    context.drop(columns=["safety_car_deployed_new"], inplace=True)
            context["safety_car_deployed"] = context["safety_car_deployed"].fillna(False)

        log.info("Race context features: %d races", len(context))
        return context

    # ------------------------------------------------------------------
    # Merge driver-level blocks
    # ------------------------------------------------------------------

    def _merge_driver_features(
        self,
        results:     pd.DataFrame,
        strategy:    pd.DataFrame,
        performance: pd.DataFrame,
        tire:        pd.DataFrame,
        advanced:    pd.DataFrame,
        context:     pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left-join all driver-level feature blocks onto the results frame.

        results is the spine: one row per driver per race.
        """
        driver_cols = ["season", "round_number", "driver_code"]
        race_cols   = ["season", "round_number"]

        base = results.copy()

        for df_part, label in [
            (strategy,    "strategy"),
            (performance, "performance"),
            (tire,        "tire"),
            (advanced,    "advanced"),
        ]:
            if df_part is None or df_part.empty:
                log.warning("Skipping empty %s block in merge", label)
                continue

            # Identify the join key (driver or race level)
            join_on = [c for c in driver_cols if c in df_part.columns]
            if not join_on:
                continue

            # Drop columns that already exist in base to avoid _x/_y suffixes
            overlap = [c for c in df_part.columns
                       if c in base.columns and c not in join_on]
            df_part = df_part.drop(columns=overlap, errors="ignore")

            base = base.merge(df_part, on=join_on, how="left")
            log.debug("Merged %s — cols now: %d", label, len(base.columns))

        # Join race-level context (wet_race, circuit_type, safety_car_deployed)
        ctx_cols = [c for c in context.columns
                    if c not in base.columns or c in race_cols]
        ctx_subset = context[[c for c in ctx_cols
                               if c in context.columns]].drop_duplicates(race_cols)
        base = base.merge(ctx_subset, on=race_cols, how="left")

        # Remove any duplicate columns introduced during merging
        base = base.loc[:, ~base.columns.duplicated()]

        log.info("Driver features merged: %d rows × %d cols", len(base), len(base.columns))
        return base

    # ------------------------------------------------------------------
    # Race-level aggregates
    # ------------------------------------------------------------------

    def _race_level_features(
        self,
        driver_features: pd.DataFrame,
        context:         pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate driver-level features to produce one row per race.

        Includes: dominant strategy, avg pit count, avg position gain,
        field avg pace, safety car flag, wet flag, circuit type.
        """
        race_cols = ["season", "round_number"]

        agg_dict: dict[str, tuple] = {}

        if "total_stops" in driver_features.columns:
            agg_dict["avg_stops_per_driver"] = ("total_stops", "mean")

        if "position_gain" in driver_features.columns:
            agg_dict["avg_position_gain"]    = ("position_gain", "mean")
            agg_dict["max_position_gain"]    = ("position_gain", "max")

        if "avg_race_pace_s" in driver_features.columns:
            agg_dict["field_avg_pace_s"]     = ("avg_race_pace_s", "mean")
            agg_dict["field_fastest_lap_s"]  = ("fastest_lap_s", "min") \
                if "fastest_lap_s" in driver_features.columns \
                else ("avg_race_pace_s", "min")

        if "total_pit_time_s" in driver_features.columns:
            agg_dict["avg_total_pit_time_s"] = ("total_pit_time_s", "mean")
            agg_dict["min_total_pit_time_s"] = ("total_pit_time_s", "min")

        if "tire_degradation_rate" in driver_features.columns:
            agg_dict["avg_deg_rate"]         = ("tire_degradation_rate", "mean")

        if "pit_stop_efficiency" in driver_features.columns:
            agg_dict["best_pit_efficiency"]  = ("pit_stop_efficiency", "min")

        if agg_dict:
            race_agg = (
                driver_features.groupby(race_cols)
                .agg(**agg_dict)
                .reset_index()
            )
        else:
            race_agg = driver_features[race_cols].drop_duplicates().copy()

        # ── Dominant strategy per race ────────────────────────────────
        if "strategy_type" in driver_features.columns:
            dominant = (
                driver_features.groupby(race_cols)["strategy_type"]
                .agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else "Unknown")
                .reset_index()
                .rename(columns={"strategy_type": "dominant_strategy"})
            )
            race_agg = race_agg.merge(dominant, on=race_cols, how="left")

        # ── Join race context ─────────────────────────────────────────
        ctx_merge_cols = [c for c in context.columns
                          if c not in race_agg.columns or c in race_cols]
        race_agg = race_agg.merge(
            context[[c for c in ctx_merge_cols if c in context.columns]],
            on=race_cols, how="left",
        )

        race_agg = race_agg.loc[:, ~race_agg.columns.duplicated()]
        log.info("Race features: %d rows × %d cols", len(race_agg), len(race_agg.columns))
        return race_agg