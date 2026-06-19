"""
run_features.py
===============
Entry-point script for the F1 Race Strategy Analytics feature engineering
pipeline.

Responsibilities
----------------
1. Configure logging (console + rotating file).
2. Parse command-line arguments.
3. Instantiate F1FeatureEngineer and run the full pipeline.
4. Print formatted summary statistics for both output datasets.
5. Handle all exceptions gracefully — log full tracebacks, exit cleanly.

Usage
-----
    # Default paths (data/processed → data/features)
    python run_features.py

    # Custom features output directory
    python run_features.py --features-dir data/features_v2

    # Custom processed input directory
    python run_features.py --processed-dir data/processed_v2

    # Both custom
    python run_features.py --processed-dir data/processed --features-dir data/features

    # Suppress INFO messages on console
    python run_features.py --quiet
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(quiet: bool = False) -> logging.Logger:
    """
    Configure the root logger with two handlers.

    Console handler
        Level : WARNING when --quiet is set, otherwise INFO.
        Format: concise timestamped single-line format.

    File handler
        Level : DEBUG always — full detail for diagnostics.
        File  : logs/features_<date>.log (10 MB cap, 5 rotating backups).

    Returns
    -------
    logging.Logger
        Named logger for this script ("run_features").
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    console_fmt = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt = "%H:%M:%S",
    )
    file_fmt = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if root.handlers:
        root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING if quiet else logging.INFO)
    ch.setFormatter(console_fmt)
    root.addHandler(ch)

    log_file = LOG_DIR / f"features_{time.strftime('%Y-%m-%d')}.log"
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes    = 10 * 1024 * 1024,
        backupCount = 5,
        encoding    = "utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    root.addHandler(fh)

    return logging.getLogger("run_features")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "run_features.py",
        description = "Run the F1 feature engineering pipeline.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog      = __doc__,
    )
    parser.add_argument(
        "--processed-dir",
        type    = Path,
        default = PROJECT_ROOT / "data" / "processed",
        metavar = "PATH",
        help    = "Directory containing cleaned Parquet files "
                  "(default: data/processed/)",
    )
    parser.add_argument(
        "--features-dir",
        type    = Path,
        default = PROJECT_ROOT / "data" / "features",
        metavar = "PATH",
        help    = "Output directory for feature Parquet files "
                  "(default: data/features/)",
    )
    parser.add_argument(
        "--quiet",
        action  = "store_true",
        help    = "Suppress INFO messages on console (errors still shown)",
    )
    return parser


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    driver_features,           # pd.DataFrame
    race_features,             # pd.DataFrame
    elapsed_s:   float,
    features_dir: Path,
) -> None:
    """
    Print a formatted summary box to stdout after the pipeline completes.

    Covers
    ------
    - Elapsed time
    - driver_features: shape, key feature distributions
    - race_features:   shape, dominant strategies, circuit types
    - Output file locations
    """
    import pandas as pd

    mins, secs  = divmod(int(elapsed_s), 60)
    elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    width = 58

    top = "╔" + "═" * width + "╗"
    mid = "╠" + "═" * width + "╣"
    bot = "╚" + "═" * width + "╝"

    def row(text: str = "") -> str:
        return f"║  {text:<{width - 2}}║"

    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    def fmt_val(v) -> str:
        """Format a scalar for display: round floats, handle NaN."""
        if pd.isna(v):
            return "N/A"
        if isinstance(v, float):
            return f"{v:,.3f}"
        return str(v)

    # ── Driver features block ─────────────────────────────────────────────
    df_lines: list[str] = [
        row("driver_features"),
        row(f"  Rows            : {len(driver_features):>8,}"),
        row(f"  Columns         : {len(driver_features.columns):>8,}"),
    ]

    # Strategy type distribution
    if "strategy_type" in driver_features.columns:
        counts = (
            driver_features["strategy_type"]
            .value_counts()
            .sort_index()
        )
        df_lines.append(row("  Strategy type breakdown:"))
        for strat, cnt in counts.items():
            pct = 100 * cnt / max(len(driver_features), 1)
            df_lines.append(row(f"    {str(strat):<12}  {cnt:>4}  ({pct:>5.1f}%)"))

    # Key numeric feature summaries
    numeric_features = {
        "position_gain":        "Position gain",
        "avg_race_pace_s":      "Avg race pace (s)",
        "pace_consistency_s":   "Pace consistency (s)",
        "tire_degradation_rate":"Tyre degradation (s/lap)",
        "total_pit_time_s":     "Total pit time (s)",
        "avg_stint_length":     "Avg stint length (laps)",
        "pit_stop_efficiency":  "Pit efficiency ratio",
    }
    available_numeric = {
        label: col
        for col, label in numeric_features.items()
        if col in driver_features.columns
    }
    if available_numeric:
        df_lines.append(row("  Key feature means:"))
        for label, col in available_numeric.items():
            mean_val = driver_features[col].mean()
            df_lines.append(row(f"    {label:<28} {fmt_val(mean_val):>10}"))

    # Undercut / overcut rates
    for flag_col, label in [
        ("undercut_attempt", "Undercut attempt rate"),
        ("overcut_attempt",  "Overcut attempt rate"),
    ]:
        if flag_col in driver_features.columns:
            rate = driver_features[flag_col].mean() * 100
            df_lines.append(row(f"  {label:<30} {rate:>6.1f}%"))

    # Wet race breakdown
    if "wet_race" in driver_features.columns:
        wet_drivers = driver_features["wet_race"].sum()
        total       = len(driver_features)
        df_lines.append(
            row(f"  Wet race entries        : {wet_drivers:>4} / {total}")
        )

    # ── Race features block ───────────────────────────────────────────────
    rf_lines: list[str] = [
        row(),
        row("race_features"),
        row(f"  Rows            : {len(race_features):>8,}"),
        row(f"  Columns         : {len(race_features.columns):>8,}"),
    ]

    # Dominant strategy distribution
    if "dominant_strategy" in race_features.columns:
        dom = race_features["dominant_strategy"].value_counts().sort_index()
        rf_lines.append(row("  Dominant strategy per race:"))
        for strat, cnt in dom.items():
            pct = 100 * cnt / max(len(race_features), 1)
            rf_lines.append(row(f"    {str(strat):<12}  {cnt:>3} races  ({pct:>5.1f}%)"))

    # Circuit type breakdown
    if "circuit_type" in race_features.columns:
        ct = race_features["circuit_type"].value_counts()
        rf_lines.append(row("  Circuit type:"))
        for ctype, cnt in ct.items():
            rf_lines.append(row(f"    {str(ctype):<12}  {cnt:>3} races"))

    # Safety car deployment rate
    if "safety_car_deployed" in race_features.columns:
        sc_count = race_features["safety_car_deployed"].sum()
        sc_pct   = 100 * sc_count / max(len(race_features), 1)
        rf_lines.append(
            row(f"  Safety car deployed     : {int(sc_count):>3} races ({sc_pct:.1f}%)")
        )

    # Race-level numeric summaries
    race_numeric = {
        "avg_stops_per_driver":  "Avg stops / driver",
        "field_avg_pace_s":      "Field avg pace (s)",
        "avg_total_pit_time_s":  "Avg total pit time (s)",
        "avg_deg_rate":          "Avg degradation (s/lap)",
    }
    available_race_numeric = {
        label: col
        for col, label in race_numeric.items()
        if col in race_features.columns
    }
    if available_race_numeric:
        rf_lines.append(row("  Key race-level means:"))
        for label, col in available_race_numeric.items():
            mean_val = race_features[col].mean()
            rf_lines.append(row(f"    {label:<28} {fmt_val(mean_val):>10}"))

    # Season breakdown
    if "season" in race_features.columns:
        seasons = race_features["season"].value_counts().sort_index()
        rf_lines.append(row("  Races per season:"))
        for season, cnt in seasons.items():
            rf_lines.append(row(f"    {season}  →  {cnt} races"))

    # ── Output locations ──────────────────────────────────────────────────
    out_lines: list[str] = [
        row(),
        row("Output files:"),
        row(f"  driver_features → {rel(features_dir)}/driver_features.parquet"),
        row(f"  race_features   → {rel(features_dir)}/race_features.parquet"),
        row(f"  Log             → {rel(LOG_DIR)}/features_{time.strftime('%Y-%m-%d')}.log"),
    ]

    lines = [
        top,
        row("F1 Feature Engineering — Summary"),
        row(f"Elapsed time : {elapsed_str}"),
        mid,
        *df_lines,
        *rf_lines,
        mid,
        *out_lines,
        bot,
    ]

    print("\n" + "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Feature validation
# ---------------------------------------------------------------------------

def validate_outputs(
    driver_features,    # pd.DataFrame
    race_features,      # pd.DataFrame
    log: logging.Logger,
) -> list[str]:
    """
    Run lightweight sanity checks on the output DataFrames.

    Returns a list of warning strings (empty = all checks passed).
    These are printed to the console and written to the log but do not
    cause the script to exit with a non-zero code.
    """
    warnings: list[str] = []

    if driver_features.empty:
        warnings.append("driver_features is empty — no features were produced.")
        return warnings

    if race_features.empty:
        warnings.append("race_features is empty — no race-level features produced.")

    # Expect at least these columns in driver_features
    expected_driver = [
        "season", "round_number", "driver_code",
        "strategy_type", "position_gain", "avg_race_pace_s",
        "tire_degradation_rate", "compound_usage",
    ]
    missing = [c for c in expected_driver if c not in driver_features.columns]
    if missing:
        warnings.append(f"driver_features missing expected columns: {missing}")

    # Check for a reasonable number of rows (at least 10 drivers × 1 race)
    if len(driver_features) < 10:
        warnings.append(
            f"driver_features has only {len(driver_features)} rows — "
            "expected at least 10 (one per driver per race)."
        )

    # strategy_type should not be all null
    if "strategy_type" in driver_features.columns:
        null_pct = driver_features["strategy_type"].isna().mean() * 100
        if null_pct > 50:
            warnings.append(
                f"strategy_type is null for {null_pct:.1f}% of rows — "
                "check that pitstops.parquet contains data."
            )

    # Degradation rate sanity: values should typically be between -1 and +5 s/lap
    if "tire_degradation_rate" in driver_features.columns:
        deg = driver_features["tire_degradation_rate"].dropna()
        if len(deg) > 0:
            extreme = ((deg < -2) | (deg > 10)).sum()
            if extreme > 0:
                warnings.append(
                    f"tire_degradation_rate has {extreme} extreme values "
                    "(outside [-2, 10] s/lap) — may indicate data quality issues."
                )

    for w in warnings:
        log.warning("Validation: %s", w)

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Orchestrate the feature engineering pipeline.

    Return codes
    ------------
    0  Pipeline completed, all validation checks passed.
    1  Pipeline completed but validation warnings were raised.
    2  Fatal error — pipeline did not complete.
    """
    parser = build_parser()
    args   = parser.parse_args()
    log    = configure_logging(quiet=args.quiet)

    log.info("run_features.py starting")
    log.info(
        "Arguments: processed_dir=%s  features_dir=%s  quiet=%s",
        args.processed_dir, args.features_dir, args.quiet,
    )

    # ── Validate input directory exists before doing anything ─────────────
    if not args.processed_dir.exists():
        log.critical(
            "processed_dir does not exist: %s\n"
            "  Run run_cleaning.py first to generate cleaned Parquet files.",
            args.processed_dir,
        )
        print(
            f"\n  ERROR: processed_dir not found: {args.processed_dir}\n"
            f"  Run 'python run_cleaning.py' first.\n"
        )
        return 2

    # ── Check expected input files are present ────────────────────────────
    required_files = ["results.parquet", "laps.parquet",
                      "pitstops.parquet", "weather.parquet"]
    missing_files  = [
        f for f in required_files
        if not (args.processed_dir / f).exists()
    ]
    if missing_files:
        log.critical(
            "Missing required input files in %s: %s",
            args.processed_dir, missing_files,
        )
        print(
            f"\n  ERROR: Missing input files: {missing_files}\n"
            f"  Ensure run_cleaning.py completed successfully.\n"
        )
        return 2

    # ── Import feature engineer ───────────────────────────────────────────
    try:
        from src.features.feature_engineering import F1FeatureEngineer
    except ImportError as exc:
        log.critical(
            "Could not import F1FeatureEngineer. "
            "Ensure you are running from the project root and all "
            "dependencies are installed.\n  Error: %s", exc,
        )
        print(
            f"\n  ERROR: Import failed — {exc}\n"
            f"  Run from the project root: python run_features.py\n"
        )
        return 2

    # ── Announce ──────────────────────────────────────────────────────────
    print(
        f"\n  F1 Feature Engineering Pipeline\n"
        f"  Processed dir : {args.processed_dir}\n"
        f"  Features dir  : {args.features_dir}\n"
    )

    # ── Instantiate ───────────────────────────────────────────────────────
    try:
        engineer = F1FeatureEngineer(
            processed_dir = args.processed_dir,
            features_dir  = args.features_dir,
        )
    except Exception as exc:
        log.critical("Failed to initialise F1FeatureEngineer: %s", exc, exc_info=True)
        print(f"\n  ERROR: Initialisation failed — {exc}\n")
        return 2

    # ── Run ───────────────────────────────────────────────────────────────
    start_time      = time.monotonic()
    driver_features = None
    race_features   = None

    try:
        driver_features, race_features = engineer.run()

    except KeyboardInterrupt:
        log.warning("Feature engineering interrupted by user.")
        print("\n  Interrupted — partial outputs may exist on disk.\n")
        return 2

    except MemoryError:
        log.critical(
            "Out of memory during feature engineering. "
            "Try reducing the number of seasons or laps loaded at once."
        )
        print(
            "\n  ERROR: Out of memory.\n"
            "  Try processing fewer seasons or increasing available RAM.\n"
        )
        return 2

    except Exception as exc:
        log.critical(
            "Feature engineering pipeline encountered a fatal error: %s",
            exc, exc_info=True,
        )
        print(
            f"\n  FATAL: {exc}\n"
            f"  Full traceback written to: {LOG_DIR}/\n"
        )
        return 2

    elapsed = time.monotonic() - start_time

    # ── Guard against None returns (should not happen, but be safe) ───────
    if driver_features is None or race_features is None:
        log.error("engineer.run() returned None — check feature engineering logs.")
        print("\n  ERROR: Pipeline returned no data. Check logs.\n")
        return 2

    # ── Validation ────────────────────────────────────────────────────────
    warnings = validate_outputs(driver_features, race_features, log)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(
        driver_features = driver_features,
        race_features   = race_features,
        elapsed_s       = elapsed,
        features_dir    = args.features_dir,
    )

    log.info(
        "Feature engineering complete: driver_features=%d rows, "
        "race_features=%d rows, elapsed=%.1fs, warnings=%d",
        len(driver_features), len(race_features), elapsed, len(warnings),
    )

    return 1 if warnings else 0


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())