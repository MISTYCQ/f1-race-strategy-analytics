"""
run_cleaning.py
===============
Entry-point script for the F1 Race Strategy Analytics cleaning pipeline.

Responsibilities
----------------
1. Configure logging (console + rotating file).
2. Parse command-line arguments.
3. Instantiate F1DataCleaner and run the full cleaning pipeline.
4. Print a formatted summary of quality metrics after completion.
5. Handle all exceptions gracefully — log and exit with a non-zero code.

Usage
-----
    # Run with default paths
    python run_cleaning.py

    # Custom output directories
    python run_cleaning.py --processed-dir data/processed --reports-dir reports

    # Suppress console output below WARNING
    python run_cleaning.py --quiet
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — makes project root importable from any working directory
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
        Format: concise single-line format for interactive use.

    File handler
        Level : DEBUG always — full detail for post-run diagnostics.
        File  : logs/cleaning_<date>.log  (10 MB cap, 5 rotating backups).

    Returns
    -------
    logging.Logger
        Named logger for this script ("run_cleaning").
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

    # Avoid duplicate handlers on repeated calls (common in notebooks).
    if root.handlers:
        root.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING if quiet else logging.INFO)
    ch.setFormatter(console_fmt)
    root.addHandler(ch)

    # Rotating file handler
    log_file = LOG_DIR / f"cleaning_{time.strftime('%Y-%m-%d')}.log"
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes    = 10 * 1024 * 1024,
        backupCount = 5,
        encoding    = "utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    root.addHandler(fh)

    return logging.getLogger("run_cleaning")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "run_cleaning.py",
        description = "Clean raw F1 Parquet files and write processed outputs.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog      = __doc__,
    )
    parser.add_argument(
        "--processed-dir",
        type    = Path,
        default = PROJECT_ROOT / "data" / "processed",
        metavar = "PATH",
        help    = "Output directory for cleaned Parquet files "
                  "(default: data/processed/)",
    )
    parser.add_argument(
        "--reports-dir",
        type    = Path,
        default = PROJECT_ROOT / "reports",
        metavar = "PATH",
        help    = "Output directory for quality_report.csv and "
                  "data_dictionary.csv (default: reports/)",
    )
    parser.add_argument(
        "--quiet",
        action  = "store_true",
        help    = "Suppress INFO messages on the console (errors still shown)",
    )
    return parser


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    report,
    elapsed_s:     float,
    processed_dir: Path,
    reports_dir:   Path,
) -> None:
    """
    Print a formatted summary box to stdout after the cleaning pipeline
    completes.

    Parameters
    ----------
    report        : CleaningReport returned by F1DataCleaner.run()
    elapsed_s     : wall-clock seconds the pipeline took
    processed_dir : where processed Parquet files were written
    reports_dir   : where CSV reports were written
    """
    mins, secs  = divmod(int(elapsed_s), 60)
    elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    width      = 56
    top        = "╔" + "═" * width + "╗"
    mid        = "╠" + "═" * width + "╣"
    bot        = "╚" + "═" * width + "╝"

    def row(text: str) -> str:
        return f"║  {text:<{width - 2}}║"

    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    # ── Per-dataset quality block ─────────────────────────────────────────
    dataset_lines: list[str] = []
    for dq in report.datasets:
        dataset_lines.append(row(f"  {dq.name}"))
        dataset_lines.append(row(f"    Rows          : {dq.row_count:>8,}"))
        dataset_lines.append(row(f"    Columns       : {dq.col_count:>8,}"))
        dataset_lines.append(row(f"    Duplicate rows: {dq.duplicate_rows:>8,}"))

        if dq.missing_pct:
            # Show the five columns with the highest missing percentage
            top5 = sorted(dq.missing_pct.items(), key=lambda x: x[1], reverse=True)[:5]
            dataset_lines.append(row("    Missing values (top 5 columns):"))
            for col, pct in top5:
                dataset_lines.append(row(f"      {col:<28} {pct:>6.2f}%"))

        if dq.invalid_counts:
            dataset_lines.append(row("    Flagged / invalid counts:"))
            for key, count in dq.invalid_counts.items():
                dataset_lines.append(row(f"      {key:<28} {count:>6,}"))

        if dq.assumptions_applied:
            dataset_lines.append(row("    Assumptions applied:"))
            for assumption in dq.assumptions_applied:
                # Wrap long assumption strings across two indented lines
                if len(assumption) <= width - 8:
                    dataset_lines.append(row(f"      • {assumption}"))
                else:
                    dataset_lines.append(row(f"      • {assumption[:width - 10]}…"))

    lines = [
        top,
        row("F1 Data Cleaning — Summary"),
        mid,
        row(f"Elapsed time      : {elapsed_str}"),
        row(f"Datasets cleaned  : {len(report.datasets)}"),
        row(f"Total races       : {report.total_races}"),
        row(f"  Dry races       : {report.dry_race_count}"),
        row(f"  Wet races       : {report.wet_race_count}"),
        mid,
        row("Per-Dataset Quality Metrics:"),
        *dataset_lines,
        mid,
        row("Output locations:"),
        row(f"  Processed data  : {rel(processed_dir)}/"),
        row(f"  Quality report  : {rel(reports_dir)}/quality_report.csv"),
        row(f"  Data dictionary : {rel(reports_dir)}/data_dictionary.csv"),
        bot,
    ]

    print("\n" + "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Orchestrate the cleaning pipeline.

    Return codes
    ------------
    0  All datasets cleaned and reports written successfully.
    1  Pipeline completed but with non-fatal warnings (check logs).
    2  Fatal error — pipeline did not complete.
    """
    parser = build_parser()
    args   = parser.parse_args()
    log    = configure_logging(quiet=args.quiet)

    log.info("run_cleaning.py starting")
    log.info(
        "Arguments: processed_dir=%s  reports_dir=%s  quiet=%s",
        args.processed_dir, args.reports_dir, args.quiet,
    )

    # ── Import cleaner after logging is configured so its module-level
    #    logger inherits the handlers we just set up on the root logger.
    try:
        from src.data.cleaner import F1DataCleaner
    except ImportError as exc:
        log.critical(
            "Could not import F1DataCleaner. "
            "Ensure you are running from the project root and all "
            "dependencies are installed.\n  Error: %s", exc,
        )
        return 2

    # ── Instantiate ───────────────────────────────────────────────────────
    try:
        cleaner = F1DataCleaner(
            processed_dir = args.processed_dir,
            reports_dir   = args.reports_dir,
        )
    except Exception as exc:
        log.critical("Failed to initialise F1DataCleaner: %s", exc, exc_info=True)
        return 2

    # ── Announce what we're about to do ───────────────────────────────────
    print(
        f"\n  F1 Data Cleaning Pipeline\n"
        f"  Processed dir : {args.processed_dir}\n"
        f"  Reports dir   : {args.reports_dir}\n"
    )

    # ── Run ───────────────────────────────────────────────────────────────
    start_time = time.monotonic()
    report     = None

    try:
        report = cleaner.run()

    except KeyboardInterrupt:
        log.warning("Cleaning pipeline interrupted by user.")
        print("\n  Interrupted — partial outputs may exist on disk.\n")
        return 2

    except MemoryError:
        log.critical(
            "Out of memory while cleaning. "
            "Try processing fewer seasons at once or increasing available RAM."
        )
        return 2

    except Exception as exc:
        log.critical(
            "Cleaning pipeline encountered a fatal error: %s", exc, exc_info=True
        )
        print(
            f"\n  FATAL: {exc}\n"
            f"  Full traceback written to: {LOG_DIR}/\n"
        )
        return 2

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.monotonic() - start_time

    print_summary(
        report        = report,
        elapsed_s     = elapsed,
        processed_dir = args.processed_dir,
        reports_dir   = args.reports_dir,
    )

    log.info(
        "Cleaning complete: %d datasets, %d total races, %.1f s elapsed",
        len(report.datasets), report.total_races, elapsed,
    )

    # Return 1 if any dataset had quality warnings (failed counts > 0)
    has_warnings = any(
        sum(dq.invalid_counts.values()) > 0
        for dq in report.datasets
    )
    return 1 if has_warnings else 0


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())