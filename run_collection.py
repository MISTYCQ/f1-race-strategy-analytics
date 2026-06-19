"""
run_collection.py
=================
Command-line entry point for the F1 data collection pipeline.

Examples
--------
# Collect default seasons (2023 + 2024):
    python run_collection.py

# Collect only 2024, overwriting existing files:
    python run_collection.py --seasons 2024 --overwrite

# Preview what would be collected without downloading anything:
    python run_collection.py --dry-run

# Collect a single race (season 2023, round 7) for quick testing:
    python run_collection.py --seasons 2023 --round 7
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from config.settings import SEASONS
from src.data.collector import F1DataCollector, RaceIdentifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download F1 race data using the FastF1 API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=SEASONS,
        help=f"Season years to collect (default: {SEASONS})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download data even if Parquet files already exist",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        dest="single_round",
        help="Collect a single round number only — useful for testing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print what would be collected without downloading anything",
    )
    return parser.parse_args()


def dry_run(seasons: list[int]) -> None:
    """Print the race schedule for each season without downloading."""
    import fastf1
    for season in seasons:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        races = schedule[schedule["RoundNumber"] > 0]
        print(f"\nSeason {season} — {len(races)} races:")
        for _, row in races.iterrows():
            print(f"  R{int(row['RoundNumber']):02d}  {row['EventName']}")


def collect_single_round(
    collector: F1DataCollector,
    season: int,
    round_number: int,
) -> None:
    """Collect exactly one round from one season."""
    import fastf1
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    row = schedule[schedule["RoundNumber"] == round_number]

    if row.empty:
        logger.error(f"Round {round_number} not found in season {season}")
        return

    event = row.iloc[0]
    race_id = RaceIdentifier(
        season       = season,
        round_number = int(event["RoundNumber"]),
        event_name   = str(event["EventName"]),
        country      = str(event.get("Country", "Unknown")),
        circuit      = str(event.get("Location", "Unknown")),
    )
    result = collector._collect_race(race_id)
    collector._results.append(result)


def main() -> int:
    args = parse_args()

    if args.dry_run:
        dry_run(args.seasons)
        return 0

    collector = F1DataCollector(
        seasons   = args.seasons,
        overwrite = args.overwrite,
    )

    if args.single_round is not None:
        logger.info(f"Single-round mode: round {args.single_round}")
        for season in args.seasons:
            collect_single_round(collector, season, args.single_round)
        summary = collector.get_summary()
        collector._log_summary(summary)
    else:
        summary = collector.run()

    # Exit 0 = all succeeded or skipped. Exit 1 = at least one failure.
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())