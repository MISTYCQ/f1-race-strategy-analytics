"""
run_analysis.py
===============
Entry-point script for the F1 Race Strategy Analytics analysis pipeline.

Responsibilities
----------------
1. Configure logging (console + rotating file).
2. Parse command-line arguments.
3. Instantiate F1StrategyAnalyser and run all five analyses.
4. Export key insight CSV tables to reports/tables/.
5. Print a formatted insights summary to stdout.
6. Handle all exceptions gracefully.

Usage
-----
    # Default paths
    python run_analysis.py

    # Custom directories
    python run_analysis.py --features-dir data/features --figures-dir reports/figures

    # Quiet mode
    python run_analysis.py --quiet
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

LOG_DIR    = PROJECT_ROOT / "logs"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(quiet: bool = False) -> logging.Logger:
    """
    Configure the root logger with two handlers.

    Console handler
        Level : WARNING when --quiet, otherwise INFO.

    File handler
        Level : DEBUG always.
        File  : logs/analysis_<date>.log  (10 MB cap, 5 backups).

    Returns
    -------
    logging.Logger named "run_analysis".
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    console_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if root.handlers:
        root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING if quiet else logging.INFO)
    ch.setFormatter(console_fmt)
    root.addHandler(ch)

    log_file = LOG_DIR / f"analysis_{time.strftime('%Y-%m-%d')}.log"
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    root.addHandler(fh)

    return logging.getLogger("run_analysis")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_analysis.py",
        description="Run F1 strategy analyses, save figures and CSV tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "features",
        metavar="PATH",
        help="Directory containing feature Parquet files (default: data/features/)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures",
        metavar="PATH",
        help="Output directory for PNG figures (default: reports/figures/)",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=TABLES_DIR,
        metavar="PATH",
        help="Output directory for CSV tables (default: reports/tables/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO messages on console (errors still shown)",
    )
    return parser


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_features(
    features_dir: Path,
    log: logging.Logger,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Load driver_features.parquet and race_features.parquet.

    Returns
    -------
    (driver_features, race_features) — either may be empty if file missing.
    """
    import pandas as pd

    def _read(name: str) -> pd.DataFrame:
        path = features_dir / name
        if not path.exists():
            log.warning("Feature file not found: %s", path)
            return pd.DataFrame()
        df = pd.read_parquet(path, engine="pyarrow")
        log.info("Loaded %-30s  rows=%d  cols=%d", name, len(df), len(df.columns))
        return df

    return _read("driver_features.parquet"), _read("race_features.parquet")


# ---------------------------------------------------------------------------
# CSV insight exporter
# ---------------------------------------------------------------------------

def export_insight_tables(
    driver:     "pd.DataFrame",
    race:       "pd.DataFrame",
    tables_dir: Path,
    log:        logging.Logger,
) -> dict[str, Path]:
    """
    Derive five insight tables and save each as a CSV file.

    Tables
    ------
    q1_strategy_win_rate.csv       win rate, podium rate, avg points per strategy
    q2_team_position_gain.csv      avg / median / std position gain per team
    q3_circuit_strategy_share.csv  strategy usage % per circuit
    q4_compound_degradation.csv    degradation rate + tyre life per compound
    q5_team_pit_efficiency.csv     pit stop efficiency ratio per team

    Returns
    -------
    dict mapping label → saved CSV Path.
    """
    import pandas as pd

    tables_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    def _write(df: pd.DataFrame, filename: str, label: str) -> None:
        path = tables_dir / filename
        df.to_csv(path, index=True)
        log.info("Saved CSV  %-38s → %s", label, path.relative_to(PROJECT_ROOT))
        saved[label] = path

    # ── Q1: strategy win rate ─────────────────────────────────────────────
    try:
        if {"strategy_type", "finish_position"}.issubset(driver.columns):
            df = driver.copy()
            df["is_win"]    = df["finish_position"] == 1
            df["is_podium"] = df["finish_position"] <= 3
            df["is_points"] = df["finish_position"] <= 10

            agg: dict[str, tuple] = {
                "entries":           ("is_win",    "count"),
                "wins":              ("is_win",    "sum"),
                "win_rate_pct":      ("is_win",    lambda x: round(x.mean() * 100, 2)),
                "podiums":           ("is_podium", "sum"),
                "podium_rate_pct":   ("is_podium", lambda x: round(x.mean() * 100, 2)),
                "points_finishes":   ("is_points", "sum"),
            }
            if "points" in df.columns:
                agg["avg_points"]   = ("points", lambda x: round(x.mean(), 2))
                agg["total_points"] = ("points", "sum")

            q1 = (
                df.groupby("strategy_type")
                .agg(**agg)
                .sort_values("win_rate_pct", ascending=False)
            )
            _write(q1, "q1_strategy_win_rate.csv", "q1_strategy_win_rate")
    except Exception as exc:
        log.error("Q1 table export failed: %s", exc, exc_info=True)

    # ── Q2: team position gain ────────────────────────────────────────────
    try:
        if {"team_name", "position_gain"}.issubset(driver.columns):
            q2 = (
                driver.dropna(subset=["position_gain", "team_name"])
                .groupby("team_name")["position_gain"]
                .agg(
                    entries           = "count",
                    avg_gain          = lambda x: round(x.mean(), 3),
                    median_gain       = lambda x: round(x.median(), 3),
                    std_gain          = lambda x: round(x.std(), 3),
                    max_gain          = "max",
                    min_gain          = "min",
                    pct_gained_places = lambda x: round((x > 0).mean() * 100, 1),
                    pct_lost_places   = lambda x: round((x < 0).mean() * 100, 1),
                )
                .sort_values("avg_gain", ascending=False)
            )
            _write(q2, "q2_team_position_gain.csv", "q2_team_position_gain")
    except Exception as exc:
        log.error("Q2 table export failed: %s", exc, exc_info=True)

    # ── Q3: circuit strategy share ────────────────────────────────────────
    try:
        if {"event_name", "strategy_type"}.issubset(driver.columns):
            counts = (
                driver.dropna(subset=["strategy_type", "event_name"])
                .groupby(["event_name", "strategy_type"])
                .size()
                .unstack(fill_value=0)
            )
            pct = counts.div(counts.sum(axis=1), axis=0).mul(100).round(1)
            pct.columns = [f"{c}_pct" for c in pct.columns]
            for col in counts.columns:
                pct[f"{col}_count"] = counts[col]
            pct["total_drivers"] = counts.sum(axis=1)

            sort_col = "1-stop_pct" if "1-stop_pct" in pct.columns else pct.columns[0]
            q3 = pct.sort_values(sort_col, ascending=False)
            _write(q3, "q3_circuit_strategy_share.csv", "q3_circuit_strategy_share")
    except Exception as exc:
        log.error("Q3 table export failed: %s", exc, exc_info=True)

    # ── Q4: compound degradation ──────────────────────────────────────────
    try:
        deg_col = "tire_degradation_rate"
        if {deg_col, "compound_usage"}.issubset(driver.columns):
            df4 = driver.dropna(subset=[deg_col]).copy()
            df4["first_compound"] = (
                df4["compound_usage"].astype(str)
                .str.split(",").str[0].str.strip().str.upper()
            )
            dry = df4[df4["first_compound"].isin(["SOFT", "MEDIUM", "HARD"])]

            agg4: dict[str, tuple] = {
                "entries":      (deg_col, "count"),
                "avg_deg_rate": (deg_col, lambda x: round(x.mean(), 5)),
                "med_deg_rate": (deg_col, lambda x: round(x.median(), 5)),
                "std_deg_rate": (deg_col, lambda x: round(x.std(), 5)),
                "min_deg_rate": (deg_col, "min"),
                "max_deg_rate": (deg_col, "max"),
            }
            if "avg_tyre_life" in driver.columns:
                agg4["avg_tyre_life"] = (
                    "avg_tyre_life", lambda x: round(x.mean(), 1)
                )
            if "longest_stint" in driver.columns:
                agg4["avg_longest_stint"] = (
                    "longest_stint", lambda x: round(x.mean(), 1)
                )

            q4 = (
                dry.groupby("first_compound")
                .agg(**agg4)
                .reindex(["SOFT", "MEDIUM", "HARD"])
                .dropna(how="all")
            )
            _write(q4, "q4_compound_degradation.csv", "q4_compound_degradation")
    except Exception as exc:
        log.error("Q4 table export failed: %s", exc, exc_info=True)

    # ── Q5: team pit stop efficiency ──────────────────────────────────────
    try:
        eff_col = "pit_stop_efficiency"
        if {eff_col, "team_name"}.issubset(driver.columns):
            df5 = (
                driver.dropna(subset=[eff_col, "team_name"])
                .copy()
                .loc[lambda x: x[eff_col].between(0.5, 2.0)]
            )
            agg5: dict[str, tuple] = {
                "entries":             (eff_col, "count"),
                "avg_efficiency":      (eff_col, lambda x: round(x.mean(), 4)),
                "median_efficiency":   (eff_col, lambda x: round(x.median(), 4)),
                "std_efficiency":      (eff_col, lambda x: round(x.std(), 4)),
                "best_efficiency":     (eff_col, "min"),
                "pct_faster_than_avg": (eff_col, lambda x: round((x < 1.0).mean() * 100, 1)),
            }
            if "total_pit_time_s" in driver.columns:
                agg5["avg_total_pit_time_s"] = (
                    "total_pit_time_s", lambda x: round(x.mean(), 2)
                )
            q5 = (
                df5.groupby("team_name")
                .agg(**agg5)
                .sort_values("avg_efficiency")
            )
            _write(q5, "q5_team_pit_efficiency.csv", "q5_team_pit_efficiency")
    except Exception as exc:
        log.error("Q5 table export failed: %s", exc, exc_info=True)

    log.info("CSV export complete — %d / 5 tables saved", len(saved))
    return saved


# ---------------------------------------------------------------------------
# Insights printer
# ---------------------------------------------------------------------------

def print_insights(
    driver:      "pd.DataFrame",
    race:        "pd.DataFrame",
    saved_figs:  dict[str, Path],
    saved_csvs:  dict[str, Path],
    elapsed_s:   float,
    figures_dir: Path,
    tables_dir:  Path,
) -> None:
    """
    Derive and print key findings for all five business questions, plus
    file locations and run metadata.
    """
    mins, secs  = divmod(int(elapsed_s), 60)
    elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    width = 64
    top   = "╔" + "═" * width + "╗"
    mid   = "╠" + "═" * width + "╣"
    bot   = "╚" + "═" * width + "╝"

    def row(text: str = "") -> str:
        return f"║  {text:<{width - 2}}║"

    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    insight_rows: list[str] = []

    # ── Q1: Strategy win rate ─────────────────────────────────────────────
    try:
        if {"strategy_type", "finish_position"}.issubset(driver.columns):
            df = driver.copy()
            df["is_win"] = df["finish_position"] == 1
            wr = (
                df.groupby("strategy_type")["is_win"]
                .mean()
                .sort_values(ascending=False)
            )
            best_strat = wr.index[0]
            best_rate  = wr.iloc[0] * 100
            n_wins     = int(df[df["strategy_type"] == best_strat]["is_win"].sum())
            n_entries  = int((df["strategy_type"] == best_strat).sum())

            insight_rows += [
                row("Q1 · Which pit strategy wins most often?"),
                row(f"     Winner  : {best_strat}"),
                row(f"     Rate    : {best_rate:.1f}%  ({n_wins} wins from {n_entries} entries)"),
            ]
            for strat, rate in wr.items():
                insight_rows.append(row(f"       {str(strat):<12}  {rate * 100:>5.1f}% win rate"))
        else:
            insight_rows.append(row("Q1 · Strategy win rate: data unavailable"))
    except Exception as exc:
        insight_rows.append(row(f"Q1 · Error: {exc}"))

    insight_rows.append(row())

    # ── Q2: Team position gain ────────────────────────────────────────────
    try:
        if {"team_name", "position_gain"}.issubset(driver.columns):
            tg = (
                driver.dropna(subset=["position_gain", "team_name"])
                .groupby("team_name")["position_gain"]
                .mean()
                .sort_values(ascending=False)
            )
            best_team  = tg.index[0]
            best_gain  = tg.iloc[0]
            worst_team = tg.index[-1]
            worst_gain = tg.iloc[-1]

            insight_rows += [
                row("Q2 · Which teams gain the most positions?"),
                row(f"     Biggest gainer : {best_team}  (avg {best_gain:+.2f} places)"),
                row(f"     Biggest loser  : {worst_team}  (avg {worst_gain:+.2f} places)"),
                row("     Full ranking:"),
            ]
            for team, gain in tg.items():
                bar_len = int(abs(gain) * 4)
                bar     = ("▶" * bar_len) if gain >= 0 else ("◀" * bar_len)
                insight_rows.append(
                    row(f"       {str(team):<28}  {gain:+.2f}  {bar}")
                )
        else:
            insight_rows.append(row("Q2 · Team position gain: data unavailable"))
    except Exception as exc:
        insight_rows.append(row(f"Q2 · Error: {exc}"))

    insight_rows.append(row())

    # ── Q3: Circuit strategy ──────────────────────────────────────────────
    try:
        if {"event_name", "strategy_type"}.issubset(driver.columns):
            pivot = (
                driver.dropna(subset=["strategy_type", "event_name"])
                .groupby(["event_name", "strategy_type"])
                .size()
                .unstack(fill_value=0)
            )
            pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100)

            insight_rows.append(row("Q3 · Which circuits favour one-stop vs two-stop?"))

            if "1-stop" in pct.columns:
                most_1stop  = pct["1-stop"].idxmax()
                least_1stop = pct["1-stop"].idxmin()
                val_most    = pct["1-stop"].max()
                val_least   = pct["1-stop"].min()
                insight_rows += [
                    row(f"     Most 1-stop  : {most_1stop}"),
                    row(f"       → {val_most:.1f}% of drivers used 1-stop"),
                    row(f"     Least 1-stop : {least_1stop}"),
                    row(f"       → {val_least:.1f}% of drivers used 1-stop"),
                ]

            if "2-stop" in pct.columns:
                most_2stop = pct["2-stop"].idxmax()
                val_2stop  = pct["2-stop"].max()
                insight_rows += [
                    row(f"     Most 2-stop  : {most_2stop}"),
                    row(f"       → {val_2stop:.1f}% of drivers used 2-stop"),
                ]
        else:
            insight_rows.append(row("Q3 · Circuit strategy: data unavailable"))
    except Exception as exc:
        insight_rows.append(row(f"Q3 · Error: {exc}"))

    insight_rows.append(row())

    # ── Q4: Compound degradation ──────────────────────────────────────────
    try:
        deg_col = "tire_degradation_rate"
        if {deg_col, "compound_usage"}.issubset(driver.columns):
            df4 = driver.dropna(subset=[deg_col]).copy()
            df4["first_compound"] = (
                df4["compound_usage"].astype(str)
                .str.split(",").str[0].str.strip().str.upper()
            )
            dry     = df4[df4["first_compound"].isin(["SOFT", "MEDIUM", "HARD"])]
            comp_dg = dry.groupby("first_compound")[deg_col].mean().sort_values()

            insight_rows.append(row("Q4 · Which tyre compounds degrade slowest?"))
            if not comp_dg.empty:
                best_comp = comp_dg.index[0]
                best_val  = comp_dg.iloc[0]
                insight_rows += [
                    row(f"     Lowest degradation : {best_comp}  ({best_val:.5f} s/lap)"),
                    row("     Degradation rates:"),
                ]
                for comp, val in comp_dg.items():
                    bar_len = min(int(val * 2000), 20)
                    bar     = "█" * bar_len
                    insight_rows.append(
                        row(f"       {str(comp):<14}  {val:.5f} s/lap  {bar}")
                    )

            if "avg_tyre_life" in driver.columns:
                life = dry.groupby("first_compound")["avg_tyre_life"].mean().sort_values(
                    ascending=False
                )
                if not life.empty:
                    insight_rows.append(row("     Average tyre life:"))
                    for comp, laps in life.items():
                        insight_rows.append(row(f"       {str(comp):<14}  {laps:.1f} laps"))
        else:
            insight_rows.append(row("Q4 · Compound degradation: data unavailable"))
    except Exception as exc:
        insight_rows.append(row(f"Q4 · Error: {exc}"))

    insight_rows.append(row())

    # ── Q5: Team pit stop efficiency ──────────────────────────────────────
    try:
        eff_col = "pit_stop_efficiency"
        if {eff_col, "team_name"}.issubset(driver.columns):
            df5 = (
                driver.dropna(subset=[eff_col, "team_name"])
                .loc[lambda x: x[eff_col].between(0.5, 2.0)]
            )
            eff = (
                df5.groupby("team_name")[eff_col]
                .mean()
                .sort_values()
            )
            fastest_team  = eff.index[0]
            fastest_val   = eff.iloc[0]
            pct_faster    = (1 - fastest_val) * 100
            slowest_team  = eff.index[-1]
            slowest_val   = eff.iloc[-1]
            pct_slower    = (slowest_val - 1) * 100

            insight_rows += [
                row("Q5 · Which teams execute the fastest pit stops?"),
                row(f"     Fastest crew : {fastest_team}"),
                row(f"       → {pct_faster:.1f}% faster than field average"),
                row(f"     Slowest crew : {slowest_team}"),
                row(f"       → {pct_slower:.1f}% slower than field average"),
                row("     Efficiency ranking  (1.0 = field avg):"),
            ]
            for team, val in eff.items():
                diff    = val - 1.0
                marker  = "▼" if diff < 0 else "▲"
                insight_rows.append(
                    row(f"       {str(team):<28}  {val:.4f}  {marker}{abs(diff)*100:.1f}%")
                )
        else:
            insight_rows.append(row("Q5 · Pit stop efficiency: data unavailable"))
    except Exception as exc:
        insight_rows.append(row(f"Q5 · Error: {exc}"))

    insight_rows.append(row())

    # ── Additional race context stats ─────────────────────────────────────
    try:
        context_rows: list[str] = [row("Race Context:")]
        if "wet_race" in driver.columns:
            wet_pct = driver["wet_race"].mean() * 100
            context_rows.append(row(f"  Wet race entries       : {wet_pct:.1f}%"))
        if "safety_car_deployed" in race.columns:
            sc_pct = race["safety_car_deployed"].mean() * 100
            context_rows.append(row(f"  Races with safety car  : {sc_pct:.1f}%"))
        if "circuit_type" in race.columns:
            ct = race["circuit_type"].value_counts()
            for ctype, cnt in ct.items():
                context_rows.append(row(f"  {str(ctype):<22} circuits: {cnt}"))
        if "season" in race.columns:
            for season, cnt in race["season"].value_counts().sort_index().items():
                context_rows.append(row(f"  Season {season}               : {cnt} races"))
        insight_rows += context_rows
    except Exception:
        pass

    # ── File output summary ───────────────────────────────────────────────
    file_rows: list[str] = [
        row(),
        row("Output files:"),
        row(f"  Figures  ({len(saved_figs)}) → {rel(figures_dir)}/"),
    ]
    for label, path in saved_figs.items():
        file_rows.append(row(f"    • {path.name}"))

    file_rows.append(row(f"  Tables   ({len(saved_csvs)}) → {rel(tables_dir)}/"))
    for label, path in saved_csvs.items():
        file_rows.append(row(f"    • {path.name}"))

    file_rows += [
        row(),
        row(f"  Log → {rel(LOG_DIR)}/analysis_{time.strftime('%Y-%m-%d')}.log"),
    ]

    lines = [
        top,
        row("F1 Race Strategy Analytics — Key Insights"),
        row(f"Elapsed : {elapsed_str}   |   "
            f"Seasons: 2023–2024   |   "
            f"Figures: {len(saved_figs)}   |   "
            f"Tables: {len(saved_csvs)}"),
        mid,
        *insight_rows,
        mid,
        *file_rows,
        bot,
    ]

    print("\n" + "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Orchestrate the full analysis pipeline.

    Return codes
    ------------
    0  All analyses ran and figures + CSVs were saved.
    1  Pipeline ran but some analyses failed — partial output saved.
    2  Fatal error — pipeline did not complete.
    """
    parser = build_parser()
    args   = parser.parse_args()
    log    = configure_logging(quiet=args.quiet)

    log.info("run_analysis.py starting")
    log.info(
        "Args: features=%s  figures=%s  tables=%s",
        args.features_dir, args.figures_dir, args.tables_dir,
    )

    # ── Validate input directory ──────────────────────────────────────────
    if not args.features_dir.exists():
        log.critical(
            "features_dir not found: %s — run run_features.py first.",
            args.features_dir,
        )
        print(
            f"\n  ERROR: features_dir not found: {args.features_dir}\n"
            f"  Run 'python run_features.py' first.\n"
        )
        return 2

    required = ["driver_features.parquet", "race_features.parquet"]
    missing  = [f for f in required if not (args.features_dir / f).exists()]
    if missing:
        log.critical("Missing feature files: %s", missing)
        print(f"\n  ERROR: Missing feature files: {missing}\n")
        return 2

    # ── Import analyser ───────────────────────────────────────────────────
    try:
        from src.analysis.strategy_analysis import F1StrategyAnalyser
    except ImportError as exc:
        log.critical("Could not import F1StrategyAnalyser: %s", exc)
        print(f"\n  ERROR: Import failed — {exc}\n"
              f"  Run from the project root: python run_analysis.py\n")
        return 2

    # ── Load feature data for CSV export and insight printing ─────────────
    driver, race = load_features(args.features_dir, log)

    if driver.empty:
        log.critical("driver_features is empty — cannot run analysis.")
        print("\n  ERROR: driver_features.parquet is empty or unreadable.\n")
        return 2

    # ── Announce ──────────────────────────────────────────────────────────
    print(
        f"\n  F1 Race Strategy Analytics Pipeline\n"
        f"  Features dir : {args.features_dir}\n"
        f"  Figures dir  : {args.figures_dir}\n"
        f"  Tables dir   : {args.tables_dir}\n"
    )

    # ── Instantiate analyser ──────────────────────────────────────────────
    try:
        analyser = F1StrategyAnalyser(
            features_dir=args.features_dir,
            figures_dir=args.figures_dir,
        )
    except Exception as exc:
        log.critical("Failed to initialise F1StrategyAnalyser: %s", exc, exc_info=True)
        print(f"\n  ERROR: Initialisation failed — {exc}\n")
        return 2

    # ── Run analyses ──────────────────────────────────────────────────────
    start_time = time.monotonic()
    saved_figs: dict[str, Path] = {}

    try:
        saved_figs = analyser.run()
    except KeyboardInterrupt:
        log.warning("Analysis interrupted by user.")
        print("\n  Interrupted — partial figures may exist on disk.\n")
        return 2
    except Exception as exc:
        log.critical("Analysis pipeline failed: %s", exc, exc_info=True)
        print(f"\n  FATAL: {exc}\n  See logs/ for full traceback.\n")
        return 2

    # ── Export CSV tables ─────────────────────────────────────────────────
    saved_csvs: dict[str, Path] = {}
    try:
        saved_csvs = export_insight_tables(driver, race, args.tables_dir, log)
    except Exception as exc:
        log.error("CSV export failed: %s", exc, exc_info=True)

    # ── Print insights ────────────────────────────────────────────────────
    elapsed = time.monotonic() - start_time

    try:
        print_insights(
            driver      = driver,
            race        = race,
            saved_figs  = saved_figs,
            saved_csvs  = saved_csvs,
            elapsed_s   = elapsed,
            figures_dir = args.figures_dir,
            tables_dir  = args.tables_dir,
        )
    except Exception as exc:
        log.error("Could not print insights: %s", exc, exc_info=True)
        print(f"\n  Analysis complete — {len(saved_figs)} figures, "
              f"{len(saved_csvs)} tables saved.\n")

    log.info(
        "Analysis complete: figures=%d  tables=%d  elapsed=%.1fs",
        len(saved_figs), len(saved_csvs), elapsed,
    )

    any_failed = len(saved_figs) < 5 or len(saved_csvs) < 5
    return 1 if any_failed else 0


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())