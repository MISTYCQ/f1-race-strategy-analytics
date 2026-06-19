"""
src/analysis/strategy_analysis.py
===================================
F1StrategyAnalyser — reads driver_features.parquet and race_features.parquet
and produces publication-quality matplotlib figures that answer five core
business questions about F1 race strategy.

Business questions answered
---------------------------
Q1  Which pit strategy wins most often?
Q2  Which teams gain the most positions?
Q3  Which circuits favour one-stop vs two-stop strategies?
Q4  Which tyre compounds have the lowest degradation rate?
Q5  Which teams execute the fastest pit stops?

Output
------
reports/figures/
    q1_strategy_win_rate.png
    q2_team_position_gain.png
    q3_circuit_strategy_heatmap.png
    q4_compound_degradation.png
    q5_team_pit_stop_speed.png
    summary_dashboard.png          ← all five panels on one canvas

Design notes
------------
- Every figure uses a consistent dark F1-themed style.
- All text, tick labels, and annotations are sized for A4 / slide export.
- No seaborn dependency — pure matplotlib + numpy.
- Figures are saved at 150 dpi (screen) and 300 dpi (print) quality;
  150 dpi is used here for fast export while remaining sharp at A4 size.
- Each analyser method is fully self-contained so it can be called
  independently in a notebook without running the whole pipeline.

Usage
-----
    from src.analysis.strategy_analysis import F1StrategyAnalyser

    analyser = F1StrategyAnalyser()
    analyser.run()                   # produce all figures
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts & CI

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
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
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"
LOGS_DIR     = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Visual style — F1 dark theme
# ---------------------------------------------------------------------------

# Base colours
C_BG        = "#0f0f0f"   # figure background
C_PANEL     = "#1a1a1a"   # axes background
C_GRID      = "#2e2e2e"   # gridline colour
C_TEXT      = "#f0f0f0"   # primary text
C_SUBTEXT   = "#9a9a9a"   # secondary text / tick labels
C_ACCENT    = "#e8002d"   # F1 red — primary highlight
C_ACCENT2   = "#ffffff"   # white — secondary highlight
C_GOLD      = "#ffd700"   # gold — winner highlight

# Strategy palette  (consistent across all charts)
STRATEGY_COLOURS: dict[str, str] = {
    "0-stop":  "#5b8dd9",
    "1-stop":  "#e8002d",
    "2-stop":  "#ffd700",
    "3+-stop": "#9b59b6",
    "Unknown": "#555555",
}

# Compound palette  (mirrors real F1 tyre colours)
COMPOUND_COLOURS: dict[str, str] = {
    "SOFT":         "#e8002d",
    "MEDIUM":       "#ffd700",
    "HARD":         "#f0f0f0",
    "INTERMEDIATE": "#3cb44b",
    "WET":          "#4363d8",
    "UNKNOWN":      "#555555",
}

# Team colour map  (official 2023/24 approximate hex values)
TEAM_COLOURS: dict[str, str] = {
    "Red Bull Racing":    "#3671c6",
    "Mercedes":           "#27f4d2",
    "Ferrari":            "#e8002d",
    "McLaren":            "#ff8000",
    "Aston Martin":       "#229971",
    "Alpine":             "#0093cc",
    "Williams":           "#64c4ff",
    "AlphaTauri":         "#5e8faa",
    "RB":                 "#5e8faa",
    "Alfa Romeo":         "#c92d4b",
    "Haas F1 Team":       "#b6babd",
    "Sauber":             "#00e701",
}

DPI      = 150
FIGSIZE  = (14, 8)    # inches — fits a 16:9 slide
FONTSIZE = 11


def _team_colour(team: str) -> str:
    """Return the official team colour or a neutral grey for unknowns."""
    for key, colour in TEAM_COLOURS.items():
        if key.lower() in team.lower():
            return colour
    return "#888888"


def _apply_base_style() -> None:
    """Apply the F1 dark theme to all subsequent matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor":     C_BG,
        "axes.facecolor":       C_PANEL,
        "axes.edgecolor":       C_GRID,
        "axes.labelcolor":      C_TEXT,
        "axes.titlecolor":      C_TEXT,
        "axes.grid":            True,
        "axes.axisbelow":       True,
        "grid.color":           C_GRID,
        "grid.linewidth":       0.6,
        "text.color":           C_TEXT,
        "xtick.color":          C_SUBTEXT,
        "ytick.color":          C_SUBTEXT,
        "xtick.labelsize":      FONTSIZE - 1,
        "ytick.labelsize":      FONTSIZE - 1,
        "axes.labelsize":       FONTSIZE,
        "axes.titlesize":       FONTSIZE + 2,
        "figure.titlesize":     FONTSIZE + 4,
        "legend.facecolor":     "#1e1e1e",
        "legend.edgecolor":     C_GRID,
        "legend.labelcolor":    C_TEXT,
        "legend.fontsize":      FONTSIZE - 1,
        "savefig.facecolor":    C_BG,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.3,
        "font.family":          "DejaVu Sans",
    })


def _save(fig: plt.Figure, path: Path, label: str) -> None:
    """Save *fig* to *path* at 150 dpi and log the action."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    log.info("Saved %-40s → %s", label, path.relative_to(PROJECT_ROOT))


def _add_watermark(fig: plt.Figure) -> None:
    """Add a subtle data-source watermark to the bottom-right corner."""
    fig.text(
        0.99, 0.01,
        "Source: FastF1 API  |  F1 Strategy Analytics",
        ha="right", va="bottom",
        fontsize=FONTSIZE - 3,
        color=C_SUBTEXT,
        alpha=0.6,
    )


def _subtitle(ax: plt.Axes, text: str) -> None:
    """Add a grey subtitle below the axes title."""
    ax.set_title(text, color=C_SUBTEXT, fontsize=FONTSIZE - 1, pad=2)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(name: str = "f1_strategy_analyser") -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt_c = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )
    fmt_f = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_c)
    logger.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        LOGS_DIR / f"analysis_{time.strftime('%Y-%m-%d')}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_f)
    logger.addHandler(fh)
    return logger


log = _build_logger()


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def _load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        log.warning("File not found — returning empty frame: %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path, engine="pyarrow")
    log.info("Loaded %-20s  rows=%6d  cols=%d", label, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# F1StrategyAnalyser
# ---------------------------------------------------------------------------

class F1StrategyAnalyser:
    """
    Reads feature Parquet files and produces five publication-quality
    matplotlib figures plus a combined summary dashboard.

    Parameters
    ----------
    features_dir : directory containing driver_features.parquet and
                   race_features.parquet
    figures_dir  : output directory for PNG exports
    """

    def __init__(
        self,
        features_dir: Path = FEATURES_DIR,
        figures_dir:  Path = FIGURES_DIR,
    ) -> None:
        self.features_dir = features_dir
        self.figures_dir  = figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        _apply_base_style()

        log.info("F1StrategyAnalyser initialised")
        log.info("  features_dir : %s", self.features_dir)
        log.info("  figures_dir  : %s", self.figures_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Path]:
        """
        Execute all five analyses and save figures.

        Returns
        -------
        dict mapping analysis label → saved figure Path.
        """
        log.info("=" * 60)
        log.info("F1StrategyAnalyser — START")
        log.info("=" * 60)

        driver = _load(self.features_dir / "driver_features.parquet", "driver_features")
        race   = _load(self.features_dir / "race_features.parquet",   "race_features")

        if driver.empty:
            log.error("driver_features is empty — cannot run analysis.")
            return {}

        saved: dict[str, Path] = {}

        analyses = [
            ("q1_strategy_win_rate",       self.plot_strategy_win_rate),
            ("q2_team_position_gain",      self.plot_team_position_gain),
            ("q3_circuit_strategy_heatmap",self.plot_circuit_strategy_heatmap),
            ("q4_compound_degradation",    self.plot_compound_degradation),
            ("q5_team_pit_stop_speed",     self.plot_team_pit_stop_speed),
        ]

        figs_for_dashboard: list[plt.Figure] = []

        for key, method in analyses:
            log.info("── Running %s", key)
            try:
                fig = method(driver, race)
                if fig is not None:
                    path = self.figures_dir / f"{key}.png"
                    _save(fig, path, key)
                    saved[key] = path
                    figs_for_dashboard.append(fig)
            except Exception as exc:
                log.error("Failed %s: %s", key, exc, exc_info=True)

        # Combined dashboard
        log.info("── Building summary dashboard")
        try:
            dash_path = self.figures_dir / "summary_dashboard.png"
            self._build_dashboard(driver, race, dash_path)
            saved["summary_dashboard"] = dash_path
        except Exception as exc:
            log.error("Failed to build dashboard: %s", exc, exc_info=True)

        log.info("=" * 60)
        log.info("Analysis complete — %d figures saved to %s",
                 len(saved), self.figures_dir.relative_to(PROJECT_ROOT))
        log.info("=" * 60)
        return saved

    # ------------------------------------------------------------------
    # Q1 — Which pit strategy wins most often?
    # ------------------------------------------------------------------

    def plot_strategy_win_rate(
        self,
        driver: pd.DataFrame,
        race:   pd.DataFrame,
    ) -> Optional[plt.Figure]:
        """
        Two-panel figure:
          Left  — Win rate (% of race wins) by strategy type.
          Right — Points scored per strategy type (box plot).

        Business insight
        ----------------
        Win rate tells us which strategy the fastest drivers chose, but
        it conflates strategy quality with car performance.  The points
        box plot adds depth: a 2-stop with a wide interquartile range
        suggests it works brilliantly sometimes but fails at others,
        whereas a 1-stop with a tight IQR is reliable but rarely wins.
        """
        if "strategy_type" not in driver.columns:
            log.warning("Q1: strategy_type column missing.")
            return None

        df = driver.copy()

        # A win is finish_position == 1
        if "finish_position" not in df.columns:
            log.warning("Q1: finish_position column missing.")
            return None

        df["is_win"] = df["finish_position"] == 1

        strategy_order = ["0-stop", "1-stop", "2-stop", "3+-stop"]
        present = [s for s in strategy_order if s in df["strategy_type"].unique()]

        # Win rate per strategy
        win_stats = (
            df.groupby("strategy_type")
            .agg(
                total_entries = ("is_win", "count"),
                wins          = ("is_win", "sum"),
            )
            .loc[lambda x: x.index.isin(present)]
            .reindex(present)
        )
        win_stats["win_rate_pct"] = (
            win_stats["wins"] / win_stats["total_entries"] * 100
        ).round(1)

        # Points per strategy — for box plot
        points_by_strategy: dict[str, np.ndarray] = {}
        if "points" in df.columns:
            for s in present:
                pts = df.loc[df["strategy_type"] == s, "points"].dropna().values
                if len(pts) > 0:
                    points_by_strategy[s] = pts

        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=FIGSIZE,
            gridspec_kw={"width_ratios": [1.2, 1]},
        )
        fig.suptitle(
            "Q1 — Which Pit Strategy Wins Most Often?",
            fontsize=FONTSIZE + 5, fontweight="bold", y=1.01,
        )
        _add_watermark(fig)

        # ── Left: win rate bar chart ──────────────────────────────────
        colours = [STRATEGY_COLOURS.get(s, "#888888") for s in present]
        bars = ax_left.barh(
            present,
            win_stats["win_rate_pct"].values,
            color=colours,
            edgecolor=C_BG,
            linewidth=0.8,
            height=0.55,
        )

        # Annotate bars with win count and win rate
        for bar, strat in zip(bars, present):
            w = bar.get_width()
            wins  = int(win_stats.loc[strat, "wins"])
            total = int(win_stats.loc[strat, "total_entries"])
            ax_left.text(
                w + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%  ({wins}/{total})",
                va="center", ha="left",
                fontsize=FONTSIZE - 1, color=C_TEXT,
            )

        ax_left.set_xlabel("Win Rate (%)", color=C_TEXT)
        ax_left.set_xlim(0, win_stats["win_rate_pct"].max() * 1.35)
        ax_left.invert_yaxis()
        ax_left.set_title("Win Rate by Strategy", color=C_TEXT,
                           fontsize=FONTSIZE + 1, fontweight="bold")
        _subtitle(ax_left, "% of race entries that resulted in a win")
        ax_left.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax_left.grid(axis="y", alpha=0)

        # Highlight the best strategy
        best_idx = win_stats["win_rate_pct"].idxmax()
        best_bar_idx = present.index(best_idx)
        bars[best_bar_idx].set_edgecolor(C_GOLD)
        bars[best_bar_idx].set_linewidth(2.5)

        # ── Right: points box plot ────────────────────────────────────
        if points_by_strategy:
            bp = ax_right.boxplot(
                [points_by_strategy[s] for s in present if s in points_by_strategy],
                vert=True,
                patch_artist=True,
                widths=0.5,
                medianprops=dict(color=C_GOLD, linewidth=2.5),
                whiskerprops=dict(color=C_SUBTEXT, linewidth=1.2),
                capprops=dict(color=C_SUBTEXT, linewidth=1.2),
                flierprops=dict(
                    marker="o", color=C_SUBTEXT,
                    markersize=3, alpha=0.5,
                    markeredgecolor="none",
                ),
            )
            present_in_box = [s for s in present if s in points_by_strategy]
            for patch, strat in zip(bp["boxes"], present_in_box):
                patch.set_facecolor(STRATEGY_COLOURS.get(strat, "#888888"))
                patch.set_alpha(0.75)
                patch.set_edgecolor(C_GRID)

            ax_right.set_xticks(range(1, len(present_in_box) + 1))
            ax_right.set_xticklabels(present_in_box, rotation=15, ha="right")
            ax_right.set_ylabel("Championship Points", color=C_TEXT)
            ax_right.set_title("Points Distribution", color=C_TEXT,
                                fontsize=FONTSIZE + 1, fontweight="bold")
            _subtitle(ax_right, "Points scored per race entry by strategy")
        else:
            ax_right.text(0.5, 0.5, "Points data\nnot available",
                          ha="center", va="center", transform=ax_right.transAxes,
                          color=C_SUBTEXT, fontsize=FONTSIZE)

        # Legend
        patches = [
            mpatches.Patch(facecolor=STRATEGY_COLOURS.get(s, "#888"), label=s)
            for s in present
        ]
        fig.legend(
            handles=patches, loc="lower center",
            ncol=len(present), framealpha=0.2,
            bbox_to_anchor=(0.5, -0.04),
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Q2 — Which teams gain the most positions?
    # ------------------------------------------------------------------

    def plot_team_position_gain(
        self,
        driver: pd.DataFrame,
        race:   pd.DataFrame,
    ) -> Optional[plt.Figure]:
        """
        Two-panel figure:
          Left  — Average position gain per team (horizontal bar).
          Right — Distribution of position gain per team (dot plot).

        Business insight
        ----------------
        Position gain is a strategy + execution metric.  A team that
        consistently gains 3+ places has either strong pit-lane execution,
        aggressive undercut strategy, or excellent race-pace management
        that allows them to recover from qualifying underperformance.
        Teams with negative average gain are losing places — often a sign
        of tyre degradation issues or poor strategy decisions.
        """
        required = {"team_name", "position_gain"}
        if not required.issubset(driver.columns):
            log.warning("Q2: missing columns %s", required - set(driver.columns))
            return None

        df = driver.dropna(subset=["position_gain", "team_name"]).copy()

        team_stats = (
            df.groupby("team_name")["position_gain"]
            .agg(["mean", "median", "std", "count"])
            .rename(columns={"mean": "avg_gain", "median": "med_gain",
                             "std": "std_gain", "count": "n"})
            .query("n >= 5")
            .sort_values("avg_gain", ascending=True)
        )

        if team_stats.empty:
            log.warning("Q2: no teams with >=5 race entries.")
            return None

        teams  = team_stats.index.tolist()
        colours = [_team_colour(t) for t in teams]

        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=FIGSIZE,
            gridspec_kw={"width_ratios": [1.3, 1]},
        )
        fig.suptitle(
            "Q2 — Which Teams Gain the Most Positions?",
            fontsize=FONTSIZE + 5, fontweight="bold", y=1.01,
        )
        _add_watermark(fig)

        # ── Left: average position gain ───────────────────────────────
        bars = ax_left.barh(
            teams,
            team_stats["avg_gain"].values,
            color=colours,
            edgecolor=C_BG,
            linewidth=0.6,
            height=0.6,
        )

        # Error bars (±1 std dev)
        ax_left.errorbar(
            team_stats["avg_gain"].values,
            range(len(teams)),
            xerr=team_stats["std_gain"].fillna(0).values,
            fmt="none",
            ecolor=C_SUBTEXT,
            elinewidth=1.0,
            capsize=3,
            alpha=0.6,
        )

        # Zero reference line
        ax_left.axvline(0, color=C_ACCENT2, linewidth=1.0, alpha=0.4, linestyle="--")

        # Annotate with mean ± std
        for bar, team in zip(bars, teams):
            w    = team_stats.loc[team, "avg_gain"]
            std  = team_stats.loc[team, "std_gain"]
            n    = int(team_stats.loc[team, "n"])
            xpos = w + 0.1 if w >= 0 else w - 0.1
            ha   = "left" if w >= 0 else "right"
            ax_left.text(
                xpos,
                bar.get_y() + bar.get_height() / 2,
                f"{w:+.2f} ±{std:.1f}  (n={n})",
                va="center", ha=ha,
                fontsize=FONTSIZE - 2, color=C_TEXT,
            )

        ax_left.set_xlabel("Average Position Gain", color=C_TEXT)
        xlim = max(abs(team_stats["avg_gain"].min()),
                   abs(team_stats["avg_gain"].max())) * 1.7
        ax_left.set_xlim(-xlim, xlim)
        ax_left.set_title("Average Position Gain per Team",
                           color=C_TEXT, fontsize=FONTSIZE + 1, fontweight="bold")
        _subtitle(ax_left, "Grid position − Finish position  (positive = gained places)")
        ax_left.grid(axis="y", alpha=0)

        # ── Right: individual race scatter per team ────────────────────
        for i, team in enumerate(teams):
            gains = df.loc[df["team_name"] == team, "position_gain"].values
            jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(gains))
            ax_right.scatter(
                gains,
                np.full(len(gains), i) + jitter,
                color=_team_colour(team),
                alpha=0.55,
                s=18,
                edgecolors="none",
            )
            # Median marker
            ax_right.scatter(
                np.median(gains), i,
                color=C_GOLD,
                s=60,
                zorder=5,
                edgecolors=C_BG,
                linewidths=0.8,
            )

        ax_right.axvline(0, color=C_ACCENT2, linewidth=1.0, alpha=0.4, linestyle="--")
        ax_right.set_yticks(range(len(teams)))
        ax_right.set_yticklabels(teams, fontsize=FONTSIZE - 2)
        ax_right.set_xlabel("Position Gain (individual races)", color=C_TEXT)
        ax_right.set_title("Per-Race Position Gain Distribution",
                            color=C_TEXT, fontsize=FONTSIZE + 1, fontweight="bold")
        _subtitle(ax_right, "Each dot = one race  |  Gold = median")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Q3 — Which circuits favour one-stop vs two-stop strategies?
    # ------------------------------------------------------------------

    def plot_circuit_strategy_heatmap(
        self,
        driver: pd.DataFrame,
        race:   pd.DataFrame,
    ) -> Optional[plt.Figure]:
        """
        Heatmap: circuits (rows) × strategy types (columns).
        Cell value = % of drivers using that strategy at that circuit.

        A secondary grouped bar inset shows the two most common circuits
        side-by-side for quick comparison.

        Business insight
        ----------------
        Low-overtaking circuits (Monaco, Singapore) almost always show
        >80% 1-stop usage because track position is everything — stopping
        twice sacrifices track position with no ability to recover.
        High-speed circuits (Monza, Spa) show more 2-stop usage because
        soft tyres degrade quickly and overtaking is feasible.
        """
        required = {"event_name", "strategy_type"}
        if not required.issubset(driver.columns):
            log.warning("Q3: missing columns %s", required - set(driver.columns))
            return None

        df = driver.dropna(subset=["strategy_type", "event_name"]).copy()

        strategy_order = ["0-stop", "1-stop", "2-stop", "3+-stop"]
        present_strats = [s for s in strategy_order if s in df["strategy_type"].unique()]

        # Circuit × strategy percentage matrix
        pivot = (
            df.groupby(["event_name", "strategy_type"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=present_strats, fill_value=0)
        )
        # Normalise rows to percentage
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

        # Sort circuits by 1-stop usage descending
        sort_col = "1-stop" if "1-stop" in pivot_pct.columns else pivot_pct.columns[0]
        pivot_pct = pivot_pct.sort_values(sort_col, ascending=False)

        circuits = pivot_pct.index.tolist()

        fig, (ax_heat, ax_bar) = plt.subplots(
            1, 2, figsize=(FIGSIZE[0] * 1.1, max(8, len(circuits) * 0.45)),
            gridspec_kw={"width_ratios": [2, 1]},
        )
        fig.suptitle(
            "Q3 — Which Circuits Favour One-Stop vs Two-Stop?",
            fontsize=FONTSIZE + 5, fontweight="bold", y=1.01,
        )
        _add_watermark(fig)

        # ── Heatmap ───────────────────────────────────────────────────
        data_matrix = pivot_pct.values   # shape: (circuits, strategies)

        # Custom colourmap: dark→red
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "f1_heat", [C_PANEL, "#3a0a0a", C_ACCENT], N=256
        )
        im = ax_heat.imshow(
            data_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=100,
            interpolation="nearest",
        )

        ax_heat.set_xticks(range(len(present_strats)))
        ax_heat.set_xticklabels(present_strats, fontsize=FONTSIZE)
        ax_heat.set_yticks(range(len(circuits)))
        ax_heat.set_yticklabels(
            [c.replace(" Grand Prix", "").replace(" GP", "")
             for c in circuits],
            fontsize=FONTSIZE - 2,
        )
        ax_heat.tick_params(axis="x", bottom=True, top=False,
                            labelbottom=True, labeltop=False)

        # Cell annotations
        for i in range(len(circuits)):
            for j in range(len(present_strats)):
                val = data_matrix[i, j]
                text_colour = C_TEXT if val < 55 else C_BG
                ax_heat.text(
                    j, i, f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=FONTSIZE - 2,
                    color=text_colour,
                    fontweight="bold" if val >= 60 else "normal",
                )

        ax_heat.set_title("Strategy Usage per Circuit (%)",
                           color=C_TEXT, fontsize=FONTSIZE + 1, fontweight="bold")
        _subtitle(ax_heat, "Cell = % of drivers using that strategy at that circuit")

        # Colourbar
        cb = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)
        cb.set_label("Usage %", color=C_TEXT, fontsize=FONTSIZE - 1)
        cb.ax.yaxis.set_tick_params(color=C_SUBTEXT)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=C_SUBTEXT)

        # ── Bar chart: 1-stop vs 2-stop by circuit ────────────────────
        cols_to_plot = [s for s in ["1-stop", "2-stop"] if s in pivot_pct.columns]
        y_pos = np.arange(len(circuits))
        bar_h = 0.35

        for k, strat in enumerate(cols_to_plot):
            offset = (k - len(cols_to_plot) / 2 + 0.5) * bar_h
            ax_bar.barh(
                y_pos + offset,
                pivot_pct[strat].values,
                height=bar_h,
                color=STRATEGY_COLOURS.get(strat, "#888"),
                edgecolor=C_BG,
                linewidth=0.5,
                label=strat,
            )

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(
            [c.replace(" Grand Prix", "").replace(" GP", "")
             for c in circuits],
            fontsize=FONTSIZE - 2,
        )
        ax_bar.set_xlabel("Usage (%)", color=C_TEXT)
        ax_bar.set_xlim(0, 105)
        ax_bar.set_title("1-stop vs 2-stop",
                          color=C_TEXT, fontsize=FONTSIZE + 1, fontweight="bold")
        _subtitle(ax_bar, "Side-by-side comparison")
        ax_bar.legend(loc="lower right", fontsize=FONTSIZE - 2)
        ax_bar.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Q4 — Which tyre compounds have the lowest degradation?
    # ------------------------------------------------------------------

    def plot_compound_degradation(
        self,
        driver: pd.DataFrame,
        race:   pd.DataFrame,
    ) -> Optional[plt.Figure]:
        """
        Three-panel figure:
          Left   — Average degradation rate by compound (bar chart).
          Centre — Degradation rate distribution by compound (violin).
          Right  — Average tyre life by compound (horizontal bar).

        Business insight
        ----------------
        A compound with low degradation rate but short average tyre life
        means it is fast but fragile — useful for an undercut but risky on
        a long stint.  A compound with higher degradation but long tyre life
        (HARD) is the natural choice for a one-stop race.  Teams cross-
        reference both metrics when choosing which compound to start on.
        """
        deg_col = "tire_degradation_rate"

        if deg_col not in driver.columns:
            log.warning("Q4: tire_degradation_rate column missing.")
            return None

        df = driver.dropna(subset=[deg_col]).copy()

        # Reconstruct per-compound degradation from compound_usage
        # If the data has per-compound breakdown we use it; otherwise we
        # use the aggregated rate split evenly by strategy group.
        compound_order = ["SOFT", "MEDIUM", "HARD"]

        # Build a mapping: compound_usage → mean degradation rate
        # Group by the first compound used as a proxy for starting compound.
        df["first_compound"] = (
            df["compound_usage"].astype(str).str.split(",").str[0].str.strip().str.upper()
        )
        df_dry = df[df["first_compound"].isin(compound_order)].copy()

        if df_dry.empty:
            log.warning("Q4: no dry-compound entries found in compound_usage.")
            df_dry = df.copy()
            df_dry["first_compound"] = "UNKNOWN"

        comp_stats = (
            df_dry.groupby("first_compound")[deg_col]
            .agg(["mean", "median", "std", "count"])
            .rename(columns={"mean": "avg_deg", "median": "med_deg",
                             "std": "std_deg", "count": "n"})
            .reindex([c for c in compound_order if c in df_dry["first_compound"].unique()])
        )

        avg_life_col = "avg_tyre_life"
        life_stats = (
            df_dry.groupby("first_compound")[avg_life_col].mean()
            if avg_life_col in df_dry.columns
            else pd.Series(dtype=float)
        )

        fig, (ax_bar, ax_violin, ax_life) = plt.subplots(
            1, 3, figsize=FIGSIZE,
        )
        fig.suptitle(
            "Q4 — Which Tyre Compounds Have the Lowest Degradation?",
            fontsize=FONTSIZE + 5, fontweight="bold", y=1.01,
        )
        _add_watermark(fig)

        compounds_present = comp_stats.index.tolist()
        colours = [COMPOUND_COLOURS.get(c, "#888") for c in compounds_present]

        # ── Left: average degradation bar ─────────────────────────────
        bars = ax_bar.bar(
            compounds_present,
            comp_stats["avg_deg"].values,
            color=colours,
            edgecolor=C_BG,
            linewidth=0.8,
            width=0.55,
        )

        ax_bar.errorbar(
            range(len(compounds_present)),
            comp_stats["avg_deg"].values,
            yerr=comp_stats["std_deg"].fillna(0).values,
            fmt="none",
            ecolor=C_SUBTEXT,
            elinewidth=1.2,
            capsize=5,
        )

        for bar, comp in zip(bars, compounds_present):
            h = bar.get_height()
            n = int(comp_stats.loc[comp, "n"])
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                h + comp_stats.loc[comp, "std_deg"] * 0.05 + 0.001,
                f"{h:.4f}\n(n={n})",
                ha="center", va="bottom",
                fontsize=FONTSIZE - 2, color=C_TEXT,
            )

        ax_bar.set_ylabel("Avg Degradation Rate (s/lap)", color=C_TEXT)
        ax_bar.set_title("Mean Degradation Rate",
                          color=C_TEXT, fontsize=FONTSIZE + 1, fontweight="bold")
        _subtitle(ax_bar, "Seconds per lap lost as tyre ages (lower = better)")
        ax_bar.set_ylim(0, comp_stats["avg_deg"].max() * 1.5)

        # Highlight best compound (lowest degradation)
        best_comp = comp_stats["avg_deg"].idxmin()
        best_idx  = compounds_present.index(best_comp)
        bars[best_idx].set_edgecolor(C_GOLD)
        bars[best_idx].set_linewidth(2.5)

        # ── Centre: violin / distribution ─────────────────────────────
        violin_data = [
            df_dry.loc[df_dry["first_compound"] == c, deg_col].dropna().values
            for c in compounds_present
        ]
        violin_data = [v for v in violin_data if len(v) >= 3]

        if violin_data:
            vp = ax_violin.violinplot(
                violin_data,
                positions=range(len(violin_data)),
                showmedians=True,
                showextrema=True,
            )
            for i, (body, comp) in enumerate(zip(vp["bodies"], compounds_present)):
                body.set_facecolor(COMPOUND_COLOURS.get(comp, "#888"))
                body.set_alpha(0.65)
                body.set_edgecolor(C_GRID)
            vp["cmedians"].set_color(C_GOLD)
            vp["cmedians"].set_linewidth(2.0)
            vp["cmins"].set_color(C_SUBTEXT)
            vp["cmaxes"].set_color(C_SUBTEXT)
            vp["cbars"].set_color(C_SUBTEXT)

            ax_violin.set_xticks(range(len(compounds_present)))
            ax_violin.set_xticklabels(compounds_present)
        else:
            ax_violin.text(0.5, 0.5, "Insufficient data\nfor violin plot",
                           ha="center", va="center",
                           transform=ax_violin.transAxes,
                           color=C_SUBTEXT, fontsize=FONTSIZE)

        ax_violin.set_ylabel("Degradation Rate (s/lap)", color=C_TEXT)
        ax_violin.set_title("Degradation Distribution",
                             color=C_TEXT, fontsize=FONTSIZE + 1, fontweight="bold")
        _subtitle(ax_violin, "Full distribution per compound  |  Gold line = median")

        # ── Right: average tyre life ───────────────────────────────────
        if not life_stats.empty:
            life_vals = life_stats.reindex(compounds_present).fillna(0)
            h_bars = ax_life.barh(
                compounds_present,
                life_vals.values,
                color=colours,
                edgecolor=C_BG,
                linewidth=0.8,
                height=0.5,
            )
            for bar, comp in zip(h_bars, compounds_present):
                w = bar.get_width()
                ax_life.text(
                    w + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f} laps",
                    va="center", ha="left",
                    fontsize=FONTSIZE - 1, color=C_TEXT,
                )
            ax_life.set_xlabel("Average Tyre Life (laps)", color=C_TEXT)
            ax_life.set_xlim(0, life_vals.max() * 1.3)
            ax_life.invert_yaxis()
            ax_life.set_title("Average Tyre Life",
                               color=C_TEXT, fontsize=FONTSIZE + 1, fontweight="bold")
            _subtitle(ax_life, "Mean laps completed per tyre set")
            ax_life.grid(axis="y", alpha=0)
        else:
            ax_life.text(0.5, 0.5, "Tyre life data\nnot available",
                         ha="center", va="center",
                         transform=ax_life.transAxes,
                         color=C_SUBTEXT, fontsize=FONTSIZE)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Q5 — Which teams execute the fastest pit stops?
    # ------------------------------------------------------------------

    def plot_team_pit_stop_speed(
        self,
        driver: pd.DataFrame,
        race:   pd.DataFrame,
    ) -> Optional[plt.Figure]:
        """
        Two-panel figure:
          Left  — Average pit stop efficiency ratio per team (lollipop chart).
          Right — Pit stop efficiency over seasons (line chart).

        Business insight
        ----------------
        Pit stop efficiency = driver's avg stop duration / field avg for
        that race.  A value of 0.90 means the crew is 10% faster than
        the average — equivalent to ~2–3 seconds saved, which is often the
        margin between an undercut succeeding or failing.  Red Bull Racing
        has historically set the standard; Ferrari and McLaren are close.
        Tracking efficiency over seasons reveals whether a team is improving
        its crew execution or falling behind.
        """
        eff_col = "pit_stop_efficiency"
        if eff_col not in driver.columns:
            log.warning("Q5: pit_stop_efficiency column missing.")
            return None

        df = driver.dropna(subset=[eff_col, "team_name"]).copy()
        df = df[df[eff_col].between(0.5, 2.0)]   # clip outliers for display

        team_stats = (
            df.groupby("team_name")[eff_col]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "avg_eff", "std": "std_eff", "count": "n"})
            .query("n >= 5")
            .sort_values("avg_eff")
        )

        if team_stats.empty:
            log.warning("Q5: no teams with sufficient pit stop data.")
            return None

        teams   = team_stats.index.tolist()
        colours = [_team_colour(t) for t in teams]

        fig, (ax_lollipop, ax_trend) = plt.subplots(
            1, 2, figsize=FIGSIZE,
            gridspec_kw={"width_ratios": [1.2, 1]},
        )
        fig.suptitle(
            "Q5 — Which Teams Execute the Fastest Pit Stops?",
            fontsize=FONTSIZE + 5, fontweight="bold", y=1.01,
        )
        _add_watermark(fig)

        # ── Left: lollipop chart ──────────────────────────────────────
        y_pos = np.arange(len(teams))

        # Reference line at 1.0 (= field average)
        ax_lollipop.axvline(
            1.0, color=C_ACCENT2, linewidth=1.2, linestyle="--", alpha=0.5,
            label="Field average (1.0)",
        )

        for i, (team, yp, colour) in enumerate(zip(teams, y_pos, colours)):
            avg = team_stats.loc[team, "avg_eff"]
            std = team_stats.loc[team, "std_eff"]

            # Stem
            ax_lollipop.plot(
                [1.0, avg], [yp, yp],
                color=colour, linewidth=1.8, alpha=0.7, zorder=2,
            )
            # Error band
            ax_lollipop.fill_betweenx(
                [yp - 0.18, yp + 0.18],
                max(0.5, avg - std), min(2.0, avg + std),
                color=colour, alpha=0.15, zorder=1,
            )
            # Dot
            ax_lollipop.scatter(
                avg, yp,
                color=colour, s=100, zorder=4,
                edgecolors=C_BG, linewidths=1.0,
            )
            # Annotation
            label_x = avg - 0.005 if avg < 1.0 else avg + 0.005
            ha       = "right"       if avg < 1.0 else "left"
            direction = "▼ faster" if avg < 1.0 else "▲ slower"
            pct_diff  = abs(1 - avg) * 100
            ax_lollipop.text(
                label_x, yp,
                f"{avg:.3f}  {direction} {pct_diff:.1f}%",
                va="center", ha=ha,
                fontsize=FONTSIZE - 2, color=colour,
            )

        ax_lollipop.set_yticks(y_pos)
        ax_lollipop.set_yticklabels(teams, fontsize=FONTSIZE - 1)
        ax_lollipop.set_xlabel("Pit Stop Efficiency Ratio", color=C_TEXT)
        ax_lollipop.set_title("Pit Stop Efficiency vs Field Average",
                               color=C_TEXT, fontsize=FONTSIZE + 1, fontweight="bold")
        _subtitle(ax_lollipop,
                  "Ratio < 1.0 = faster than field  |  Shaded band = ±1 std dev")
        xlim = max(abs(team_stats["avg_eff"].min() - 1),
                   abs(team_stats["avg_eff"].max() - 1)) * 1.6 + 1
        ax_lollipop.set_xlim(2 - xlim, xlim)
        ax_lollipop.legend(fontsize=FONTSIZE - 2, loc="upper right")
        ax_lollipop.grid(axis="y", alpha=0)

        # ── Right: efficiency trend by season ─────────────────────────
        if "season" in driver.columns:
            season_team = (
                df.groupby(["season", "team_name"])[eff_col]
                .mean()
                .reset_index()
            )
            seasons     = sorted(season_team["season"].unique())
            top_teams   = team_stats.head(6).index.tolist()   # top 6 only

            for team in top_teams:
                t_data = season_team[season_team["team_name"] == team].sort_values("season")
                if len(t_data) < 1:
                    continue
                colour = _team_colour(team)
                ax_trend.plot(
                    t_data["season"].values,
                    t_data[eff_col].values,
                    marker="o", markersize=7,
                    color=colour, linewidth=2.0,
                    label=team, zorder=3,
                )
                # Label end of line
                if len(t_data) > 0:
                    last = t_data.iloc[-1]
                    ax_trend.text(
                        last["season"] + 0.05,
                        last[eff_col],
                        team.replace(" F1 Team", "").replace(" Racing", ""),
                        va="center", ha="left",
                        fontsize=FONTSIZE - 3, color=colour,
                    )

            ax_trend.axhline(
                1.0, color=C_ACCENT2, linewidth=1.0,
                linestyle="--", alpha=0.4, label="Field avg",
            )
            ax_trend.set_xticks(seasons)
            ax_trend.set_xticklabels([str(s) for s in seasons])
            ax_trend.set_ylabel("Efficiency Ratio", color=C_TEXT)
            ax_trend.set_title("Pit Stop Efficiency by Season",
                                color=C_TEXT, fontsize=FONTSIZE + 1, fontweight="bold")
            _subtitle(ax_trend, "Top 6 teams by mean efficiency  |  Lower = faster")
            ax_trend.legend(fontsize=FONTSIZE - 3, loc="upper right")
        else:
            ax_trend.text(0.5, 0.5, "Season data\nnot available",
                          ha="center", va="center",
                          transform=ax_trend.transAxes,
                          color=C_SUBTEXT, fontsize=FONTSIZE)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Summary dashboard — all five analyses on one canvas
    # ------------------------------------------------------------------

    def _build_dashboard(
        self,
        driver:    pd.DataFrame,
        race:      pd.DataFrame,
        out_path:  Path,
    ) -> None:
        """
        Compose a 2×3 grid dashboard embedding the key panel from each of
        the five analyses, plus a title / insight card in the 6th cell.
        """
        fig = plt.figure(figsize=(22, 14))
        fig.patch.set_facecolor(C_BG)

        gs = fig.add_gridspec(
            2, 3,
            hspace=0.48,
            wspace=0.38,
            left=0.07, right=0.97,
            top=0.90, bottom=0.06,
        )

        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
        ]

        fig.suptitle(
            "F1 Race Strategy Analytics — Summary Dashboard",
            fontsize=FONTSIZE + 8, fontweight="bold",
            color=C_TEXT, y=0.97,
        )
        _add_watermark(fig)

        # ── Panel 1: Strategy win rate (bar) ──────────────────────────
        self._dash_strategy_win_rate(driver, axes[0])

        # ── Panel 2: Team position gain (bar) ─────────────────────────
        self._dash_team_position_gain(driver, axes[1])

        # ── Panel 3: Circuit strategy (top 10 circuits heatmap) ───────
        self._dash_circuit_heatmap(driver, axes[2])

        # ── Panel 4: Compound degradation ─────────────────────────────
        self._dash_compound_degradation(driver, axes[3])

        # ── Panel 5: Team pit stop efficiency ─────────────────────────
        self._dash_pit_efficiency(driver, axes[4])

        # ── Panel 6: Key insights text card ───────────────────────────
        self._dash_insight_card(driver, race, axes[5])

        _save(fig, out_path, "summary_dashboard")

    # Dashboard sub-panels (condensed versions of full analyses)

    def _dash_strategy_win_rate(self, driver: pd.DataFrame, ax: plt.Axes) -> None:
        if "strategy_type" not in driver.columns or "finish_position" not in driver.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color=C_SUBTEXT)
            return

        df = driver.copy()
        df["is_win"] = df["finish_position"] == 1
        order   = ["0-stop", "1-stop", "2-stop", "3+-stop"]
        present = [s for s in order if s in df["strategy_type"].unique()]

        win_rate = (
            df.groupby("strategy_type")["is_win"]
            .mean() * 100
        ).reindex(present)

        colours = [STRATEGY_COLOURS.get(s, "#888") for s in present]
        bars = ax.barh(present, win_rate.values, color=colours,
                       edgecolor=C_BG, height=0.55)
        ax.axvline(0, color=C_GRID, linewidth=0.5)

        for bar, strat in zip(bars, present):
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%", va="center", ha="left",
                    fontsize=FONTSIZE - 2, color=C_TEXT)

        ax.invert_yaxis()
        ax.set_xlabel("Win Rate (%)", fontsize=FONTSIZE - 1)
        ax.set_title("Q1 · Strategy Win Rate",
                     color=C_ACCENT, fontsize=FONTSIZE, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    def _dash_team_position_gain(self, driver: pd.DataFrame, ax: plt.Axes) -> None:
        if "position_gain" not in driver.columns or "team_name" not in driver.columns:
            return

        stats = (
            driver.dropna(subset=["position_gain", "team_name"])
            .groupby("team_name")["position_gain"]
            .mean()
            .sort_values()
        )
        colours = [_team_colour(t) for t in stats.index]
        bars = ax.barh(stats.index, stats.values,
                       color=colours, edgecolor=C_BG, height=0.6)
        ax.axvline(0, color=C_ACCENT2, linewidth=0.8, linestyle="--", alpha=0.4)

        for bar in bars:
            w  = bar.get_width()
            xp = w + 0.05 if w >= 0 else w - 0.05
            ha = "left"    if w >= 0 else "right"
            ax.text(xp, bar.get_y() + bar.get_height() / 2,
                    f"{w:+.1f}", va="center", ha=ha,
                    fontsize=FONTSIZE - 3, color=C_TEXT)

        ax.set_xlabel("Avg Position Gain", fontsize=FONTSIZE - 1)
        ax.set_title("Q2 · Team Position Gain",
                     color=C_ACCENT, fontsize=FONTSIZE, fontweight="bold")
        ax.tick_params(axis="y", labelsize=FONTSIZE - 3)

    def _dash_circuit_heatmap(self, driver: pd.DataFrame, ax: plt.Axes) -> None:
        if "event_name" not in driver.columns or "strategy_type" not in driver.columns:
            return

        order   = ["1-stop", "2-stop"]
        present = [s for s in order if s in driver["strategy_type"].unique()]
        if not present:
            return

        pivot = (
            driver.dropna(subset=["strategy_type", "event_name"])
            .groupby(["event_name", "strategy_type"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=present, fill_value=0)
        )
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        sort_col  = present[0]
        pivot_pct = pivot_pct.sort_values(sort_col, ascending=False).head(12)

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "f1_heat2", [C_PANEL, "#3a0a0a", C_ACCENT], N=256
        )
        ax.imshow(pivot_pct.values, aspect="auto",
                  cmap=cmap, vmin=0, vmax=100, interpolation="nearest")

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, fontsize=FONTSIZE - 1)
        ax.set_yticks(range(len(pivot_pct)))
        ax.set_yticklabels(
            [c.replace(" Grand Prix", "").replace(" GP", "")
             for c in pivot_pct.index],
            fontsize=FONTSIZE - 3,
        )

        for i in range(len(pivot_pct)):
            for j in range(len(present)):
                val = pivot_pct.values[i, j]
                tc  = C_TEXT if val < 55 else C_BG
                ax.text(j, i, f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=FONTSIZE - 3, color=tc)

        ax.set_title("Q3 · Circuit Strategy Heatmap",
                     color=C_ACCENT, fontsize=FONTSIZE, fontweight="bold")

    def _dash_compound_degradation(self, driver: pd.DataFrame, ax: plt.Axes) -> None:
        deg_col = "tire_degradation_rate"
        if deg_col not in driver.columns:
            return

        df = driver.dropna(subset=[deg_col]).copy()
        df["first_compound"] = (
            df["compound_usage"].astype(str).str.split(",").str[0]
            .str.strip().str.upper()
        )
        order   = ["SOFT", "MEDIUM", "HARD"]
        present = [c for c in order if c in df["first_compound"].unique()]
        if not present:
            return

        means   = df.groupby("first_compound")[deg_col].mean().reindex(present)
        colours = [COMPOUND_COLOURS.get(c, "#888") for c in present]

        bars = ax.bar(present, means.values, color=colours,
                      edgecolor=C_BG, linewidth=0.8, width=0.5)
        for bar, comp in zip(bars, present):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                    f"{h:.4f}", ha="center", va="bottom",
                    fontsize=FONTSIZE - 2, color=C_TEXT)

        ax.set_ylabel("Deg Rate (s/lap)", fontsize=FONTSIZE - 1)
        ax.set_title("Q4 · Compound Degradation",
                     color=C_ACCENT, fontsize=FONTSIZE, fontweight="bold")

    def _dash_pit_efficiency(self, driver: pd.DataFrame, ax: plt.Axes) -> None:
        eff_col = "pit_stop_efficiency"
        if eff_col not in driver.columns:
            return

        df    = driver.dropna(subset=[eff_col, "team_name"]).copy()
        df    = df[df[eff_col].between(0.5, 2.0)]
        stats = (
            df.groupby("team_name")[eff_col]
            .mean()
            .sort_values()
        )
        colours = [_team_colour(t) for t in stats.index]

        y_pos = np.arange(len(stats))
        ax.barh(y_pos, stats.values, color=colours, edgecolor=C_BG, height=0.6)
        ax.axvline(1.0, color=C_ACCENT2, linewidth=0.8,
                   linestyle="--", alpha=0.5)

        for i, (team, val) in enumerate(stats.items()):
            xp = val + 0.005 if val >= 1 else val - 0.005
            ha = "left"      if val >= 1 else "right"
            ax.text(xp, i, f"{val:.3f}",
                    va="center", ha=ha,
                    fontsize=FONTSIZE - 3, color=C_TEXT)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(stats.index, fontsize=FONTSIZE - 3)
        ax.set_xlabel("Efficiency Ratio", fontsize=FONTSIZE - 1)
        ax.set_title("Q5 · Team Pit Stop Efficiency",
                     color=C_ACCENT, fontsize=FONTSIZE, fontweight="bold")

    def _dash_insight_card(
        self,
        driver: pd.DataFrame,
        race:   pd.DataFrame,
        ax:     plt.Axes,
    ) -> None:
        """Generate a text-based insight card summarising key findings."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        insights: list[str] = ["Key Strategic Insights", ""]

        # Q1 insight
        if "strategy_type" in driver.columns and "finish_position" in driver.columns:
            df = driver.copy()
            df["is_win"] = df["finish_position"] == 1
            win_rate = (
                df.groupby("strategy_type")["is_win"].mean() * 100
            ).sort_values(ascending=False)
            if not win_rate.empty:
                best = win_rate.index[0]
                rate = win_rate.iloc[0]
                insights.append(f"🏆  {best} strategy wins {rate:.1f}% of entries")

        # Q2 insight
        if "position_gain" in driver.columns and "team_name" in driver.columns:
            best_team = (
                driver.dropna(subset=["position_gain", "team_name"])
                .groupby("team_name")["position_gain"]
                .mean()
                .idxmax()
            )
            best_gain = (
                driver.groupby("team_name")["position_gain"]
                .mean()[best_team]
            )
            insights.append(f"📈  {best_team} gains most places (+{best_gain:.1f} avg)")


        # Q3 insight
        if "event_name" in driver.columns and "strategy_type" in driver.columns:
            one_stop_circuits = (
                driver[driver["strategy_type"] == "1-stop"]
                .groupby("event_name").size()
                .sort_values(ascending=False)
            )
            if not one_stop_circuits.empty:
                top_circuit = one_stop_circuits.index[0].replace(" Grand Prix", "")
                insights.append(f"🗺️   {top_circuit} most favours 1-stop strategy")

        # Q4 insight
        if "tire_degradation_rate" in driver.columns and "compound_usage" in driver.columns:
            df4 = driver.dropna(subset=["tire_degradation_rate"]).copy()
            df4["first_compound"] = (
                df4["compound_usage"].astype(str).str.split(",")
                .str[0].str.strip().str.upper()
            )
            comp_deg = (
                df4[df4["first_compound"].isin(["SOFT", "MEDIUM", "HARD"])]
                .groupby("first_compound")["tire_degradation_rate"]
                .mean()
            )
            if not comp_deg.empty:
                lowest = comp_deg.idxmin()
                val    = comp_deg.min()
                insights.append(f"🔵  {lowest} has lowest degradation ({val:.4f} s/lap)")

        # Q5 insight
        if "pit_stop_efficiency" in driver.columns and "team_name" in driver.columns:
            eff = (
                driver.dropna(subset=["pit_stop_efficiency", "team_name"])
                .groupby("team_name")["pit_stop_efficiency"]
                .mean()
                .sort_values()
            )
            if not eff.empty:
                fastest_team = eff.index[0]
                eff_val      = eff.iloc[0]
                pct_faster   = (1 - eff_val) * 100
                insights.append(
                    f"⚡  {fastest_team} fastest crew "
                    f"({pct_faster:.1f}% below field avg)"
                )

        # Wet race stat
        if "wet_race" in driver.columns:
            wet_pct = driver["wet_race"].mean() * 100
            insights.append(f"🌧️   {wet_pct:.1f}% of race entries were wet")

        # Safety car stat
        if "safety_car_deployed" in race.columns:
            sc_pct = race["safety_car_deployed"].mean() * 100
            insights.append(f"🚗  Safety car deployed in {sc_pct:.1f}% of races")

        insights += ["", "Data: FastF1 API  |  Seasons 2023–2024"]

        y_start = 0.94
        line_h  = 0.085

        for i, line in enumerate(insights):
            is_title = i == 0
            colour   = C_ACCENT if is_title else C_TEXT
            size     = FONTSIZE + 1 if is_title else FONTSIZE - 1
            weight   = "bold" if is_title else "normal"

            ax.text(
                0.05, y_start - i * line_h,
                line,
                transform=ax.transAxes,
                fontsize=size,
                color=colour,
                fontweight=weight,
                va="top",
            )

        # Border around the card
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(C_ACCENT)
            spine.set_linewidth(1.5)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate F1 strategy analysis figures."
    )
    parser.add_argument(
        "--features-dir",
        type    = Path,
        default = FEATURES_DIR,
        help    = "Directory containing feature Parquet files",
    )
    parser.add_argument(
        "--figures-dir",
        type    = Path,
        default = FIGURES_DIR,
        help    = "Output directory for PNG figures",
    )
    args = parser.parse_args()

    analyser = F1StrategyAnalyser(
        features_dir = args.features_dir,
        figures_dir  = args.figures_dir,
    )
    saved = analyser.run()

    print(f"\n  {len(saved)} figures saved to: {args.figures_dir}\n")
    for label, path in saved.items():
        print(f"    {label:<35} → {path.name}")
    print()