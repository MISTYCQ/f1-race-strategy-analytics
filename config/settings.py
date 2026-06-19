"""
config/settings.py
==================
Centralised configuration for the F1 Strategy Dashboard project.

All constants, paths, and toggleable options live here so that
every other module imports from a single source of truth.
No magic strings scattered across the codebase.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
# Resolve once so every path is absolute regardless of where scripts are run.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR          = PROJECT_ROOT / "data"
RAW_DIR           = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed"
EXPORTS_DIR       = DATA_DIR / "exports"

RAW_RACES_DIR     = RAW_DIR / "races"
RAW_LAPS_DIR      = RAW_DIR / "laps"
RAW_PITS_DIR      = RAW_DIR / "pit_stops"
RAW_WEATHER_DIR   = RAW_DIR / "weather"
RAW_TELEMETRY_DIR = RAW_DIR / "telemetry"

LOGS_DIR          = PROJECT_ROOT / "logs"
REPORTS_DIR       = PROJECT_ROOT / "reports"
FIGURES_DIR       = REPORTS_DIR / "figures"

# FastF1 uses a local disk cache to avoid re-downloading session data.
# Keeping it inside the project makes it portable.
FASTF1_CACHE_DIR  = PROJECT_ROOT / ".fastf1_cache"

# ── Seasons & events in scope ─────────────────────────────────────────────────
# 2023–2024 covers the modern ground-effect regulation era.
SEASONS: list[int] = [2023, 2024]

# Sessions to download. "R" = Race only (no qualifying, practice).
SESSION_TYPE: str = "R"

# ── Tire compound mapping ─────────────────────────────────────────────────────
# FastF1 returns the colour-coded "Compound" column:
# SOFT / MEDIUM / HARD / INTERMEDIATE / WET
# We use this as it is circuit-agnostic (C1–C5 varies per event).
VALID_DRY_COMPOUNDS: list[str] = ["SOFT", "MEDIUM", "HARD"]
VALID_ALL_COMPOUNDS: list[str] = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

# ── Filtering assumptions ─────────────────────────────────────────────────────
# Dry-only flag: set True to exclude races with any recorded rainfall.
# Reduces confounding from weather on strategy choices.
# False = collect everything, apply filter in the cleaning phase.
FILTER_DRY_ONLY: bool = False

# Minimum laps to count a race (safety car / red flag shortened events).
MIN_RACE_LAPS: int = 20

# ── Parquet settings ──────────────────────────────────────────────────────────
# "snappy" balances compression speed vs size — good default for analytics.
PARQUET_COMPRESSION: str = "snappy"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# ── Retry policy (for network-bound FastF1 calls) ─────────────────────────────
MAX_RETRIES: int    = 3
RETRY_WAIT_MIN: int = 2    # seconds
RETRY_WAIT_MAX: int = 10   # seconds