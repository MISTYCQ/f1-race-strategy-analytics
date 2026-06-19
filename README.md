# 🏎️ F1 Race Strategy Analytics

End-to-end Formula 1 analytics platform built with FastF1, Python, Pandas, SQL, Tableau, and Power BI.

Analyzed 46 Formula 1 Grands Prix across the 2023–2024 seasons to uncover insights on race strategies, pit stop performance, and tire degradation.

## Dataset Summary

| Metric | Value |
|----------|----------|
| Seasons Analyzed | 2023–2024 |
| Grand Prix Weekends | 46 |
| Driver-Race Records | 918 |
| Laps Processed | 50,000+ |
| Teams | 10 |
| Circuits | 20+ |

## Overview

Analyzed 46 Formula 1 Grands Prix across the 2023–2024 seasons using FastF1 and Python to study race strategy patterns, pit stop efficiency, and tire degradation.

Built an end-to-end analytics pipeline covering:

- Data Collection
- Data Cleaning
- Feature Engineering
- Exploratory Analysis
- Interactive Dashboarding

## Tech Stack

- Python
- Pandas
- FastF1
- NumPy
- PyArrow
- Matplotlib
- SQL
- Tableau
- Power BI
- Git & GitHub

## Dataset Summary

| Metric | Value |
|----------|----------|
| Seasons Analyzed | 2023–2024 |
| Grand Prix Weekends | 46 |
| Driver-Race Records | 918 |
| Laps Processed | 50,000+ |
| Teams | 10 |
| Circuits | 20+ |

## Dashboard Preview

![Dashboard](assets/images/dashboard.png)

## Key Insights

- Two-stop strategies were the most commonly used race strategy.
- Strategy preference varies significantly by circuit.
- Hard compounds demonstrated the longest average tire life.
- Mercedes and Red Bull showed among the fastest pit stop performances.
- Tire degradation patterns differ substantially across compounds and tracks.

## Key Visualizations

### Circuit Strategy Heatmap

<img src="assets/images/q3_circuit_strategy_heatmap.png" width="900">

### Tire Degradation Analysis

<img src="assets/images/q4_compound_degradation.png" width="900">

### Team Pit Stop Efficiency

<img src="assets/images/q5_team_pit_stop_speed.png" width="900">

## Data Pipeline

FastF1 API
    ↓
Data Collection
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Exploratory Analysis
    ↓
Visualization & Dashboarding

## How To Run

```bash
git clone https://github.com/MISTYCQ/f1-race-strategy-analytics.git

cd f1-race-strategy-analytics

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python run_collection.py
python run_cleaning.py
python run_features.py
python run_analysis.py
