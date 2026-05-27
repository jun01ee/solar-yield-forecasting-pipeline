"""Data quality checks for operational forecast batches."""

from __future__ import annotations

import pandas as pd


def validate_hourly_forecast(frame: pd.DataFrame, expected_rows: int = 168) -> None:
    """Validate row count, timestamp uniqueness, hourly cadence, and null safety."""

    if len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} forecast rows, got {len(frame)}")
    if frame["timestamp"].isna().any():
        raise ValueError("timestamp contains null values")
    if frame["timestamp"].duplicated().any():
        raise ValueError("timestamp contains duplicates")

    timestamps = pd.to_datetime(frame["timestamp"]).sort_values()
    deltas = timestamps.diff().dropna()
    if not (deltas == pd.Timedelta(hours=1)).all():
        raise ValueError("forecast timestamps must be continuous hourly values")

    required_features = [
        "cloud_low",
        "cloud_mid",
        "cloud_high",
        "water_vapour",
        "sunshine_duration",
        "temperature",
        "relative_humidity",
        "surface_pressure",
    ]
    missing = [column for column in required_features if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
