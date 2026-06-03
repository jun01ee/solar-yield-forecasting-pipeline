"""Feature engineering utilities shared by training and inference."""

from __future__ import annotations

from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = [
    "cloud_low_fraction",
    "cloud_mid_fraction",
    "cloud_high_fraction",
    "water_vapour",
    "sunshine_fraction",
    "temperature",
    "rh_fraction",
    "surface_pressure",
]

TIME_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

DEFAULT_LAG_HOURS = (1, 3, 6, 24)
DEFAULT_ROLLING_WINDOWS = (3, 6, 24)
DELTA_SOURCE_COLUMNS = [
    "cloud_low_fraction",
    "cloud_mid_fraction",
    "cloud_high_fraction",
    "sunshine_fraction",
    "rh_fraction",
    "surface_pressure",
]
DEFAULT_DELTA_LAGS = (1,)


def _lag_feature_name(column: str, lag_hour: int) -> str:
    return f"{column}_lag_{lag_hour}h"


def _rolling_mean_feature_name(column: str, window_hour: int) -> str:
    return f"{column}_rolling_mean_{window_hour}h"


def _delta_feature_name(column: str, lag_hour: int) -> str:
    return f"{column}_delta_{lag_hour}h"


def _build_dynamic_feature_columns(
    lag_hours: Iterable[int] = DEFAULT_LAG_HOURS,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    delta_lags: Iterable[int] = DEFAULT_DELTA_LAGS,
) -> list[str]:
    lag_columns = [
        _lag_feature_name(column, lag_hour)
        for lag_hour in lag_hours
        for column in BASE_FEATURE_COLUMNS
    ]
    rolling_columns = [
        _rolling_mean_feature_name(column, window_hour)
        for window_hour in rolling_windows
        for column in BASE_FEATURE_COLUMNS
    ]
    delta_columns = [
        _delta_feature_name(column, lag_hour)
        for lag_hour in delta_lags
        for column in DELTA_SOURCE_COLUMNS
    ]
    return lag_columns + rolling_columns + delta_columns


DYNAMIC_FEATURE_COLUMNS = _build_dynamic_feature_columns()
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + TIME_FEATURE_COLUMNS + DYNAMIC_FEATURE_COLUMNS
TARGET_COLUMNS = ["direct_clear_sky_factor", "diffuse_clear_sky_factor"]
RAW_FRACTION_INPUT_COLUMNS = [
    "sunshine_duration",
    "cloud_low",
    "cloud_mid",
    "cloud_high",
    "relative_humidity",
]


def add_fraction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized cloud, humidity, and sunshine features."""

    result = frame.copy()
    result["sunshine_fraction"] = result["sunshine_duration"] / 3600.0
    result["cloud_low_fraction"] = result["cloud_low"] / 100.0
    result["cloud_mid_fraction"] = result["cloud_mid"] / 100.0
    result["cloud_high_fraction"] = result["cloud_high"] / 100.0
    result["rh_fraction"] = result["relative_humidity"] / 100.0
    return result


def _ensure_fraction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame containing the normalized weather feature columns."""

    if all(column in frame.columns for column in BASE_FEATURE_COLUMNS):
        return frame.copy()

    missing_raw = [column for column in RAW_FRACTION_INPUT_COLUMNS if column not in frame.columns]
    if missing_raw:
        missing_features = [
            column for column in BASE_FEATURE_COLUMNS if column not in frame.columns
        ]
        raise ValueError(
            "frame must contain normalized feature columns or raw weather columns; "
            f"missing normalized={missing_features}, missing raw={missing_raw}"
        )
    return add_fraction_features(frame)


def add_time_features(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Add cyclic hour-of-day and day-of-year features."""

    if timestamp_column not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_column}")

    result = frame.copy()
    timestamps = pd.to_datetime(result[timestamp_column])
    hour = timestamps.dt.hour + (timestamps.dt.minute / 60.0)
    day_of_year = timestamps.dt.dayofyear

    result["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    result["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    result["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    return result


def add_weather_dynamics(
    frame: pd.DataFrame,
    lag_hours: Iterable[int] = DEFAULT_LAG_HOURS,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    timestamp_column: str = "timestamp",
    fill_early_rows: bool = True,
) -> pd.DataFrame:
    """Add past-only lag, rolling mean, and delta weather features."""

    if timestamp_column not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_column}")

    missing_features = [column for column in BASE_FEATURE_COLUMNS if column not in frame.columns]
    if missing_features:
        raise ValueError(f"missing required weather feature columns: {missing_features}")

    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column])
    result = result.sort_values(timestamp_column).reset_index(drop=True)

    for lag_hour in lag_hours:
        for column in BASE_FEATURE_COLUMNS:
            lagged = result[column].shift(lag_hour)
            if fill_early_rows:
                lagged = lagged.fillna(result[column])
            result[_lag_feature_name(column, lag_hour)] = lagged

    for window_hour in rolling_windows:
        min_periods = 1 if fill_early_rows else window_hour
        for column in BASE_FEATURE_COLUMNS:
            result[_rolling_mean_feature_name(column, window_hour)] = (
                result[column].rolling(window=window_hour, min_periods=min_periods).mean()
            )

    for lag_hour in DEFAULT_DELTA_LAGS:
        for column in DELTA_SOURCE_COLUMNS:
            lagged = result[column].shift(lag_hour)
            delta = result[column] - lagged
            if fill_early_rows:
                delta = delta.fillna(0.0)
            result[_delta_feature_name(column, lag_hour)] = delta

    return result


def add_model_features(
    frame: pd.DataFrame,
    mode: Literal["training", "inference"] = "inference",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Add the full model feature set for training or inference."""

    if mode not in {"training", "inference"}:
        raise ValueError("mode must be either 'training' or 'inference'")

    result = _ensure_fraction_features(frame)
    result = add_time_features(result, timestamp_column=timestamp_column)
    result = add_weather_dynamics(
        result,
        timestamp_column=timestamp_column,
        fill_early_rows=(mode == "inference"),
    )

    if mode == "training":
        result = result.dropna(subset=DYNAMIC_FEATURE_COLUMNS).reset_index(drop=True)
    return result


def prepare_model_matrix(
    frame: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return model features in the trained order with conservative missing-value handling."""

    selected_columns = BASE_FEATURE_COLUMNS if feature_columns is None else list(feature_columns)
    features = frame.loc[:, selected_columns].copy()
    return features.ffill().fillna(0.0)


def apply_physical_overrides(frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Clip model predictions and enforce simple physical boundary rules."""

    if predictions.shape[1] != 2:
        raise ValueError("predictions must contain direct and diffuse columns")

    result = frame.copy()
    result["pred_direct_raw"] = predictions[:, 0]
    result["pred_diffuse_raw"] = predictions[:, 1]

    result["final_pred_direct"] = np.where(
        result["sunshine_fraction"] == 0.0,
        0.0,
        np.where(
            result["cloud_low_fraction"] == 1.0,
            0.0,
            np.clip(result["pred_direct_raw"], 0.0, 1.0),
        ),
    )
    result["final_pred_diffuse"] = np.where(
        result["sunshine_fraction"] == 0.0,
        0.0,
        np.clip(result["pred_diffuse_raw"], 0.0, 1.0),
    )
    return result


def calculate_interior_weights(targets: pd.DataFrame, edge_tolerance: float = 0.01) -> np.ndarray:
    """Weight non-boundary samples higher so the model learns dynamic daylight behavior."""

    direct = targets.iloc[:, 0]
    diffuse = targets.iloc[:, 1]
    is_edge_direct = (direct <= edge_tolerance) | (direct >= 1.0 - edge_tolerance)
    is_edge_diffuse = (diffuse <= edge_tolerance) | (diffuse >= 1.0 - edge_tolerance)
    return np.where(~is_edge_direct & ~is_edge_diffuse, 5.0, 1.0)
