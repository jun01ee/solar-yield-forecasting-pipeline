"""Feature engineering utilities shared by training and inference."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "cloud_low_fraction",
    "cloud_mid_fraction",
    "cloud_high_fraction",
    "water_vapour",
    "sunshine_fraction",
    "temperature",
    "rh_fraction",
    "surface_pressure",
]

TARGET_COLUMNS = ["direct_clear_sky_factor", "diffuse_clear_sky_factor"]


def add_fraction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized cloud, humidity, and sunshine features."""

    result = frame.copy()
    result["sunshine_fraction"] = result["sunshine_duration"] / 3600.0
    result["cloud_low_fraction"] = result["cloud_low"] / 100.0
    result["cloud_mid_fraction"] = result["cloud_mid"] / 100.0
    result["cloud_high_fraction"] = result["cloud_high"] / 100.0
    result["rh_fraction"] = result["relative_humidity"] / 100.0
    return result


def prepare_model_matrix(
    frame: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Return model features in the trained order with conservative missing-value handling."""

    features = frame.loc[:, list(feature_columns)].copy()
    return features.ffill().bfill()


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
