import numpy as np
import pandas as pd

from solar_yield.features import (
    add_fraction_features,
    apply_physical_overrides,
    calculate_interior_weights,
    prepare_model_matrix,
)
from solar_yield.quality import validate_hourly_forecast


def test_add_fraction_features_normalizes_weather_inputs():
    frame = pd.DataFrame(
        {
            "sunshine_duration": [1800],
            "cloud_low": [25],
            "cloud_mid": [50],
            "cloud_high": [75],
            "relative_humidity": [80],
        }
    )

    result = add_fraction_features(frame)

    assert result.loc[0, "sunshine_fraction"] == 0.5
    assert result.loc[0, "cloud_low_fraction"] == 0.25
    assert result.loc[0, "cloud_mid_fraction"] == 0.5
    assert result.loc[0, "cloud_high_fraction"] == 0.75
    assert result.loc[0, "rh_fraction"] == 0.8


def test_prepare_model_matrix_preserves_feature_order_and_fills_gaps():
    frame = pd.DataFrame(
        {
            "cloud_low_fraction": [0.2, np.nan],
            "cloud_mid_fraction": [0.1, 0.2],
            "cloud_high_fraction": [0.4, 0.5],
            "water_vapour": [10.0, 11.0],
            "sunshine_fraction": [0.0, 0.8],
            "temperature": [20.0, 21.0],
            "rh_fraction": [0.5, 0.6],
            "surface_pressure": [1010.0, 1011.0],
        }
    )

    matrix = prepare_model_matrix(frame)

    assert list(matrix.columns)[0] == "cloud_low_fraction"
    assert not matrix.isna().any().any()
    assert matrix.loc[1, "cloud_low_fraction"] == 0.2


def test_apply_physical_overrides_clips_and_handles_night():
    frame = pd.DataFrame(
        {
            "sunshine_fraction": [0.0, 1.0, 1.0],
            "cloud_low_fraction": [0.2, 1.0, 0.1],
        }
    )
    predictions = np.array([[0.8, 0.4], [0.7, 1.2], [1.5, -0.1]])

    result = apply_physical_overrides(frame, predictions)

    assert result.loc[0, "final_pred_direct"] == 0.0
    assert result.loc[0, "final_pred_diffuse"] == 0.0
    assert result.loc[1, "final_pred_direct"] == 0.0
    assert result.loc[1, "final_pred_diffuse"] == 1.0
    assert result.loc[2, "final_pred_direct"] == 1.0
    assert result.loc[2, "final_pred_diffuse"] == 0.0


def test_calculate_interior_weights_prioritizes_non_boundary_targets():
    targets = pd.DataFrame(
        {
            "direct_clear_sky_factor": [0.0, 0.5, 1.0],
            "diffuse_clear_sky_factor": [0.2, 0.6, 0.8],
        }
    )

    weights = calculate_interior_weights(targets)

    assert weights.tolist() == [1.0, 5.0, 1.0]


def test_validate_hourly_forecast_accepts_continuous_7_day_batch():
    timestamps = pd.date_range("2026-01-01", periods=168, freq="h")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cloud_low": 0,
            "cloud_mid": 0,
            "cloud_high": 0,
            "water_vapour": 10,
            "sunshine_duration": 0,
            "temperature": 20,
            "relative_humidity": 50,
            "surface_pressure": 1010,
        }
    )

    validate_hourly_forecast(frame)
