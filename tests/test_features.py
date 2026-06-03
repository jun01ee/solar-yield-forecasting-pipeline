import numpy as np
import pandas as pd

from solar_yield.features import (
    BASE_FEATURE_COLUMNS,
    DYNAMIC_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    TIME_FEATURE_COLUMNS,
    add_fraction_features,
    add_model_features,
    add_time_features,
    add_weather_dynamics,
    apply_physical_overrides,
    calculate_interior_weights,
    prepare_model_matrix,
)
from solar_yield.quality import validate_hourly_forecast


def _raw_weather_frame(periods=30):
    values = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=periods, freq="h"),
            "cloud_low": values % 100,
            "cloud_mid": (values + 10.0) % 100,
            "cloud_high": (values + 20.0) % 100,
            "water_vapour": 10.0 + values * 0.1,
            "sunshine_duration": 3600.0,
            "temperature": 20.0 + values * 0.05,
            "relative_humidity": 50.0 + values % 40,
            "surface_pressure": 1010.0 + values * 0.1,
        }
    )


def _normalized_weather_frame(cloud_low_values, timestamps=None):
    values = np.asarray(cloud_low_values, dtype=float)
    if timestamps is None:
        timestamps = pd.date_range("2026-01-01", periods=len(values), freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "cloud_low_fraction": values,
            "cloud_mid_fraction": values,
            "cloud_high_fraction": values,
            "water_vapour": 10.0 + values,
            "sunshine_fraction": values,
            "temperature": 20.0 + values,
            "rh_fraction": values,
            "surface_pressure": 1010.0 + values,
        }
    )


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


def test_add_time_features_creates_bounded_cyclic_columns():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 00:00:00", "2026-01-01 06:00:00"]
            )
        }
    )

    result = add_time_features(frame)

    assert np.isclose(result.loc[0, "hour_sin"], 0.0)
    assert np.isclose(result.loc[0, "hour_cos"], 1.0)
    assert np.isclose(result.loc[1, "hour_sin"], 1.0)
    assert np.isclose(result.loc[1, "hour_cos"], 0.0)
    for column in TIME_FEATURE_COLUMNS:
        assert result[column].between(-1.0, 1.0).all()


def test_add_weather_dynamics_sorts_before_lagging_without_future_leakage():
    timestamps = pd.to_datetime(
        ["2026-01-01 02:00:00", "2026-01-01 00:00:00", "2026-01-01 01:00:00"]
    )
    frame = _normalized_weather_frame([0.3, 0.1, 0.2], timestamps=timestamps)

    result = add_weather_dynamics(
        frame,
        lag_hours=(1,),
        rolling_windows=(3,),
        fill_early_rows=False,
    )

    assert result["timestamp"].tolist() == sorted(timestamps.tolist())
    assert np.isnan(result.loc[0, "cloud_low_fraction_lag_1h"])
    assert result.loc[1, "cloud_low_fraction_lag_1h"] == 0.1
    assert result.loc[2, "cloud_low_fraction_lag_1h"] == 0.2
    assert np.isnan(result.loc[0, "cloud_low_fraction_delta_1h"])
    assert np.isclose(result.loc[1, "cloud_low_fraction_delta_1h"], 0.1)
    assert np.isclose(result.loc[2, "cloud_low_fraction_delta_1h"], 0.1)


def test_add_weather_dynamics_rolling_mean_uses_current_and_past_rows_only():
    frame = _normalized_weather_frame([0.1, 0.2, 0.9])

    result = add_weather_dynamics(
        frame,
        lag_hours=(),
        rolling_windows=(2,),
        fill_early_rows=True,
    )

    assert result.loc[0, "cloud_low_fraction_rolling_mean_2h"] == 0.1
    assert np.isclose(result.loc[1, "cloud_low_fraction_rolling_mean_2h"], 0.15)
    assert np.isclose(result.loc[2, "cloud_low_fraction_rolling_mean_2h"], 0.55)


def test_add_model_features_inference_keeps_rows_and_fills_early_history():
    frame = _raw_weather_frame(periods=168)

    result = add_model_features(frame, mode="inference")
    matrix = prepare_model_matrix(result, FEATURE_COLUMNS)

    assert len(result) == 168
    assert list(matrix.columns) == FEATURE_COLUMNS
    assert not matrix.isna().any().any()
    assert result.loc[0, "cloud_low_fraction_lag_24h"] == result.loc[0, "cloud_low_fraction"]
    assert result.loc[0, "cloud_low_fraction_delta_1h"] == 0.0


def test_add_model_features_training_drops_rows_without_full_lag_history():
    frame = _raw_weather_frame(periods=30)

    result = add_model_features(frame, mode="training")

    assert len(result) == 6
    assert result.loc[0, "timestamp"] == frame.loc[24, "timestamp"]
    assert not result[DYNAMIC_FEATURE_COLUMNS].isna().any().any()


def test_expanded_feature_columns_are_unique_and_ordered_by_group():
    assert FEATURE_COLUMNS == BASE_FEATURE_COLUMNS + TIME_FEATURE_COLUMNS + DYNAMIC_FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    assert "cloud_low_fraction_lag_1h" in DYNAMIC_FEATURE_COLUMNS
    assert "cloud_low_fraction_rolling_mean_24h" in DYNAMIC_FEATURE_COLUMNS
    assert "cloud_low_fraction_delta_1h" in DYNAMIC_FEATURE_COLUMNS


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
