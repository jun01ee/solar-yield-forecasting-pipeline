# CV Project Brief

## Solar Yield Forecasting MLOps Pipeline

Built an end-to-end Databricks Free Edition pipeline that forecasts solar irradiance attenuation and converts it into plane-of-array Global Tilted Irradiance for a Perth solar asset.

## What It Demonstrates

- Data engineering: hourly Open-Meteo ingestion, schema normalization, time-series validation, and Bronze/Silver/Gold layer design.
- Machine learning: multi-output XGBoost regression with RegressorChain dependency modelling and sample weighting for physically meaningful daylight observations.
- MLOps: MLflow experiment tracking, model artifact loading, scheduled inference notebook design, and reproducible dependency files.
- Domain modelling: pvlib clear-sky modelling, solar position calculation, and tilted-panel irradiance projection.
- Software engineering: reusable Python package, unit-tested feature logic, configuration validation, and safer separation of secrets from source code.

## Suggested CV Bullet

Developed a Databricks-based solar yield forecasting pipeline using Open-Meteo, pvlib, XGBoost, and MLflow; engineered weather-to-irradiance features, registered a chained multi-output model, and scheduled daily inference to produce 7-day plane-of-array GTI forecasts for operational monitoring.

Extended the pipeline from static tabular regression to time-aware irradiance forecasting by adding lagged atmospheric features, rolling cloud dynamics, chronological backtesting, and model comparison against persistence and clear-sky baselines.
