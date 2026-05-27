# Model Card

## Intended Use

Forecast hourly direct and diffuse clear-sky attenuation factors for a Perth solar asset, then convert those factors into a 7-day plane-of-array GTI forecast.

## Inputs

- Low, mid, and high cloud cover fractions
- Total column integrated water vapour
- Sunshine fraction
- Temperature
- Relative humidity fraction
- Surface pressure

## Outputs

- Direct clear-sky attenuation factor
- Diffuse clear-sky attenuation factor
- Derived GTI total, direct, and diffuse plane-of-array components

## Training Strategy

Historical Open-Meteo observations are transformed into clear-sky factors using pvlib clear-sky estimates. A chained multi-output XGBoost model predicts direct attenuation first, then uses that signal while predicting diffuse attenuation.

Interior daylight observations are weighted more heavily than exact boundary observations so the model learns useful variation during operational daylight hours.

## Guardrails

- Predictions are clipped to `[0, 1]`.
- Night-time rows are forced to zero.
- Fully overcast low-cloud rows force the direct factor to zero.
- Forecast batches should pass row count, uniqueness, null, required-column, and hourly-cadence checks before scoring.

## Limitations

- Forecast quality is bounded by Open-Meteo forecast accuracy.
- The current example is configured for a single Perth site.
- No measured plant output is used, so this forecasts irradiance rather than inverter-level AC power.
- The current GTI projection uses an isotropic sky diffuse model; more advanced models may improve site-specific accuracy.
