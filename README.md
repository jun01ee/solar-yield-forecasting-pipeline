# Solar Yield Forecasting & Operational Pipeline

An end-to-end Machine Learning and Data Engineering pipeline designed to predict solar attenuation factors and compute **Global Tilted Irradiance (GTI)** for physical solar arrays. This project pairs a custom-weighted multi-output machine learning architecture with a high-fidelity physical geometry engine.

---

## 🌌 The Core Challenges & Solutions

### 1. The Atmospheric Problem
* **The Challenge:** Standard regression models struggle to capture the complex, non-linear scattering of light through distinct atmospheric tiers (low, mid, and high cloud decks, water vapor, etc.). Models frequently fail at the physical boundaries—underestimating clear-sky peaks or yielding residual noise during absolute night hours.
* **Our Approach:** * **Custom Sample Weighting:** We apply a higher sample priority weight ($5\times$) to fluid, interior physical points, forcing the tree split criteria to optimize for dynamic atmospheric shifts rather than over-indexing on stable, edge-case zeros.
    * **Post-Processing Overrides:** Direct boundary overrides are applied to the inference matrix to handle physical realities (e.g., clamping outputs to absolute zero when the sunshine fraction is zero or low cloud cover is 100%).

### 2. The Geospatial Problem
* **The Challenge:** Attenuation factors only describe horizontal light tracking. Real-world solar infrastructure relies on Plane-of-Array (POA) irradiance, which is highly sensitive to the array's physical location on Earth, panel tilt, and panel azimuth. Misalignment casts models into perpetual shadow equations.
* **Our Approach:** * **Vector Projection via `pvlib`:** We leverage the `pvlib` astronomy engine, parameterized explicitly for our local asset profile (e.g., Latitude: `-31.95`, Longitude: `115.86`, True North orientation).
    * **Geometric Correction:** Horizontal Direct and Diffuse predictions are dynamically projected into Total Global Tilted Irradiance (GTI) waves using solar zenith and azimuth vectors.

### 3. The Temporal Problem
* **The Challenge:** Live meteorological API forecasts regularly present data synchronization friction. High-frequency time-series data can trigger timezone drift (UTC vs. local operational time) or trigger row-replication broadcast loops during Spark-to-Pandas DataFrame conversions, collapsing a 14-day wave into a single vertical stack.
* **Our Approach:** * **Timezone Hardening:** Timestamps are strictly normalized, stripping timezone metadata before Spark serializations to keep the Catalyst optimizer stable, while enforcing explicit localization (`Australia/Perth`) right before astronomical vector tracking.
    * **Sequential Indexing:** Live forecast arrays are structured using explicit Pandas `date_range` tracking to ensure a continuous, left-to-right 336-hour chronological sequence.

---

## 🏗️ Pipeline Architecture

The pipeline is split into two primary components within Databricks:

* **1. Model Training Notebook:** Implements the `RegressorChain` model paired with our custom atmospheric interior sample weights.
* **2. Production Inference Notebook:** Handles live 14-day Open-Meteo API ingestion, feature standardization, MLflow model serving, and the final `pvlib` geometry correction matrix.
* **Production Gold Layer Table:** Saves the finalized continuous time series directly into a optimized Delta Lake table format.

* **Bronze Layer:** Live 14-day hourly weather forecast ingestion via the Open-Meteo API.
* **Silver Layer:** Automated feature engineering, scaling cloud fractions, and handling missing transmission packets safely via chronological forward/backward fills.
* **Gold Layer:** Scoring via MLflow, execution of geometric corrections, and delivery of production-ready clear-sky attenuation profiles.

---

## 🛠️ Configuration & Deployment

### Dependencies
```bash
pip install xgboost scikit-learn pvlib openmeteo-requests mlflow pandas numpy matplotlib seaborn
```
### Deployment Parameters
To deploy this pipeline for a different asset array, update the following parameters in the production execution cell:
```Python
LATITUDE = -31.95
LONGITUDE = 115.86
TZ = "Australia/Perth"
SURFACE_TILT = 25.0     # Panel tilt angle
SURFACE_AZIMUTH = 0.0   # 0 = True North (Southern Hemisphere)
RUN_ID = "YOUR_MLFLOW_RUN_ID"
```
### Automation
The production notebook is scheduled via Databricks Workflows to execute daily at 05:00 AM AWST, refreshing the 14-day operational outlook before local sunrise.

## 📊 Sample Visualizations
(Placeholder: Insert your exported 14-Day GTI Power Yield Time Series Profile chart here to showcase your smooth diurnal waves, clear-sky direct components, and diffuse scattering profiles for your presentation).