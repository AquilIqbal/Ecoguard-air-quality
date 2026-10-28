# EcoGuard

EcoGuard is a Flask-based air quality monitoring dashboard for Indian cities. It combines live AQI data, official forecast data, machine-learning-based next-day prediction, city and India map views, and health guidance in one interface.

## Current Features

- Live AQI for the selected city
- Tomorrow AQI prediction using the active XGBoost model
- Three chart modes:
  - `Official`
  - `AI Forecast`
  - `Compare`
- City autocomplete search
- `Use My Location` with reverse-geocoding based city detection
- City and India AQI map views
- India map fallback for non-live stations
- Local cache fallback when upstream API data is temporarily missing
- Light mode and dark mode toggle
- ML model performance panel with real saved metrics

## Dashboard Sections

### 1. Top Bar

- EcoGuard branding
- Theme toggle (`Light Mode` / `Dark Mode`)

Purpose:
- Provides app identity and display preference switching.

### 2. Search and Location Section

- Current city display
- City search input
- Search submit button
- `Use My Location` button

Purpose:
- Lets the user load AQI information for a supported Indian city either manually or by GPS.

### 3. Summary Cards

#### Current AQI

- Shows the live AQI value for the selected city
- Displays the current AQI category

#### Tomorrow's Forecast

- Shows the AI-predicted AQI for tomorrow
- Displays percentage increase or decrease relative to current AQI

#### Model Accuracy

- Shows the active model's `R2` score as a percentage

#### Active Stations

- Indicates the station coverage count used in the dashboard context

### 4. AQI Trend and Forecast Chart

Tabs:
- `Official`: official WAQI daily forecast
- `AI Forecast`: EcoGuard AI-generated forecast
- `Compare`: visual comparison of official and AI forecast

Special handling:
- If official forecast is unavailable for a city, the dashboard now shows that explicitly instead of pretending full official data exists.
- In that case, the AI forecast still uses a full 7-day timeline so the chart remains useful.

### 5. Health Advisory

- Converts AQI level into a readable health message

Purpose:
- Helps users understand the effect of current air quality in practical terms.

### 6. ML Model Performance

Models shown:
- XGBoost
- Ensemble
- Random Forest

Metrics shown:
- `MAE`
- `RMSE`
- `R2`

Purpose:
- Shows comparative model performance and highlights which model is active in production.

### 7. Live AQI Heatmap

Map modes:
- `City View`
- `India View`

Purpose:
- City View focuses on the selected city
- India View gives a country-level AQI overview

India View behavior:
- Live AQI stations are rendered with active data
- When WAQI returns stations with no live AQI (`aqi: "-"`), the dashboard still shows them as fallback markers instead of going blank

## How the System Works

1. The user selects a city through search or geolocation.
2. Flask fetches live AQI and forecast data from WAQI.
3. Current pollutant values are sent into the trained ML model.
4. The ML model predicts tomorrow's AQI.
5. The backend prepares chart data, map data, health advice, and model metrics.
6. The frontend renders cards, chart, maps, and interaction controls.

## Data Sources

- WAQI API:
  - live AQI feed
  - official forecast
  - station map data
- OpenStreetMap Nominatim:
  - reverse geocoding for `Use My Location`
- Local ML assets:
  - trained XGBoost model
  - saved model comparison metrics

## Reliability Handling

EcoGuard now includes fallback handling so the UI remains usable when upstream services are incomplete.

### Cached Data Fallback

The app stores last-known-good responses under `data/cache/` for:
- city AQI data
- forecast data
- city heatmap data
- India heatmap data

If WAQI temporarily fails or returns incomplete data, the app falls back to cached responses instead of rendering empty sections.

### Official Forecast Gaps

Some cities may not have official forecast data at a given moment. In those cases:
- the `Official` chart mode shows a clear unavailable message
- the `AI Forecast` and `Compare` modes still remain usable

### India Map Live Coverage Gaps

WAQI may return many stations with no live AQI value. EcoGuard handles this by:
- using live stations for heat intensity
- still displaying non-live stations as fallback markers

## Tech Stack

- Backend:
  - Flask
  - Requests
  - NumPy
  - Pickle
- Frontend:
  - HTML / Jinja
  - CSS
  - JavaScript
  - Chart.js
  - Leaflet
  - Leaflet Heat
- Machine Learning:
  - XGBoost runtime model loaded from `ml/model_loader.py`

## Project Structure

```text
EcoGuard/
|-- app.py
|-- README.md
|-- requirements.txt
|-- requirements_ml.txt
|-- indian_cities.txt
|-- data/
|   |-- cache/
|-- templates/
|   |-- base.html
|   |-- index.html
|-- static/
|   |-- css/
|   |   |-- style.css
|   |-- js/
|       |-- chart.js
|       |-- map.js
|       |-- search.js
|       |-- location.js
|-- ml/
|   |-- model_loader.py
|   |-- feature_engineering.py
|   |-- preprocess_and_train.py
|   |-- train_all_models.py
|   |-- train_global_model.py
|   |-- xgboost_forecasting.py
|   |-- lstm_forecasting.py
|   |-- process_kaggle_data.py
|   |-- fetch_india_data.py
|   |-- compare_approaches.py
|   |-- models/
|       |-- xgboost_model.pkl
|       |-- model_comparison.csv
|       |-- random_forest_model.pkl
|       |-- random_forest_global.pkl
|       |-- xgboost_global.pkl
|       |-- model_comparison_global.csv
```

## Main Runtime Files

### Backend

- `app.py`
  - main Flask application
  - route handling
  - API integration
  - fallback cache handling
  - chart/map payload preparation

### Templates

- `templates/base.html`
  - base HTML layout
  - shared JS/CSS includes
  - theme toggle logic

- `templates/index.html`
  - main dashboard structure
  - cards, chart, maps, health advisory, model panel

### Frontend Modules

- `static/js/chart.js`
  - chart initialization
  - tab switching
  - `Official`, `AI Forecast`, and `Compare` modes

- `static/js/map.js`
  - city and India map rendering
  - map mode switching
  - fallback marker behavior for India view

- `static/js/search.js`
  - autocomplete dropdown
  - keyboard navigation

- `static/js/location.js`
  - browser geolocation
  - reverse lookup based city loading

- `static/css/style.css`
  - layout and dashboard styling
  - light and dark theme palettes

### ML Runtime

- `ml/model_loader.py`
  - loads the active trained model
  - returns tomorrow AQI prediction
  - returns model info and model comparison metrics

- `ml/models/xgboost_model.pkl`
  - active runtime prediction model

- `ml/models/model_comparison.csv`
  - saved metrics used in the UI model performance panel

## Main Routes

- `/`
  - dashboard page

- `/search-city?q=...`
  - city autocomplete

- `/get-city?lat=...&lon=...`
  - reverse geocoding for location button

- `/aqi-heatmap?city=...`
  - city-level AQI map points

- `/aqi-india-heatmap`
  - India-level AQI map points

- `/model-info`
  - active model metrics API

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If you also need the ML-specific environment:

```bash
pip install -r requirements_ml.txt
```

### 2. Configure the WAQI API key

Set the `WAQI_API_KEY` environment variable before starting the app. For local
development, you can alternatively place the token in `Api key's.txt` in the
project root. That local file is ignored by Git and must never be committed.

If you previously committed or shared the token, revoke it and create a new one.

### 3. Run the application

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

## ML Notes

- The current production model used by the dashboard is `XGBoost`.
- The `ACTIVE` badge in the UI corresponds to the runtime prediction model.
- Other ML scripts in `ml/` are retained for training, experimentation, and dissertation explanation.

## Metric Definitions

- `MAE`:
  - Mean Absolute Error
  - average prediction error magnitude
  - lower is better

- `RMSE`:
  - Root Mean Squared Error
  - penalizes larger errors more strongly
  - lower is better

- `R2`:
  - coefficient of determination
  - indicates how much variance the model explains
  - higher is better

## Important Clarifications

- `Current AQI` is a live AQI reading and can differ from the official daily forecast.
- `Tomorrow's Forecast` is generated by the ML model, not copied from the official forecast.
- If geolocation returns labels like `Pune City`, the backend normalizes them to the supported city name such as `Pune`.

## Academic Context

This project was developed as part of an MSc CA dissertation workflow focused on air quality analytics, forecasting, and user-facing environmental decision support.

## Author

- Aquil Iqbal

## Course

- Dissertation / MSc CA
