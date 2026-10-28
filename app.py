from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import requests
import sys
import os
import re
import json
from datetime import datetime, timedelta

# Add ml folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))

# Import our model loader
try:
    from model_loader import (
        predict_tomorrow_aqi,
        get_model_info,
        get_all_model_metrics,
        get_aqi_category as get_ml_aqi_category
    )
    ML_MODEL_AVAILABLE = True
    print("[OK] ML Model integration successful!")
except ImportError as e:
    print(f"[WARN] ML Model not available: {e}")
    ML_MODEL_AVAILABLE = False

app = Flask(__name__)
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'cache')


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def sanitize_cache_key(key):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', key.lower())


def save_cache(cache_key, payload):
    try:
        ensure_cache_dir()
        cache_path = os.path.join(CACHE_DIR, f"{sanitize_cache_key(cache_key)}.json")
        wrapped = {
            "cached_at": datetime.now().isoformat(),
            "data": payload
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(wrapped, f, ensure_ascii=True)
    except Exception as e:
        print(f"Cache save failed for {cache_key}: {e}")


def load_cache(cache_key):
    try:
        cache_path = os.path.join(CACHE_DIR, f"{sanitize_cache_key(cache_key)}.json")
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            wrapped = json.load(f)
        return wrapped.get("data")
    except Exception as e:
        print(f"Cache load failed for {cache_key}: {e}")
        return None

# -----------------------
# Load API key
# -----------------------
def load_api_key():
    api_key = os.environ.get("WAQI_API_KEY")
    if api_key:
        return api_key.strip()

    for path in ["Api key's.txt", "uploads/Api key's.txt"]:
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except:
            continue
    return None


def load_city_list():
    for path in ["indian_cities.txt", "uploads/indian_cities.txt"]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except:
            continue
    return []


def normalize_city_name(raw_city):
    if not raw_city:
        return None

    city_list = load_city_list()
    if not city_list:
        return raw_city

    raw_clean = raw_city.strip().lower()
    normalized = re.sub(r"\b(city|district|division|municipal corporation|municipality)\b", "", raw_clean)
    normalized = re.sub(r"[^a-z\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    city_lookup = {city.lower(): city for city in city_list}

    if raw_clean in city_lookup:
        return city_lookup[raw_clean]
    if normalized in city_lookup:
        return city_lookup[normalized]

    for city in city_list:
        city_lower = city.lower()
        if city_lower in raw_clean or city_lower in normalized:
            return city

    return raw_city

# -----------------------
# AQI Category
# -----------------------
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# -----------------------
# Health Advisory
# -----------------------
def get_health_advisory(aqi):
    if aqi is None:
        return None

    if aqi <= 50:
        return {"level": "Good", "color": "success",
                "message": "Air quality is good. Safe for all outdoor activities."}
    elif aqi <= 100:
        return {"level": "Satisfactory", "color": "info",
                "message": "Minor breathing discomfort for sensitive individuals."}
    elif aqi <= 200:
        return {"level": "Moderate", "color": "warning",
                "message": "People with asthma should limit outdoor exertion."}
    elif aqi <= 300:
        return {"level": "Poor", "color": "danger",
                "message": "Avoid prolonged outdoor activities."}
    elif aqi <= 400:
        return {"level": "Very Poor", "color": "danger",
                "message": "Serious health effects. Stay indoors."}
    else:
        return {"level": "Severe", "color": "dark",
                "message": "Emergency conditions. Avoid going outside."}

# -----------------------
# Fetch AQI + coordinates
# -----------------------
def fetch_city_data(city):
    API_KEY = load_api_key()
    if not API_KEY:
        print("[ERROR] No API key found")
        cached_city = load_cache(f"city_data_{city}")
        return cached_city

    try:
        url = f"https://api.waqi.info/feed/{city}/?token={API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("status") != "ok":
            print(f"[ERROR] API returned status: {data.get('status')} for {city}")
            return None

        iaqi = data["data"]["iaqi"]
        geo = data["data"]["city"]["geo"]

        result = {
            "current_aqi": data["data"]["aqi"],
            "pm25": iaqi.get("pm25", {}).get("v", 50),
            "pm10": iaqi.get("pm10", {}).get("v", 80),
            "no2": iaqi.get("no2", {}).get("v", 20),
            "so2": iaqi.get("so2", {}).get("v", 10),
            "lat": geo[0],
            "lon": geo[1]
        }
        
        print(f"[OK] Got data for {city}: AQI={result['current_aqi']}, Coords=({result['lat']}, {result['lon']})")
        save_cache(f"city_data_{city}", result)
        return result

    except Exception as e:
        print(f"[ERROR] API error for {city}: {e}")
        cached_city = load_cache(f"city_data_{city}")
        return cached_city

# -----------------------
# Forecast for graph
# -----------------------
def fetch_forecast_data(city):
    API_KEY = load_api_key()
    if not API_KEY:
        cached_forecast = load_cache(f"forecast_{city}")
        if cached_forecast:
            return cached_forecast.get("labels", []), cached_forecast.get("values", [])
        return None

    try:
        url = f"https://api.waqi.info/feed/{city}/?token={API_KEY}"
        data = requests.get(url, timeout=10).json()

        if data.get("status") != "ok":
            return None

        forecast = data["data"].get("forecast", {}).get("daily", {}).get("pm25", [])
        
        if not forecast:
            print(f"[WARN] No forecast data for {city}")
            return None

        # Keep only today+future entries so chart timeline aligns with "tomorrow" card.
        today = datetime.now().date()
        future_days = []
        for day in forecast:
            try:
                d = datetime.strptime(day["day"], "%Y-%m-%d").date()
                if d >= today:
                    future_days.append(day)
            except Exception:
                continue

        labels, values = [], []
        for day in future_days[:7]:
            d = datetime.strptime(day["day"], "%Y-%m-%d")
            labels.append(d.strftime("%d %b"))
            values.append(day["avg"])

        print(f"[OK] Got forecast for {city}: {len(labels)} days")
        save_cache(f"forecast_{city}", {"labels": labels, "values": values})
        return labels, values

    except Exception as e:
        print(f"[WARN] Forecast error for {city}: {e}")
        cached_forecast = load_cache(f"forecast_{city}")
        if cached_forecast:
            return cached_forecast.get("labels", []), cached_forecast.get("values", [])
        return None


def build_predicted_series(actual_series, current_aqi=None, tomorrow_aqi=None):
    """
    Build a smooth predicted series aligned with the displayed timeline.
    """
    if not actual_series:
        return []

    n = len(actual_series)
    safe_actual = [int(round(v)) for v in actual_series]
    start = int(round(current_aqi)) if current_aqi is not None else safe_actual[0]

    if n == 1:
        return [max(0, min(500, start))]

    if tomorrow_aqi is not None:
        lower = max(0, int(start * 0.6))
        upper = min(500, int(start * 1.4))
        day1 = max(lower, min(upper, int(round(tomorrow_aqi))))
    else:
        day1 = safe_actual[1]

    predicted = [start, day1]
    step = day1 - start

    for _ in range(2, n):
        step *= 0.85  # dampen drift so long-range curve is realistic
        next_val = int(round(predicted[-1] + step))
        predicted.append(max(0, min(500, next_val)))

    return predicted

# -----------------------
# Home
# -----------------------
@app.route("/")
def home():
    city = request.args.get("city", "delhi")
    print(f"\n[INFO] Loading page for city: {city}")

    data = fetch_city_data(city)

    current_aqi = category = health = None
    tomorrow_prediction = None
    model_info = None
    model_metrics = []
    lat = lon = None

    if data:
        current_aqi = data["current_aqi"]
        lat, lon = data["lat"], data["lon"]
        category = get_aqi_category(current_aqi)
        health = get_health_advisory(current_aqi)

        # Use ML model for tomorrow's prediction
        if ML_MODEL_AVAILABLE:
            try:
                ml_prediction = predict_tomorrow_aqi(
                    current_aqi=current_aqi,
                    pm25=data["pm25"],
                    pm10=data["pm10"],
                    no2=data["no2"],
                    so2=data["so2"]
                )
                
                tomorrow_prediction = {
                    'aqi': ml_prediction['tomorrow_aqi'],
                    'change': ml_prediction['change_percent'],
                    'category': get_aqi_category(ml_prediction['tomorrow_aqi']),
                    'confidence': ml_prediction['confidence'],
                    'accuracy': ml_prediction.get('model_accuracy', 71)
                }
                
                model_info = get_model_info()
                model_metrics = get_all_model_metrics()
                print(f"[OK] ML Prediction: Tomorrow={tomorrow_prediction['aqi']}, Change={tomorrow_prediction['change']}%")
                
            except Exception as e:
                print(f"[ERROR] ML prediction error: {e}")
                tomorrow_prediction = None

    if ML_MODEL_AVAILABLE and model_info is None:
        try:
            model_info = get_model_info()
            model_metrics = get_all_model_metrics()
        except Exception:
            model_metrics = []

    # Get forecast for chart
    forecast = fetch_forecast_data(city)
    official_forecast_available = bool(forecast and forecast[0] and forecast[1])
    if forecast:
        labels, actual = forecast
    else:
        labels, actual = [], []

    # Ensure chart starts from today and set the first official bar to current AQI.
    if current_aqi:
        today_label = datetime.now().strftime("%d %b")
        if labels:
            if labels[0] != today_label:
                labels = [today_label] + labels
                actual = [int(current_aqi)] + actual
            else:
                actual[0] = int(current_aqi)
        else:
            labels = [today_label]
            actual = [int(current_aqi)]
        labels = labels[:7]
        actual = actual[:7]

    # If official forecast is unavailable, build a 7-day timeline so AI views remain useful.
    if current_aqi and not official_forecast_available:
        today = datetime.now()
        labels = [(today + timedelta(days=i)).strftime("%d %b") for i in range(7)]
        actual = [int(current_aqi)] + [0] * 6

    # Create predicted data for chart
    if labels and actual and len(labels) > 0:
        predicted = build_predicted_series(
            actual_series=actual,
            current_aqi=current_aqi,
            tomorrow_aqi=(tomorrow_prediction['aqi'] if tomorrow_prediction else None)
        )
    else:
        # No forecast from API - create projection from current AQI
        if current_aqi:
            labels = ['Today', 'Tomorrow', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
            actual = [int(current_aqi)] * 7
            predicted = build_predicted_series(
                actual_series=actual,
                current_aqi=current_aqi,
                tomorrow_aqi=(tomorrow_prediction['aqi'] if tomorrow_prediction else None)
            )
        else:
            labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            actual = [0, 0, 0, 0, 0, 0, 0]
            predicted = [0, 0, 0, 0, 0, 0, 0]

    print(f"[INFO] Chart: {len(labels)} points, Actual: {actual[:3]}..., Predicted: {predicted[:3]}...")

    return render_template(
        "index.html",
        selected_city=city,
        current_aqi=current_aqi,
        tomorrow_prediction=tomorrow_prediction,
        category=category,
        health=health,
        model_info=model_info,
        model_metrics=model_metrics,
        labels=labels,
        actual=actual,
        predicted=predicted,
        official_forecast_available=official_forecast_available,
        today_label=datetime.now().strftime("%d %b"),
        tomorrow_label=(datetime.now() + timedelta(days=1)).strftime("%d %b"),
        lat=lat,
        lon=lon
    )

# -----------------------
# GPS reverse lookup
# -----------------------
@app.route("/get-city")
def get_city_from_coords():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        data = requests.get(url, headers={"User-Agent": "EcoGuard"}, timeout=10).json()
        address = data.get("address", {})
        city = (
            address.get("city")
            or address.get("town")
            or address.get("municipality")
            or address.get("county")
            or address.get("city_district")
            or address.get("state_district")
            or address.get("village")
            or address.get("suburb")
            or address.get("hamlet")
        )
        return jsonify({"city": normalize_city_name(city)})
    except:
        return jsonify({"city": None})

# -----------------------
# City autocomplete
# -----------------------
@app.route("/search-city")
def search_city():
    q = request.args.get("q", "").lower()
    if not q:
        return jsonify([])

    try:
        for path in ["indian_cities.txt", "uploads/indian_cities.txt"]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cities = [c.strip() for c in f.readlines()]
                    results = [c for c in cities if c.lower().startswith(q)][:8]
                    return jsonify(results)
            except:
                continue
    except:
        pass
    
    return jsonify([])


@app.route("/aqi-heatmap")
def aqi_heatmap():
    city = request.args.get("city", "delhi")
    data = fetch_city_data(city)

    if not data:
        print(f"[ERROR] No heatmap data for {city}")
        cached_points = load_cache(f"city_heatmap_{city}")
        return jsonify(cached_points or [])

    lat, lon = data["lat"], data["lon"]
    API_KEY = load_api_key()

    # Bounding box around city
    lat1, lon1 = lat - 0.5, lon - 0.5
    lat2, lon2 = lat + 0.5, lon + 0.5

    try:
        url = (
            f"https://api.waqi.info/map/bounds/"
            f"?latlng={lat1},{lon1},{lat2},{lon2}&token={API_KEY}"
        )

        r = requests.get(url, timeout=10)
        stations = r.json().get("data", [])

        points = []
        for s in stations:
            if s.get("aqi") != "-" and s.get("lat") and s.get("lon"):
                points.append({
                    "lat": s["lat"],
                    "lon": s["lon"],
                    "aqi": int(s["aqi"]),
                    "intensity": min(int(s["aqi"]) / 300, 1)
                })

        print(f"[OK] Heatmap for {city}: {len(points)} points")
        save_cache(f"city_heatmap_{city}", points)
        return jsonify(points)

    except Exception as e:
        print(f"[ERROR] Heatmap error: {e}")
        cached_points = load_cache(f"city_heatmap_{city}")
        return jsonify(cached_points or [])


@app.route("/aqi-india-heatmap")
def india_aqi_heatmap():
    API_KEY = load_api_key()
    if not API_KEY:
        cached_points = load_cache("india_heatmap")
        return jsonify(cached_points or [])

    # India bounding box
    lat1, lon1 = 6.5, 68.0
    lat2, lon2 = 37.5, 97.5

    def infer_city_label(station_name):
        if not isinstance(station_name, str) or not station_name.strip():
            return "Unknown location"

        tokens = [t.strip() for t in station_name.split(",") if t.strip()]
        if not tokens:
            return "Unknown location"

        states_and_country = {
            "india", "andhra pradesh", "arunachal pradesh", "assam", "bihar",
            "chhattisgarh", "goa", "gujarat", "haryana", "himachal pradesh",
            "jharkhand", "karnataka", "kerala", "madhya pradesh", "maharashtra",
            "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab",
            "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
            "uttar pradesh", "uttarakhand", "west bengal", "delhi",
            "jammu and kashmir", "ladakh", "puducherry", "chandigarh"
        }
        locality_keywords = {
            "sector", "block", "phase", "road", "rd", "station", "depot", "imt",
            "market", "industrial", "area", "colony", "campus", "office", "near"
        }

        def clean_token(token):
            token = token.split(" - ")[0].strip()
            token = re.sub(r"\(.*?\)", "", token).strip()
            return token

        # Prefer broader place names from right side of the station string.
        for token in reversed(tokens):
            name = clean_token(token)
            if not name:
                continue
            low = name.lower()
            if low in states_and_country:
                continue
            if any(k in low for k in locality_keywords):
                continue
            if re.search(r"\d", name):
                continue
            if len(name) >= 3:
                return name

        # Fallback when all tokens look like locality/station names.
        if len(tokens) >= 2:
            return clean_token(tokens[-2]) or clean_token(tokens[0]) or "Unknown location"
        return clean_token(tokens[0]) or "Unknown location"

    try:
        url = (
            f"https://api.waqi.info/map/bounds/"
            f"?latlng={lat1},{lon1},{lat2},{lon2}&token={API_KEY}"
        )

        r = requests.get(url, timeout=15)
        payload = r.json()
        status = payload.get("status")
        stations = payload.get("data", [])

        if status != "ok":
            print(f"[ERROR] India heatmap API status: {status}")
            return jsonify([])

        if not isinstance(stations, list):
            print("[ERROR] India heatmap data is not a list")
            return jsonify([])

        points = []
        for s in stations:
            if not isinstance(s, dict):
                continue

            lat = s.get("lat")
            lon = s.get("lon")
            raw_aqi = s.get("aqi")
            station_name = s.get("station", {}).get("name") if isinstance(s.get("station"), dict) else None

            if lat is None or lon is None:
                continue

            try:
                lat = float(lat)
                lon = float(lon)
            except (TypeError, ValueError):
                continue

            name = infer_city_label(station_name)
            is_live = False
            aqi = None
            intensity = 0.0

            # WAQI can return '-' for stations without current AQI.
            if raw_aqi not in [None, "-", ""]:
                aqi_str = str(raw_aqi).strip()
                match = re.search(r"\d+", aqi_str)
                if match:
                    aqi = int(match.group())
                    intensity = min(aqi / 300, 1)
                    is_live = True

            points.append({
                "lat": lat,
                "lon": lon,
                "intensity": intensity,
                "aqi": aqi,
                "name": name,
                "live": is_live
            })

        print(f"[OK] India heatmap: {len(points)} points")
        save_cache("india_heatmap", points)
        return jsonify(points)

    except Exception as e:
        print(f"[ERROR] India heatmap error: {e}")
        cached_points = load_cache("india_heatmap")
        return jsonify(cached_points or [])


@app.route("/model-info")
def model_info_api():
    if not ML_MODEL_AVAILABLE:
        return jsonify({"available": False})
    
    try:
        info = get_model_info()
        return jsonify({
            "available": True,
            "name": info['name'],
            "accuracy": info['accuracy_percent'],
            "mae": info['mae'],
            "rmse": info['rmse'],
            "r2": info['r2']
        })
    except:
        return jsonify({"available": False})


if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("   ECOGUARD - Air Quality Monitoring System")
    print("=" * 62 + "\n")
    
    if ML_MODEL_AVAILABLE:
        print("[OK] ML Model: Loaded")
        try:
            info = get_model_info()
            print(f"   Model: {info['name']}")
            print(f"   Accuracy: {info['accuracy_percent']}%")
        except:
            pass
    else:
        print("[WARN] ML Model: Not available")
    
    print("\n[INFO] Starting Flask server...")
    print(f"   Open: http://localhost:5000\n")
    
    app.run(debug=True)
