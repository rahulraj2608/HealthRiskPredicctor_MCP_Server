import os
import sys
import warnings
from datetime import datetime, timedelta
from dotenv import load_dotenv

import joblib
import pandas as pd
import requests
from fastmcp import FastMCP
from google import genai

# Load environment variables from .env file
load_dotenv()
warnings.filterwarnings("ignore")

# Initialize FastMCP Server
mcp = FastMCP("Personalized City Health & Risk Predictor")

# ------------------------------------------------------------------
# Safe ML Model Bundle Loader
# ------------------------------------------------------------------
MODEL_LINEAR_PATH = "health_models_bundle.joblib"
MODEL_HEAT_PATH = "heat_admissions_xgb_bundle.joblib"

respiratory_model = None
cardio_model = None
linear_feature_order = []
heat_xgb_model = None
heat_feature_order = []

try:
    if os.path.exists(MODEL_LINEAR_PATH):
        linear_bundle = joblib.load(MODEL_LINEAR_PATH)
        respiratory_model = linear_bundle.get("respiratory_model")
        cardio_model = linear_bundle.get("cardio_model")
        linear_feature_order = linear_bundle.get("features", [])
    else:
        print(f"Warning: '{MODEL_LINEAR_PATH}' not found. Default baseline predictions will be used.")

    if os.path.exists(MODEL_HEAT_PATH):
        heat_bundle = joblib.load(MODEL_HEAT_PATH)
        heat_xgb_model = heat_bundle.get("model")
        heat_feature_order = heat_bundle.get("features", [])
    else:
        print(f"Warning: '{MODEL_HEAT_PATH}' not found. Default baseline predictions will be used.")
except Exception as e:
    print(f"Warning: Error loading model files ({e}). Operating with baseline predictions.")


# ------------------------------------------------------------------
# Geocoding Helper: City Name -> Lat, Lon, Location Label, Country Code
# ------------------------------------------------------------------
def get_coordinates_from_city(city_name: str) -> tuple[float, float, str, str]:
    """
    Geocodes a city name to (latitude, longitude, formatted_location_string, country_code)
    using the Open-Meteo Geocoding API.
    """
    encoded_city = requests.utils.quote(city_name)
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1&language=en&format=json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            lat = float(results[0]["latitude"])
            lon = float(results[0]["longitude"])
            country_code = results[0].get("country_code", "US").upper()
            location_label = f"{results[0].get('name')}, {results[0].get('country', '')}"
            return lat, lon, location_label, country_code
        else:
            print(f"Warning: City '{city_name}' not found. Defaulting to New York, United States.")
            return 40.7128, -74.0060, "New York, United States", "US"
    except Exception as e:
        print(f"Warning: Geocoding API error ({e}). Defaulting to New York, United States.")
        return 40.7128, -74.0060, "New York, United States", "US"


# ------------------------------------------------------------------
# World Bank API Helper: Healthcare Access & Food Security Indices
# ------------------------------------------------------------------
def fetch_world_bank_indices(country_code: str) -> tuple[float, float]:
    """
    Fetches Healthcare Access Index and Food Security Index from World Bank API
    using ISO country code (e.g., 'JP', 'US', 'BD').
    Fetches the last 10 records (mrv=10) and uses the most recent non-null value.
    Returns normalized values scaled between 0.0 and 1.0.
    """
    healthcare_index = 0.75  # Default baseline fallback
    food_security_index = 0.70  # Default baseline fallback

    if not country_code:
        return healthcare_index, food_security_index

    # 1. Healthcare Access Index (UHC Service Coverage Index: SH.UHC.SRVS.CV.XD)
    uhc_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/SH.UHC.SRVS.CV.XD?format=json&mrv=10"
    try:
        r = requests.get(uhc_url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if len(data) > 1 and data[1]:
                for entry in data[1]:
                    if entry.get("value") is not None:
                        healthcare_index = round(float(entry["value"]) / 100.0, 4)
                        break
    except Exception as e:
        print(f"Warning: World Bank Healthcare API error ({e}). Using default fallback.")

    # 2. Food Security Index (100% - Food Insecurity %: SN.ITK.MSFI.ZS)
    food_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/SN.ITK.MSFI.ZS?format=json&mrv=10"
    try:
        r = requests.get(food_url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if len(data) > 1 and data[1]:
                for entry in data[1]:
                    if entry.get("value") is not None:
                        food_security_index = round((100.0 - float(entry["value"])) / 100.0, 4)
                        break
    except Exception as e:
        print(f"Warning: World Bank Food Security API error ({e}). Using default fallback.")

    return healthcare_index, food_security_index


# ------------------------------------------------------------------
# Environmental Data Helper (Fixes Open-Meteo 503 & 400 API Errors)
# ------------------------------------------------------------------
def fetch_live_environmental_data(latitude: float, longitude: float) -> dict:
    """
    Fetches weather & air quality data dynamically for geocoded coordinates.
    - Weather: Uses Historical Archive API (archive-api.open-meteo.com) to solve 503 limits on 105-day ranges.
    - Air Quality: Uses hourly parameters to solve 400 Bad Request errors.
    """
    today = datetime.now().date()
    start_date = (today - timedelta(days=105)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    # 1. Historical Archive Endpoint for Weather
    weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}&"
        f"daily=temperature_2m_mean,temperature_2m_max,precipitation_sum,relative_humidity_2m_mean&"
        f"start_date={start_date}&end_date={end_date}&timezone=auto"
    )

    # 2. Hourly Endpoint for Air Quality
    air_quality_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={latitude}&longitude={longitude}&"
        f"hourly=pm2_5,us_aqi&"
        f"start_date={start_date}&end_date={end_date}&timezone=auto"
    )

    try:
        w_res = requests.get(weather_url, timeout=12)
        w_res.raise_for_status()
        w_resp = w_res.json().get("daily", {})

        a_res = requests.get(air_quality_url, timeout=12)
        a_res.raise_for_status()
        a_resp = a_res.json().get("hourly", {})

        def sanitize_list(raw_list, default_val):
            if not raw_list:
                return [default_val]
            return [v if v is not None else default_val for v in raw_list]

        temp = sanitize_list(w_resp.get("temperature_2m_mean"), 25.0)
        temp_max = sanitize_list(w_resp.get("temperature_2m_max"), 30.0)
        precip = sanitize_list(w_resp.get("precipitation_sum"), 0.0)
        humidity = sanitize_list(w_resp.get("relative_humidity_2m_mean"), 60.0)

        # Aggregate 24-hour hourly blocks into daily averages/maxes
        raw_pm25 = a_resp.get("pm2_5", [])
        raw_aqi = a_resp.get("us_aqi", [])

        pm25 = []
        for i in range(0, len(raw_pm25), 24):
            chunk = [v for v in raw_pm25[i : i + 24] if v is not None]
            if chunk:
                pm25.append(sum(chunk) / len(chunk))
        if not pm25:
            pm25 = [15.0]

        aqi = []
        for i in range(0, len(raw_aqi), 24):
            chunk = [v for v in raw_aqi[i : i + 24] if v is not None]
            if chunk:
                aqi.append(max(chunk))
        if not aqi:
            aqi = [50.0]

        temp_mean = sum(temp) / max(len(temp), 1)

        heat_wave_days_curr = sum(1 for t in temp_max[-7:] if t > 32.0)
        heat_wave_days_lag1 = sum(1 for t in temp_max[-56:-49] if t > 32.0) if len(temp_max) >= 56 else 0.0
        heat_wave_days_lag2 = sum(1 for t in temp_max[-105:-98] if t > 32.0) if len(temp_max) >= 105 else 0.0

        extreme_events = sum(1 for p, t in zip(precip[-7:], temp_max[-7:]) if p > 25 or t > 35)

        idx_lag1 = -50 if len(temp) >= 50 else -1
        idx_lag2 = -99 if len(temp) >= 99 else -1

        return {
            "temperature_celsius": float(temp[-1]),
            "temp_anomaly_celsius": float(temp[-1] - temp_mean),
            "precipitation_mm": float(precip[-1]),
            "humidity": float(humidity[-1]),
            "pm25_ugm3": float(pm25[-1]),
            "air_quality_index": float(aqi[-1]),
            "heat_wave_days": float(heat_wave_days_curr),
            "extreme_weather_events": float(extreme_events),

            "temperature_celsius_lag_1": float(temp[idx_lag1]),
            "temp_anomaly_celsius_lag_1": float(temp[idx_lag1] - temp_mean),
            "heat_wave_days_lag_1": float(heat_wave_days_lag1),
            "pm25_ugm3_lag_1": float(pm25[idx_lag1] if len(pm25) >= abs(idx_lag1) else pm25[-1]),
            "humidity_lag_1": float(humidity[idx_lag1] if len(humidity) >= abs(idx_lag1) else humidity[-1]),

            "temperature_celsius_lag_2": float(temp[idx_lag2]),
            "temp_anomaly_celsius_lag_2": float(temp[idx_lag2] - temp_mean),
            "heat_wave_days_lag_2": float(heat_wave_days_lag2),
            "pm25_ugm3_lag_2": float(pm25[idx_lag2] if len(pm25) >= abs(idx_lag2) else pm25[-1]),
            "humidity_lag_2": float(humidity[idx_lag2] if len(humidity) >= abs(idx_lag2) else humidity[-1]),
        }
    except Exception as e:
        print(f"Warning: Environmental API fetch failed ({e}). Using default fallback parameters.")
        return {
            "temperature_celsius": 28.0, "temp_anomaly_celsius": 1.5,
            "precipitation_mm": 5.0, "humidity": 65.0, "pm25_ugm3": 35.0,
            "air_quality_index": 85.0, "heat_wave_days": 2.0, "extreme_weather_events": 1.0,
            "temperature_celsius_lag_1": 25.0, "temperature_celsius_lag_2": 22.0,
            "temp_anomaly_celsius_lag_1": 0.5, "temp_anomaly_celsius_lag_2": -1.0,
            "heat_wave_days_lag_1": 1.0, "heat_wave_days_lag_2": 0.0,
            "pm25_ugm3_lag_1": 30.0, "pm25_ugm3_lag_2": 25.0,
            "humidity_lag_1": 60.0, "humidity_lag_2": 55.0,
        }


# ------------------------------------------------------------------
# Gemini 3.6 Flash API Helper: Personalized Medical Advisory
# ------------------------------------------------------------------
def generate_gemini_precautions(
    patient_info: dict,
    env_data: dict,
    predictions: dict
) -> str:
    """Generates tailored medical precautions using Gemini 3.6 Flash."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Warning: GEMINI_API_KEY environment variable is not set. Unable to invoke Gemini API."

    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
You are an expert Clinical Epidemiologist and Environmental Health Specialist. 
Analyze the following patient profile, location environmental conditions, socio-economic factors, and machine learning model predictions, then generate tailored personalized medical precautions.

### PATIENT PERSONAL PROFILE
- City/Location: {patient_info['location_label']} (Lat {patient_info['latitude']}, Lon {patient_info['longitude']})
- Age: {patient_info['age']} years old
- Height: {patient_info['height_cm']} cm
- Weight: {patient_info['weight_kg']} kg
- Calculated BMI: {patient_info['bmi']}

### REGIONAL SOCIOECONOMIC INDICES (World Bank Data)
- Healthcare Access Index: {patient_info['healthcare_access_index']}
- Food Security Index: {patient_info['food_security_index']}

### REAL-TIME ENVIRONMENTAL CONDITIONS
- Temperature: {env_data['temperature_celsius']}°C (Anomaly: {env_data['temp_anomaly_celsius']}°C)
- Humidity: {env_data['humidity']}%
- PM2.5 Level: {env_data['pm25_ugm3']} µg/m³ (AQI: {env_data['air_quality_index']})
- Current Heat Wave Days: {env_data['heat_wave_days']}
- 7-Week Lag Heat Wave Days: {env_data['heat_wave_days_lag_1']}
- 14-Week Lag Heat Wave Days: {env_data['heat_wave_days_lag_2']}

### REGIONAL ML MODEL RISK PREDICTIONS
(Dataset Baseline Averages: Heat Admissions = 6.84, Cardio Mortality = 30.69, Respiratory Disease Rate = 75.64)
- Predicted Respiratory Disease Rate: {predictions['respiratory_disease_rate']}
- Predicted Cardio Mortality Rate: {predictions['cardio_mortality_rate']}
- Predicted Heat Admissions Count: {predictions['heat_related_admissions']}

### YOUR TASK:
Provide a structured, personalized medical advisory covering:
1. **Personalized Overall Risk Level**: (Low, Moderate, High, or Critical) taking into account the patient's age and BMI combined with local environmental indicators.
2. **Targeted Precautions**: 3–5 actionable, concrete precautions specific to this individual's age and BMI profile in this specific climate.
3. **Symptoms to Watch For**: Specific early warning signs this individual should monitor given current local conditions.
4. **Physiological Explanation**: A brief explanation of how current environmental conditions interact with their age and BMI.
"""
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error contacting Gemini API: {e}"


# ------------------------------------------------------------------
# MCP Tool Definition
# ------------------------------------------------------------------
@mcp.tool()
def predict_personalized_health_outcomes(
    city: str,
    age: float,
    weight_kg: float,
    height_cm: float,
    healthcare_access_index: float | None = None,
    food_security_index: float | None = None,
) -> dict:
    """
    Accepts a city name, geocodes coordinates dynamically, fetches real-time World Bank indices,
    fetches environmental metrics, runs ML models, and queries Gemini 3.6 Flash for personalized precautions.
    """
    # 1. Geocode City Name -> Lat, Lon, Location String, Country Code
    latitude, longitude, location_label, country_code = get_coordinates_from_city(city)

    # 2. Fetch World Bank Healthcare Access & Food Security Indices if not overridden
    auto_healthcare, auto_food = fetch_world_bank_indices(country_code)
    final_healthcare = healthcare_access_index if healthcare_access_index is not None else auto_healthcare
    final_food_security = food_security_index if food_security_index is not None else auto_food

    # 3. Calculate Body Mass Index (BMI)
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 2) if height_m > 0 else 0.0

    now = datetime.now()
    year = now.year
    month = now.month
    week = now.isocalendar()[1]

    # 4. Fetch environmental data dynamically using geocoded lat and lon
    env = fetch_live_environmental_data(latitude, longitude)

    raw_input = {
        "healthcare_access_index": final_healthcare,
        "food_security_index": final_food_security,
        "pm25_ugm3": env["pm25_ugm3"],
        "air_quality_index": env["air_quality_index"],
        "temperature_celsius": env["temperature_celsius"],
        "temp_anomaly_celsius": env["temp_anomaly_celsius"],
        "heat_wave_days": env["heat_wave_days"],
        "extreme_weather_events": env["extreme_weather_events"],
        "precipitation_mm": env["precipitation_mm"],
        "humidity": env["humidity"],
        "year": year,
        "month": month,
        "week": week,
        "latitude": latitude,
        "longitude": longitude,
        "temperature_celsius_lag_1": env["temperature_celsius_lag_1"],
        "temperature_celsius_lag_2": env["temperature_celsius_lag_2"],
        "temp_anomaly_celsius_lag_1": env["temp_anomaly_celsius_lag_1"],
        "temp_anomaly_celsius_lag_2": env["temp_anomaly_celsius_lag_2"],
        "heat_wave_days_lag_1": env["heat_wave_days_lag_1"],
        "heat_wave_days_lag_2": env["heat_wave_days_lag_2"],
        "pm25_ugm3_lag_1": env["pm25_ugm3_lag_1"],
        "pm25_ugm3_lag_2": env["pm25_ugm3_lag_2"],
        "humidity_lag_1": env["humidity_lag_1"],
        "humidity_lag_2": env["humidity_lag_2"],
    }

    # 5. ML Model Inference with baseline fallbacks
    resp_pred, cardio_pred, heat_pred = 75.64, 30.69, 6.84

    if respiratory_model and cardio_model and linear_feature_order:
        linear_df = pd.DataFrame([raw_input])[linear_feature_order]
        resp_pred = respiratory_model.predict(linear_df)[0]
        cardio_pred = cardio_model.predict(linear_df)[0]

    if heat_xgb_model and heat_feature_order:
        heat_df = pd.DataFrame([raw_input])[heat_feature_order]
        heat_pred = heat_xgb_model.predict(heat_df)[0]

    predictions = {
        "respiratory_disease_rate": round(float(resp_pred), 4),
        "cardio_mortality_rate": round(float(cardio_pred), 4),
        "heat_related_admissions": round(float(heat_pred), 4),
    }

    patient_info = {
        "city_query": city,
        "location_label": location_label,
        "country_code": country_code,
        "latitude": latitude,
        "longitude": longitude,
        "healthcare_access_index": final_healthcare,
        "food_security_index": final_food_security,
        "age": age,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "bmi": bmi,
    }

    # 6. Generate Gemini Medical Advisory
    gemini_advisory = generate_gemini_precautions(patient_info, env, predictions)

    return {
        "patient_profile": patient_info,
        "fetched_environmental_data": env,
        "predictions": predictions,
        "gemini_personalized_precautions": gemini_advisory,
    }


# ------------------------------------------------------------------
# Entry Point (Supports Interactive Terminal Mode & FastMCP Mode)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # If launched directly in interactive terminal, prompt for manual user test
    if sys.stdin.isatty():
        print("==================================================")
        print("   PERSONALIZED HEALTH & RISK PREDICTOR SYSTEM   ")
        print("==================================================\n")
        try:
            print("Please enter target city and patient details:")
            city_in = input(" -> City Name (default 'Tokyo'): ").strip() or "Tokyo"
            age_in = float(input(" -> Age in years (default '45'): ") or "45")
            weight_in = float(input(" -> Weight in kg (default '75'): ") or "75")
            height_in = float(input(" -> Height in cm (default '175'): ") or "175")

            print("\nEvaluating location, fetching World Bank indices & Open-Meteo weather, and executing Gemini 3.6 Flash...\n")
            results = predict_personalized_health_outcomes(
                city=city_in,
                age=age_in,
                weight_kg=weight_in,
                height_cm=height_in,
            )

            print("--- PATIENT PROFILE & LOCATION ---")
            for k, v in results["patient_profile"].items():
                print(f"  {k}: {v}")

            print("\n--- FETCHED REAL-TIME ENVIRONMENTAL DATA ---")
            for k, v in results["fetched_environmental_data"].items():
                print(f"  {k}: {v}")

            print("\n--- ML MODEL REGIONAL PREDICTIONS ---")
            for k, v in results["predictions"].items():
                print(f"  {k}: {v}")

            print("\n==================================================")
            print("   GEMINI 3.6 FLASH PERSONALIZED MEDICAL ADVISORY ")
            print("==================================================")
            print(results["gemini_personalized_precautions"])
            print("\n==================================================\n")

        except (KeyboardInterrupt, EOFError, ValueError) as err:
            print(f"\nTerminal prompt skipped or interrupted: {err}")

    print("Starting FastMCP Server listener...")
    mcp.run()