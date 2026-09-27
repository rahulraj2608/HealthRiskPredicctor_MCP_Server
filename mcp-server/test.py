import os
import requests
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file if available
load_dotenv()


def verify_all_apis(test_city="Tokyo"):
    print("==================================================")
    print("       TESTING API CONNECTIONS & STATUS           ")
    print("==================================================\n")

    # 1. Test Geocoding API
    print("1. Testing Open-Meteo Geocoding API...")
    encoded_city = requests.utils.quote(test_city)
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1&language=en&format=json"
    country_code = "US"
    try:
        r = requests.get(geo_url, timeout=10)
        if r.status_code == 200 and "results" in r.json() and len(r.json()["results"]) > 0:
            res = r.json()["results"][0]
            lat, lon = float(res["latitude"]), float(res["longitude"])
            country_code = res.get("country_code", "US").upper()
            location_label = f"{res.get('name')}, {res.get('country', '')}"
            print(f"   [SUCCESS] Status 200 OK — Found '{location_label}' (Lat: {lat}, Lon: {lon}, Country Code: {country_code})")
        else:
            print("   [FAILED] Geocoding API returned no results. Defaulting to fallback coordinates.")
            lat, lon = 40.7128, -74.0060
    except Exception as e:
        print(f"   [ERROR] Geocoding connection failed: {e}")
        lat, lon = 40.7128, -74.0060

    # 2. Test Weather API (Historical Archive Endpoint)
    print("\n2. Testing Open-Meteo Weather API (Archive Endpoint)...")
    weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"daily=temperature_2m_mean,temperature_2m_max,precipitation_sum,relative_humidity_2m_mean&"
        f"start_date=2026-01-01&end_date=2026-01-07&timezone=auto"
    )
    try:
        r = requests.get(weather_url, timeout=10)
        if r.status_code == 200:
            temps = r.json().get("daily", {}).get("temperature_2m_mean", [])
            temp_val = temps[-1] if temps else "N/A"
            print(f"   [SUCCESS] Status 200 OK — Fetched Mean Temperature: {temp_val}°C")
        else:
            print(f"   [FAILED] Weather API returned status code {r.status_code}")
    except Exception as e:
        print(f"   [ERROR] Weather connection failed: {e}")

    # 3. Test Air Quality API (Hourly Endpoint)
    print("\n3. Testing Open-Meteo Air Quality API (Hourly Endpoint)...")
    air_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&"
        f"hourly=pm2_5,us_aqi&"
        f"forecast_days=1&timezone=auto"
    )
    try:
        r = requests.get(air_url, timeout=10)
        if r.status_code == 200:
            hourly_data = r.json().get("hourly", {})
            pm25_list = [v for v in hourly_data.get("pm2_5", []) if v is not None]
            aqi_list = [v for v in hourly_data.get("us_aqi", []) if v is not None]

            pm25_val = round(sum(pm25_list) / max(len(pm25_list), 1), 2) if pm25_list else "N/A"
            aqi_val = max(aqi_list) if aqi_list else "N/A"

            print(f"   [SUCCESS] Status 200 OK — Fetched AQI Max: {aqi_val}, PM2.5 Avg: {pm25_val} µg/m³")
        else:
            print(f"   [FAILED] Air Quality API returned status code {r.status_code}")
    except Exception as e:
        print(f"   [ERROR] Air Quality connection failed: {e}")

    # 4. Test World Bank Open Data API (Scanning last 10 records for non-null values)
    print("\n4. Testing World Bank Open Data API...")
    uhc_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/SH.UHC.SRVS.CV.XD?format=json&mrv=10"
    food_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/SN.ITK.MSFI.ZS?format=json&mrv=10"
    try:
        r_uhc = requests.get(uhc_url, timeout=10)
        r_food = requests.get(food_url, timeout=10)

        healthcare_index = "N/A"
        food_security_index = "N/A"

        if r_uhc.status_code == 200 and len(r_uhc.json()) > 1 and r_uhc.json()[1]:
            for entry in r_uhc.json()[1]:
                if entry.get("value") is not None:
                    healthcare_index = round(float(entry["value"]) / 100.0, 4)
                    break

        if r_food.status_code == 200 and len(r_food.json()) > 1 and r_food.json()[1]:
            for entry in r_food.json()[1]:
                if entry.get("value") is not None:
                    food_security_index = round((100.0 - float(entry["value"])) / 100.0, 4)
                    break

        print(f"   [SUCCESS] World Bank Data for '{country_code}' — Healthcare Index: {healthcare_index}, Food Security Index: {food_security_index}")
    except Exception as e:
        print(f"   [ERROR] World Bank API connection failed: {e}")

    # 5. Test Gemini 3.6 Flash API
    print("\n5. Testing Gemini 3.6 Flash API...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("   [WARNING] GEMINI_API_KEY environment variable is missing!")
        print("   -> Add GEMINI_API_KEY to your .env file or export it in your terminal.")
    else:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents="Hello, reply with 'API Connected Successfully'",
            )
            print(f"   [SUCCESS] Gemini Response: {response.text.strip()}")
        except Exception as e:
            print(f"   [ERROR] Gemini API call failed: {e}")


if __name__ == "__main__":
    verify_all_apis("Tokyo")