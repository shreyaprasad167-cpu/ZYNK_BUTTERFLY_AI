# Butterfly AI frontend

A Streamlit city-systems dashboard for the existing Butterfly AI FastAPI backend. The welcome screen uses a quiet night-sky treatment, an animated butterfly, and a letter-by-letter “BUTTERFLY AI” reveal. It opens into a neighbourhood-first Bengaluru map, followed by a simple commute scenario and the resulting model estimate and ripple. The copy and layout aim to feel like a thoughtful city guide rather than a generic AI console.

`preview.html` demonstrates the welcome transition and dashboard layout. Its map uses interactive dark OpenFreeMap street tiles with all 25 CSV neighbourhood markers; you can drag, zoom, search, and open marker details. The preview's simulation remains visual only. Run `app.py` for the live map selection and backend predictions. The map style is from OpenFreeMap, based on OpenStreetMap, and does not require a Google Maps or CARTO API key. Map tiles require an internet connection.

## Run it locally

From the repository root, install the frontend packages and start Streamlit:

```bash
python3 -m venv frontend/.venv
source frontend/.venv/bin/activate
python -m pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

By default, the app calls `http://127.0.0.1:8001`, matching the Mac SSH tunnel you set up to forward local port 8001 to LinuxONE port 8000. Keep that tunnel open while using the frontend. If Streamlit runs on the same LinuxONE host as FastAPI, set `BUTTERFLY_API_URL` to `http://127.0.0.1:8000` instead. Set the variable if your backend uses another reachable address. Example:

```bash
export BUTTERFLY_API_URL="http://127.0.0.1:8000"
```

Keep the FastAPI service running in a separate terminal. If the frontend runs on another computer, the API must have a reachable address from that computer; the developer's SSH tunnel on their own Mac is not a shared address. The frontend makes API calls from the Streamlit server, so browser CORS is not needed for this setup.

## Data note

`data/Butterfly_AI_25_Areas.csv` is the supplied area file. It explicitly labels the 25 rows as `synthetic` and illustrative, not verified Bengaluru statistics. The dashboard keeps that label visible and treats the map and reference metrics as a prototype. Replace the CSV with verified, appropriately licensed data before using it as a real city dashboard.

## How the scenario works

The slider shifts a percentage-point share from `estimated_private_vehicle_share` to `estimated_public_transport_share`, matching the scenario in the supplied notebook. It sends the original and adjusted 36-feature rows to `/predict`. The prediction difference is propagated from `Traffic` through the notebook's signed edge weights for three rounds. The app does not substitute a made-up score if the API is offline.
