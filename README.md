# Butterfly AI backend

FastAPI service with the existing Supabase scenario routes and the `/predict` route.

## Local development (Mac)

Use Python 3.13 and the matching dependencies in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API. Supabase routes require valid `SUPABASE_URL` and `SUPABASE_KEY` values in `.env`; prediction loads `butterfly_model.joblib` from this folder by default.

## LinuxONE (s390x)

The currently working LinuxONE virtual environment uses Ubuntu's architecture-compatible scientific Python packages. On a new Ubuntu host, install them and create a virtual environment with access to those system packages:

```bash
sudo apt update
sudo apt install -y python3.12-venv python3-numpy python3-pandas python3-scipy python3-sklearn python3-joblib
python3 -m venv --system-site-packages api-venv
source api-venv/bin/activate
python -m pip install -r requirements-linuxone.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

The committed model was saved with scikit-learn 1.9.0. The current LinuxONE package is scikit-learn 1.4.1.post1; it emitted compatibility warnings but the smoke prediction succeeded. Re-train/export the pipeline with the same scikit-learn version used by the serving environment and the verified training dataset before treating model outputs as production-ready. The CSV supplied with this task identifies its rows as `synthetic` and says they are illustrative, so it is not suitable evidence for real-zone accuracy.

## Frontend integration

The Streamlit frontend is in `frontend/` and calls the existing FastAPI `POST /predict` route from the Streamlit server. It sends the baseline and scenario feature rows, compares the two congestion predictions, and applies the signed ripple graph in the frontend. Since the API request is server-to-server, this Streamlit setup does not need browser CORS configuration.

To run the frontend on your Mac while FastAPI runs on LinuxONE, keep the SSH tunnel to local port 8001 open, then run:

```bash
python3 -m venv frontend/.venv
source frontend/.venv/bin/activate
python -m pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

The frontend defaults to `http://127.0.0.1:8001`, matching the Mac SSH tunnel forwarding to LinuxONE port 8000. If Streamlit runs on LinuxONE beside FastAPI, set `BUTTERFLY_API_URL=http://127.0.0.1:8000`. For any other host, configure `BUTTERFLY_API_URL` to the backend's reachable address; the local SSH tunnel is private to the Mac where it runs. The map uses OpenFreeMap dark tiles and requires internet access, but no map API key.

The supplied 25-area CSV is marked synthetic and illustrative. Do not present its scores as verified Bengaluru measurements.

## Git safety

`.env`, private key files, virtual environments, and CSV files are ignored. Keep real credentials out of Git. The model artifact is included because the API needs it to predict.
