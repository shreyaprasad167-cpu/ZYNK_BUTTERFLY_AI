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

## Browser frontend

Set `CORS_ORIGINS` in the server's private `.env` to the frontend's exact origin (scheme and host, comma-separated if needed), then restart the API. CORS only permits browser requests; it does not authenticate callers. `127.0.0.1` and the SSH tunnel are for local development. A remote frontend needs a reachable HTTPS API address configured by the server owner.

## Git safety

`.env`, private key files, virtual environments, and CSV files are ignored. Keep real credentials out of Git. The model artifact is included because the API needs it to predict.
