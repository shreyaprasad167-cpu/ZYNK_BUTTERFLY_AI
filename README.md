# Butterfly AI

Butterfly AI is a prototype for exploring how a transport scenario can affect estimated Bengaluru congestion and related city systems. It includes a FastAPI prediction backend and a Streamlit dashboard with an interactive dark map.

## What you need

- Git, or download this repository as a ZIP.
- Python 3.13 recommended for the pinned backend model dependencies.
- Internet access for installing packages and loading the OpenFreeMap map tiles.
- The model file `butterfly_model.joblib`, included in this repository.

No Google Maps API key is needed. The dashboard map uses OpenFreeMap tiles.

## Get the project

Using Git:

```bash
git clone https://github.com/shreyaprasad167-cpu/ZYNK_BUTTERFLY_AI.git
cd ZYNK_BUTTERFLY_AI
```

Or on GitHub, select **Code → Download ZIP**, extract it, then open a terminal in the extracted `ZYNK_BUTTERFLY_AI` folder.

## Run the whole project locally

Run the backend and dashboard in two separate terminal windows. Keep both running while you use the app.

### macOS

In Terminal window 1, from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

In Terminal window 2, from the same project folder:

```bash
source .venv/bin/activate
export BUTTERFLY_API_URL=http://127.0.0.1:8000
python -m pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

### Windows (PowerShell)

Open PowerShell in the project folder. In terminal window 1:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

If PowerShell blocks environment activation, run this once in that PowerShell window, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open a second PowerShell window in the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
$env:BUTTERFLY_API_URL = "http://127.0.0.1:8000"
python -m pip install -r frontend/requirements.txt
python -m streamlit run frontend/app.py
```

Open the Streamlit URL printed in the terminal, usually **http://localhost:8501**. The backend API documentation is at **http://127.0.0.1:8000/docs**.

### Using the dashboard

1. Choose one of the 25 listed areas on the map or in the area selector.
2. Adjust the scenario slider.
3. Select **Run the ripple simulation**.
4. Review the baseline and scenario congestion estimates and the signed ripple cards.

The frontend calls the backend at `BUTTERFLY_API_URL`. The local instructions set this to port 8000. If you change the backend port, set this variable to the matching address.

## Optional: run the backend on IBM LinuxONE

This is for people who have access to an IBM LinuxONE Ubuntu host. On that host, clone the project (or copy the repository files, including `main.py` and `butterfly_model.joblib`) and run from its project folder:

```bash
sudo apt update
sudo apt install -y python3.12-venv python3-numpy python3-pandas python3-scipy python3-sklearn python3-joblib
python3 -m venv --system-site-packages api-venv
source api-venv/bin/activate
python -m pip install -r requirements-linuxone.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

To use that private backend from a Mac, keep the backend running. In a second Mac Terminal window, create an SSH tunnel (replace the placeholders with the host, username, and private-key path supplied to you by your LinuxONE administrator):

```bash
ssh -i ~/path/to/private-key.pem -N -L 8001:127.0.0.1:8000 username@linuxone-host
```

Leave the tunnel terminal open. In another Mac terminal, from the project folder, start the frontend:

```bash
source .venv/bin/activate
export BUTTERFLY_API_URL=http://127.0.0.1:8001
python -m pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

For Windows, open PowerShell and use the Windows path to your private key:

```powershell
ssh -i "$HOME\Downloads\private-key.pem" -N -L 8001:127.0.0.1:8000 username@linuxone-host
```

Keep that window open. In a second PowerShell window, from the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
$env:BUTTERFLY_API_URL = "http://127.0.0.1:8001"
python -m pip install -r frontend/requirements.txt
python -m streamlit run frontend/app.py
```

An SSH tunnel is private to the computer that opened it. Your friends need their own LinuxONE account/access and key to use this option. Otherwise, use the fully local setup above.

## Notes about predictions and data

- The saved model was exported with scikit-learn 1.9.0. Keep the serving scikit-learn version aligned with the one used to export the model. The supplied LinuxONE system package was scikit-learn 1.4.1.post1; it loaded the model and returned a smoke-test prediction with compatibility warnings. Re-export the model with the LinuxONE version or arrange a matching runtime before relying on it for deployment.
- The supplied 25-area CSV labels its rows as **synthetic** and illustrative. The map and sample values are useful for a prototype demonstration, not verified Bengaluru measurements.
- The dashboard sends baseline and changed feature rows to `POST /predict`. Its displayed ripple is the notebook's signed network response to the prediction difference; it is a conceptual relationship, not proof of causation.
- Supabase scenario routes need `SUPABASE_URL` and `SUPABASE_KEY` in a local `.env`. The prediction dashboard does not need Supabase credentials. Never commit real secrets or private SSH keys.

## More details

- [Frontend guide](frontend/README.md)
- [LinuxONE dependency list](requirements-linuxone.txt)
