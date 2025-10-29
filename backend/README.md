# Backend — Accent Detection (FastAPI)

This folder contains a minimal FastAPI backend that exposes a single endpoint:

- POST /detect-accent — accepts an uploaded audio file and returns a mock JSON response.

Files:

- `app/main.py` — FastAPI app and endpoint.
- `openapi.json` — exported OpenAPI spec for generating clients.
- `requirements.txt` — Python dependencies.

Run locally (recommended inside a venv):

1. Create and activate a virtual environment (example for bash on Windows):

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

2. Run with uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Notes:

- FastAPI will automatically serve an interactive API docs at `http://localhost:8000/docs`.
- `openapi.json` is included so you can generate a TypeScript client via `@hey-api/openapi-ts` or similar tools.
