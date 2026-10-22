# IITM TDS Invoice Action Agent

A2A 1.0 HTTP+JSON invoice decision agent. Configure `A2A_BEARER_TOKEN`, `MODEL_API_KEY`, optional `MODEL_BASE_URL` / `MODEL_NAME`, and `DATABASE_PATH`; run with `uvicorn app:app --host 0.0.0.0 --port 8000`. The deployment platform must provide public HTTPS and forward requests without redirects.

The model receives the full submitted batch and is called once per batch using structured JSON output. Proposals are independently validated and persisted in SQLite. No action is executed before a matching results continuation arrives. Set `DATABASE_PATH` to durable storage in deployment.
