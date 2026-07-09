# Admin UI e2e (Playwright)

A standalone Playwright suite that drives the LiteLLM Admin UI in a real browser, separate from the Python harness in the sibling folders. It proves the flows a proxy admin actually clicks through, and cross-checks each one against the proxy API so a green test means the browser action really took effect on the server

The single spec here logs in as the proxy admin, creates a virtual key scoped to one model through the create-key modal, and then asserts both halves of the contract: `/key/info` reflects the alias and model the form set (the write persisted), and the generated key serves `/chat/completions` on its scoped model while being denied `key_model_access_denied` on a model outside that scope (the gateway enforces the scope)

## Prerequisites

You need a running proxy that serves the UI on `http://localhost:4000` (the repo's `tests/e2e/docker-compose.yml` brings one up with the `gemini-2.5-flash` and `gpt-5.5` models this spec uses). The admin credentials are username `admin` and password = the proxy master key

## Running

1. Install dependencies and the browser from this folder:

   ```bash
   npm install
   npm run install:browser
   ```

2. Bring up a proxy (from `tests/e2e/`):

   ```bash
   docker compose up -d
   curl -fs http://localhost:4000/health/liveliness
   ```

3. Run the suite:

   ```bash
   npm test
   ```

## Configuration

The base URL and credentials come from the environment so the same spec runs against localhost or a deployed proxy

- `LITELLM_PROXY_URL` (default `http://localhost:4000`)
- `LITELLM_MASTER_KEY` (default `sk-1234`)
- `E2E_UI_ALLOWED_MODEL` (default `gemini-2.5-flash`), the model the key is scoped to
- `E2E_UI_DENIED_MODEL` (default `gpt-5.5`), a model outside the scope used to assert the denial
