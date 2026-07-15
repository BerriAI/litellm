---
name: testing-dashboard-ui
description: Run the LiteLLM Admin dashboard locally and test UI changes end-to-end (sidebar, badges, account-menu toggles, feature-flag-gated pages). Use when verifying any ui/litellm-dashboard change in a browser.
---

# Testing the LiteLLM Admin dashboard end-to-end

## Ports / routing
- Dashboard dev server: `npm run dev` in `ui/litellm-dashboard`, served on **:3000**, root path is `/` (NOT `/ui/` — that 404s on the dev server).
- Proxy: **:4000**. Login page is served by the proxy and redirects back to the dev server after auth.
- Login: username `admin`, password = proxy master key (`sk-1234` in `litellm/proxy/dev_config.yaml`).

## Running the proxy WITH a database (needed for login + UI settings)
The bare proxy has no DB, and admin login fails with "Authentication Error, Not connected to DB!". UI settings writes also need `STORE_MODEL_IN_DB=True`. To get a working login + settings:

1. Start Postgres (Docker is available via `sudo -n docker`):
   `sudo -n docker run -d --name litellm-pg -e POSTGRES_PASSWORD=litellm -e POSTGRES_USER=litellm -e POSTGRES_DB=litellm -p 5432:5432 postgres:16`
2. Start the proxy (use `setsid ... < /dev/null & disown` so it survives the exec call timing out; plain `nohup &` in a timed-out exec gets killed):
   ```
   cd <repo> && DATABASE_URL="postgresql://litellm:litellm@localhost:5432/litellm" STORE_MODEL_IN_DB=True \
     setsid uv run --no-sync python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml \
     --use_v2_migration_resolver > /tmp/proxy.log 2>&1 < /dev/null & disown
   ```
   Use `uv run --no-sync` (system python lacks deps like httpx). Migrations auto-apply on boot (~45s). Verify: `curl -s -o /dev/null -w "%{http_code}" http://localhost:4000/health/readiness` -> 200.

## Feature-flag-gated sidebar items
Some sidebar items only render when a UI setting is on (checked in `leftnav.tsx` filter, e.g. `enableProjectsUI`, `enableChatUI` from `useUISettings().values.enable_projects_ui` / `enable_chat_ui`). Enable via the master-key API (intended path):
```
curl -s -X PATCH http://localhost:4000/update/ui_settings \
  -H "Authorization: Bearer sk-1234" -H "Content-Type: application/json" \
  -d '{"enable_projects_ui": true, "enable_chat_ui": true}'
```
Verify with `GET /get/ui_settings`. Reload the dashboard after changing. (Projects lives under ACCESS CONTROL, Chat under AI GATEWAY.)

## Badges & the "Hide Feature Badges" toggle
- `NewBadge` (blue "New") and `BetaBadge` (gold "Beta") both read the `disableShowBadges` localStorage flag via `useDisableShowBadges`. The account-menu toggle "Hide Feature Badges" (aria-label "Toggle hide feature badges") sets that key and both badges hide/show together, labels stay visible.
- The account menu is the "Account / Admin" button at the bottom of the sidebar.
- In the annotated DOM, a badge renders as `<sup title="Beta">Beta</sup>` / `<sup title="New">New</sup>` next to the nav label — reliable way to assert presence/absence.

## Node version for running dashboard unit tests
System node may be v20.x which breaks vitest (`ERR_REQUIRE_ESM` from vite 7). Use node 22 via nvm before running tests:
`source ~/.nvm/nvm.sh && nvm use 22 && npx vitest run <files>`.

## Lint / pre-commit
- `make bootstrap` first on a fresh clone (installs dashboard node_modules; otherwise `make pre-commit` fails).
- `make pre-commit` before committing (runs prettier + eslint + budgets). Warnings within budget are fine.

## Devin Secrets Needed
None for this UI-only flow. The local proxy master key `sk-1234` comes from `dev_config.yaml`; no external provider secrets are required to test sidebar/badge UI.
