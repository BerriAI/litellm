# Team Admin Permissions Validation - Setup Guide

This guide explains how to set up the environment to validate that a team admin can perform the required actions.

## What You Need to Set Up

### 1. Database

The LiteLLM proxy uses **PostgreSQL** or **SQLite** for storing teams, keys, models, and memberships.

**Option A: Use default (Neon dev database)**  
If you run the proxy without `DATABASE_URL`, it connects to a default Neon dev database bundled with `litellm-proxy-extras`. This works for quick local testing.

**Option B: Use your own PostgreSQL**
```bash
# Create a database (e.g. via Docker)
docker run -d --name litellm-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=litellm -p 5432:5432 postgres:15

# Set DATABASE_URL
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/litellm"
```

**Option C: Use SQLite (simplest for local dev)**
```bash
export DATABASE_URL="sqlite:///./litellm.db"
```

### 2. Proxy Configuration

Create or update a config file (e.g. `dev_config.yaml`) with:

```yaml
model_list:
  - model_name: openai/gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY  # or your key

general_settings:
  master_key: sk-1234  # Proxy admin key
  store_virtual_keys: true  # Required for team keys

litellm_settings:
  drop_params: true
  telemetry: false
```

### 3. Start the Proxy

```bash
poetry run litellm --config dev_config.yaml --port 4000
```

Wait for `/health` to return 200 (migrations run on startup, ~15–20 seconds).

### 4. Create a Team and Team Admin

**Step 1: Create a team (as proxy admin)**
```bash
curl -X POST 'http://localhost:4000/team/new' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{"team_id": "team-validation-test", "team_alias": "Validation Team"}'
```

**Step 2: Create a user and add as team admin**

If using **SSO/JWT**: Log in via the UI, then use the proxy admin to update the team membership to make the user an admin.

If using **internal auth** (no SSO):
```bash
# Create user (if your setup supports it)
# Add user to team as admin
curl -X POST 'http://localhost:4000/team/member_add' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "team-validation-test",
    "user_id": "<USER_ID>",
    "role": "admin"
  }'
```

**Step 3: Get a token for the team admin**

- **JWT (SSO)**: Log in via UI, select the team, copy the JWT from the browser.
- **API key**: Generate a key for the team admin user with `team_id` set.

```bash
# As proxy admin, generate a key for the team admin user
curl -X POST 'http://localhost:4000/key/generate' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "<TEAM_ADMIN_USER_ID>",
    "team_id": "team-validation-test",
    "models": ["all-team-models"]
  }'
```

The response contains the key; use it as `TEAM_ADMIN_TOKEN`.

### 5. Run the Validation Script

```bash
# Install httpx if not present
poetry add httpx

# Run validation
export PROXY_URL="http://localhost:4000"
export TEAM_ADMIN_TOKEN="<jwt-or-api-key-for-team-admin>"
export TEAM_ID="team-validation-test"

# Optional: for add/remove member test
export TEST_USER_ID="<user-id-to-add-then-remove>"

poetry run python scripts/validate_team_admin_permissions.py
```

Or with arguments:

```bash
poetry run python scripts/validate_team_admin_permissions.py \
  --proxy-url http://localhost:4000 \
  --team-admin-token <token> \
  --team-id team-validation-test \
  --test-user-id <optional-user-for-member-test>
```

**Skip destructive tests:**
- `--skip-model-crud` – Skip add/edit/delete model (avoids creating a test model)
- `--skip-member-test` – Skip add/remove member (or provide `--test-user-id`)

## Validation Checklist

| # | Capability | API / UI | Notes |
|---|------------|----------|-------|
| 1 | View all team keys | `GET /key/list?include_team_keys=true&team_id=<team_id>` | Team admin sees all keys for their team |
| 2 | Add/remove member | `POST /team/member_add`, `POST /team/member_delete` | Requires `TEST_USER_ID` |
| 3 | Add/edit/delete model | `POST /model/new`, `POST /model/update`, `POST /model/delete` | Use `--skip-model-crud` to avoid creating test model |
| 4 | Create team key with all team models | `POST /key/generate` with `models: ["all-team-models"]` | Script cleans up the created key |
| 5 | See all team models in test key dropdown | `GET /models?team_id=<team_id>` | Used by UI for model selector |

## Troubleshooting

- **"Database not connected"**: Ensure `DATABASE_URL` is set and the proxy has finished migrations.
- **403 on key list**: Ensure `include_team_keys=true` is passed and the user is a team admin (role=admin in `members_with_roles`).
- **403 on model/new**: Team admin must specify `model_info.team_id` matching their admin team.
- **Enterprise features**: Assigning team admins is an enterprise feature; ensure `LITELLM_LICENSE` is set if required.
