# Memory gateway pilot on Render

Run this branch as an isolated forwarding gateway. Colleagues keep their existing
upstream LiteLLM key and model name and change only their gateway base URL. They
need no plugin or client-side memory tools. Choosing the pilot URL opts them into
the pilot; returning to the original URL stops using and collecting pilot memory.

Every preparation and answer call uses that caller's upstream key. The upstream
gateway continues to enforce its model permissions, budgets, rate limits, and
guardrails. The pilot checks the key against the upstream model catalog, then
registers its hash as a local virtual key so LiteLLM's normal authentication and
memory authorization still apply. No upstream key or provider credential is
configured on Render. The administrator credential belongs only to this pilot.

The forwarding pilot isolates memories by virtual key. Upstream management APIs
may deny ordinary keys access to user/team/org details, so the pilot does not
infer those identities from client metadata. Install the feature directly in an
organization's gateway to use its existing user/team/project/org policies.
Never connect this pilot to an older gateway's production database.

## Create the service

1. Create a separate Render Postgres 16 database in the same region as the web
   service. Restrict public database access; use its internal connection URL.
2. Create a Python web service from this repository and the Memory V2 branch.
   A Standard service and the smallest paid Postgres plan are sufficient starting
   points for a small pilot. They incur Render hosting charges.
3. Set the build command to `bash deploy/memory-pilot/build.sh`, the start command
   to `bash deploy/memory-pilot/start.sh`, and the health path to
   `/health/readiness`. The build includes the dashboard from this branch.
4. Set these environment variables in Render:

   | Variable | Value |
   | --- | --- |
   | `DATABASE_URL` | The new database's internal connection URL |
   | `UPSTREAM_LITELLM_BASE_URL` | Your original gateway URL, without `/v1` |
   | `LITELLM_MASTER_KEY` | A new random `sk-` administrator key |
   | `LITELLM_SALT_KEY` | A separate random encryption secret; preserve it across deploys |
   | `PYTHON_VERSION` | `3.12.14` |
   | `NODE_VERSION` | `24.19.0` |
   | `NEXT_TELEMETRY_DISABLED` | `1` |
   | `PORT` | `4000` |

5. After deployment, open `/ui/memory`, sign in as `admin` using the pilot's master
   key, and save a policy for **Whole gateway**, **Enabled automatically**,
   **Private to each virtual key**. This policy persists across restarts. Memory
   stays disabled until an administrator enables it.

Equivalent activation through the API, with secrets supplied in shell variables:

```bash
curl --fail-with-body "$PILOT_URL/v2/memory/policies" \
  -H "Authorization: Bearer $PILOT_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -X PUT \
  -d '{"target_type":"gateway","target_id":"*","activation":"automatic","scope":"key"}'
```

Administrators can instead require opt-in, disable a particular registered key,
or disable the whole gateway. Under an opt-in policy, callers set their preference
with `PUT /v2/memory/preference` and `{"enabled":true}` using their own key.

## Try it

Set an OpenAI-compatible client's base URL to `https://YOUR-SERVICE.onrender.com/v1`.
For Claude Code, set `ANTHROPIC_BASE_URL` to `https://YOUR-SERVICE.onrender.com`.
Retain the same gateway key and model setting.

In one conversation, say “Remember that my demo project is Cobalt Heron and its
staging port is 8347.” In a **new conversation**, ask “What is my demo project and
its staging port?” Check actual saved entries with `GET /v2/memory/entries` using
the same key. An unrelated key must not see them. Administrators can inspect,
correct, or delete entries in Memory; callers can use the self-service API.

## Behavior and limits

- Supported surfaces: Chat Completions, Responses, and Anthropic Messages,
  including their native streaming responses and client tool continuation.
- The selected model must support function calling. Memory preparation adds up
  to three billed model calls before the visible answer, with a 60-second bound.
  It uses the original conversation, so long coding sessions can add substantial
  prompt-token usage and latency. Existing upstream quotas apply to these calls.
- Preparation stores durable facts supported by the conversation, then searches
  and reads relevant entries. Search is bounded keyword matching in Postgres.
  There is no vector database, extraction model, scheduler, or nightly process.
- On gateway/backend deployments without shared Redis, first-time activation
  can take up to 30 seconds to reach another process. Policy revocation is
  checked against the primary database before memory operations.
- Each memory scope can hold up to 1,000 entries. Creation checks this limit
  atomically; correction and deletion remain available when the scope is full.
- Stored references are untrusted data. They cannot grant API permissions or
  change the namespace derived from authentication. Current user corrections
  take precedence. Replacements require the current revision.
- Memory/model preparation errors fail the request rather than silently claiming
  successful memory. Administrators can disable memory to restore ordinary calls.
- Switching away or disabling memory stops automatic use; it does not delete
  existing entries. Delete memories explicitly through Memory or the API.
- Shared upstream keys share a pilot namespace. Give each person a distinct key
  when their memories must be private from each other.
- Other API surfaces are outside this forwarding pilot. Use the original gateway
  for embeddings, images, realtime, batches, and administration.
