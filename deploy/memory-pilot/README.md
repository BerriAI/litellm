# Memory gateway pilot on Render

Run this branch as an isolated forwarding gateway. Colleagues keep their existing
upstream LiteLLM key and model name and change only their gateway base URL. They
need no plugin or client-side memory tools. Memory is off by default. Each person enables it explicitly for their key;
returning to the original URL stops using and collecting pilot memory.

Every model call uses that caller's upstream key. The upstream
gateway continues to enforce its model permissions, budgets, rate limits, and
guardrails. The pilot checks the key against the upstream model catalog, then
registers its hash as a local virtual key so LiteLLM's normal authentication and
memory authorization still apply. No upstream key or provider credential is
configured on Render. The administrator credential belongs only to this pilot.

The forwarding pilot isolates memories by virtual key. Upstream management APIs
may deny ordinary keys access to user/team/org details, so the pilot does not
infer those identities from client metadata. Install the feature directly in an
organization's gateway to use its existing user/team/project/org policies.
A regular gateway deployment reuses its existing PostgreSQL database with normal
schema migrations. It does not need a separate memory database or vector service.
This forwarding pilot has a separate database for isolation. Its memories are not
automatically available on the original gateway. Sharing requires both deployments
to run this feature against the same database and authenticated namespace; the
pilot must not be connected to an older gateway's production database.

## Create the service

1. Create a separate Render Postgres 16 database in the same region as the web
   service. Restrict public database access; use its internal connection URL.
2. Create a Python web service from this repository and the Memory V2 branch.
   A Standard service and the smallest paid Postgres plan are sufficient starting
   points for a small pilot. They incur Render hosting charges.
3. Set the build command to `bash deploy/memory-pilot/build.sh`, the start command
   to `bash deploy/memory-pilot/start.sh`, and the health path to
   `/health/readiness`. The build includes the dashboard from this branch.
   Set the service's maximum shutdown delay to 300 seconds so active requests
   can drain during a deployment. Uvicorn allows 290 seconds before cleanup
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
   key, expand **Advanced settings**, and save a policy for **Whole gateway**, **Users choose whether to opt in**,
   **Private to each virtual key**. This policy persists across restarts. Memory
   stays disabled until an administrator enables it.

Equivalent activation through the API, with secrets supplied in shell variables:

```bash
curl --fail-with-body "$PILOT_URL/v2/memory/policies" \
  -H "Authorization: Bearer $PILOT_ADMIN_KEY" \
  -H 'Content-Type: application/json' \
  -X PUT \
  -d '{"target_type":"gateway","target_id":"*","activation":"opt_in","scope":"key"}'
```

Administrators can instead enable memory automatically, disable a particular registered key,
or disable the whole gateway. Under an opt-in policy, callers set their preference
with `PUT /v2/memory/preference` and `{"enabled":true}` using their own key.
In a normal gateway, signed-in users can also select their key in Memory and turn
on the switch. Pilot administrators can do this for a registered key. The switch
shows the actual state; turning it off stops saving and recall but keeps saved
memories visible. Memories appear newest first, with optional details.

## Try it

Set an OpenAI-compatible client's base URL to `https://YOUR-SERVICE.onrender.com/v1`.
For Claude Code, set `ANTHROPIC_BASE_URL` to `https://YOUR-SERVICE.onrender.com`.
Retain the same gateway key and model setting. First opt in using the preference
API above, or ask the pilot administrator to turn on memory for your key.

In one conversation, say “Remember that my demo project is Cobalt Heron and its
staging port is 8347.” In a **new conversation**, ask “What is my demo project and
its staging port?” Check actual saved entries with `GET /v2/memory/entries` using
the same key. An unrelated key must not see them. Administrators can inspect,
correct, or delete entries in Memory; callers can use the self-service API.

## Behavior and limits

- Streaming keeps LiteLLM's configured SSE keepalives across silent memory
  rounds. The pilot sends comments every 15 seconds of silence and disables
  proxy buffering. A failure after streaming starts arrives as a native SSE
  error; before streaming starts, HTTP errors retain their retry delay
- Model calls retain LiteLLM's normal timeout and retry settings. The separate
  upstream credential check has a 20-second timeout. Deployments drain existing
  requests for up to five minutes; requests still running after that can be
  interrupted. Schedule pilot updates outside active office usage
- Supported surfaces: Chat Completions, Responses, and Anthropic Messages,
  including their native streaming responses and client tool continuation.
- The selected model must support function calling. The actual answering model
  receives catalog, fuzzy search, full-read, and observation-capture tools beside
  its normal client tools. The gateway executes only its own memory tools.
- A request allows at most eight model rounds and sixteen memory calls per round.
  One final reflection round can acknowledge an empty observation batch. Additional
  rounds use the same model and caller budget, and add latency and token spend.
- Captures are immediately visible after a confirmed save. Each observation keeps
  its title, relevance guidance, scope, kind, certainty, evidence, source, and actor.
  Corrections append observations. Agents receive no memory deletion tool.
- Search uses weighted fuzzy matching over the authorized scope. There is no vector
  database, extraction model, or nightly consolidation.
- Searches accept up to 16 distinct terms. Fuzzy matching checks up to 256 distinct
  words per field; exact terms still match anywhere in the field.
- Fixed instructions and tool definitions preserve prompt-prefix caching after
  warm-up. Dynamic catalogs and checkpoint IDs stay at the conversation tail.
  Complete-response caching is bypassed for memory rounds on both gateways so
  permission checks, retrieval, and capture execute against current state.
- Hidden tool continuations expire after 24 hours, hold at most one megabyte each,
  and are limited to 1,000 per key and scope. They contain gateway-added fragments,
  not another copy of the complete incoming transcript. Responses retrieval and
  continuation use gateway-owned response IDs; deleting one removes its model
  responses and temporary continuation records, not saved memories.
- `/input_items` returns 501 for gateway-owned response IDs. Retain the original
  client input; the hidden provider transcript is not a public input history.
- Foreground requests with one completion are supported. Use modern tools instead
  of legacy functions. The special Cursor conversion route, background responses,
  multiple completions, and WebSocket inference are outside this implementation.
- On gateway/backend deployments without shared Redis, first-time activation
  can take up to 30 seconds to reach another process. Policy revocation is
  checked against the primary database before memory operations.
- Each memory scope can hold up to 1,000 entries. Creation checks this limit
  atomically; correction and deletion remain available when the scope is full.
- Stored references are untrusted data. They cannot grant API permissions or
  change the namespace derived from authentication. Current user corrections
  take precedence. Replacements require the current revision.
- Invalid tool arguments return errors to the model. Infrastructure and model
  failures fail the request or stream instead of reporting a successful save. Administrators can disable memory to restore ordinary calls.
- Switching away or disabling memory stops automatic use; it does not delete
  existing entries. Delete memories explicitly through Memory or the API.
- Shared upstream keys share a pilot namespace. Give each person a distinct key
  when their memories must be private from each other.
- Other API surfaces are outside this forwarding pilot. Use the original gateway
  for embeddings, images, realtime, batches, and administration.
