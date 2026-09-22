# LiteAsk

LiteAsk adds a chat button to the gateway dashboard for proxy administrators. It runs in the gateway deployment and uses the current dashboard session, including the deployment's configured SSO. It does not require Render, a separate agent service, a shared admin key, or another sign-in

Set `LITELLM_LITEASK_MODEL` to a model alias already configured on the gateway, then restart the gateway or management backend. Use a model that supports function calling. Each administrator must be allowed to call that model; the gateway applies its normal inference authorization, limits, guardrails and spend tracking to every model call

```yaml
environment_variables:
  LITELLM_LITEASK_MODEL: admin-assistant
model_list:
  - model_name: admin-assistant
    litellm_params:
      model: openai/gpt-5.5
      api_key: os.environ/OPENAI_API_KEY
```

For a split deployment, configure the model and provider credential on the management backend. LiteAsk reuses the gateway's chat handler internally while `/chat/completions` remains absent from the public management route table

Without the setting, the widget stays hidden. Only a current database user with the `proxy_admin` role can use it. Read-only administrators and team administrators cannot. The server checks the live user record on every entry and before each internal operation, so role removal, deletion or SCIM deactivation stops access

## Operations

Ask LiteAsk to inspect keys, teams, users and budgets, query spend, or inspect request logs. It supports creating, updating, blocking and deleting keys; creating, updating and deleting teams, users and budgets; and adding, updating or removing team members. Key updates can assign a key to a team. Request details are available when the gateway stores them

The operation catalog is an explicit allowlist built from the gateway's OpenAPI schema. It does not expose every dashboard feature. Models, providers, SSO settings and arbitrary HTTP requests are outside this initial catalog

Reads run immediately. A change produces a review card showing the exact operation and arguments. Confirming executes those sealed arguments under your current credential. Canceling discards the proposal. Approvals expire after five minutes and are bound to your identity, credential and conversation

Changes require the gateway's existing shared Redis connection. Reads remain available without it. Proposals must be recorded in Redis before they are shown, then atomically consumed before a change runs. Missing records and Redis failures refuse the change. This prevents replay across workers and gateway restarts while Redis retains its current state. Restoring an older Redis snapshot can restore old approvals, so the approval namespace must not be rolled back. An interrupted request may have completed; check the gateway before preparing another change. LiteAsk never automatically retries a write

Generated keys appear in a separate copy panel and are excluded from model input and subsequent conversation history. Other tool results are redacted and size limited. Conversations are held in browser memory and cleared when the administrator signs out, switches accounts, or starts a new chat. Closing the drawer retains the current conversation until then

## API

`GET /management/v1/liteask/config` reports whether LiteAsk is configured and whether changes are available. `POST /management/v1/liteask/chat` takes a conversation UUID and user/assistant messages. `POST /management/v1/liteask/approve` takes that same UUID and the proposal token, with no replacement arguments. All three endpoints require normal gateway authentication and a current proxy admin identity
