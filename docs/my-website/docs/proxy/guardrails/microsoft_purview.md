import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Microsoft Purview Guardrail

LiteLLM supports [Microsoft Purview](https://learn.microsoft.com/en-us/purview/purview) DLP policies via the [Microsoft Graph `processContent` API](https://learn.microsoft.com/en-us/graph/api/dataSecurityAndGovernance-processContent).

## Supported modes

| Mode | What it does |
|------|-------------|
| `pre_call` | Evaluates the user prompt against DLP policies before the LLM call. Blocks if a `restrictAccess/block` policy action fires. |
| `post_call` | Evaluates the LLM response against DLP policies. Blocks if a `restrictAccess/block` policy action fires. |
| `logging_only` | Sends both prompt and response to Purview for audit. Never blocks the request. |

## Prerequisites

1. **An Entra app registration** with the following Microsoft Graph application permissions:
   - `InformationProtectionPolicy.Read.All`
   - `ProtectionScopes.Compute.User`
   - `Content.Process.User`

2. **A DLP policy in Microsoft Purview** targeting your app registration's `client_id` as a Protected App. Without an active policy, `policyActions` in the API response will always be empty.

3. **Entra user object IDs** — each request must carry the Entra object ID of the end-user (not a username or email). The guardrail skips the DLP check and logs a warning when no user ID can be resolved.

## Quick Start

### 1. Register your app in Entra

```bash
# Create app registration and note the appId (client_id) and tenantId
az ad app create --display-name "LiteLLM-Purview"
az ad sp create --id <appId>

# Create a client secret
az ad app credential reset --id <appId> --append
```

Grant the permissions listed above in the Azure portal under **App registrations → API permissions**, then **Grant admin consent**.

### 2. Define the guardrail in `config.yaml`

<Tabs>
<TabItem value="pre_call" label="pre_call">

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: purview-prompt-dlp
    litellm_params:
      guardrail: microsoft_purview
      mode: pre_call
      api_key: os.environ/AZURE_CLIENT_SECRET   # client_secret
      tenant_id: os.environ/AZURE_TENANT_ID
      client_id: os.environ/AZURE_CLIENT_ID
      default_on: true
```

</TabItem>
<TabItem value="post_call" label="post_call">

```yaml
guardrails:
  - guardrail_name: purview-response-dlp
    litellm_params:
      guardrail: microsoft_purview
      mode: post_call
      api_key: os.environ/AZURE_CLIENT_SECRET
      tenant_id: os.environ/AZURE_TENANT_ID
      client_id: os.environ/AZURE_CLIENT_ID
      default_on: true
```

</TabItem>
<TabItem value="logging_only" label="logging_only (audit)">

```yaml
guardrails:
  - guardrail_name: purview-audit
    litellm_params:
      guardrail: microsoft_purview
      mode: logging_only
      api_key: os.environ/AZURE_CLIENT_SECRET
      tenant_id: os.environ/AZURE_TENANT_ID
      client_id: os.environ/AZURE_CLIENT_ID
      default_on: true
```

</TabItem>
</Tabs>

### 3. Start LiteLLM Gateway

```shell
litellm --config config.yaml --detailed_debug
```

### 4. Test request

Pass the Entra object ID of the end-user in `metadata`:

```shell
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, what is the capital of France?"}],
    "metadata": {"user_id": "<entra-user-object-id>"}
  }'
```

**When a DLP policy blocks the request**, LiteLLM returns:

```json
{
  "error": {
    "status_code": 400,
    "message": {
      "error": "Microsoft Purview DLP: Content blocked by policy",
      "activity": "uploadText"
    }
  }
}
```

## Supported Params

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `guardrail` | `str` | Yes | Must be `"microsoft_purview"` |
| `mode` | `str` | Yes | `pre_call`, `post_call`, or `logging_only` |
| `api_key` | `str` | Yes | Entra app client secret (can use `os.environ/VAR`) |
| `tenant_id` | `str` | Yes | Entra tenant ID |
| `client_id` | `str` | Yes | Entra app registration client ID (also used as the Protected App identifier in Purview) |
| `default_on` | `bool` | No | Run this guardrail for every request. Default: `false` |
| `purview_app_name` | `str` | No | App name reported to Purview in `processContent`. Default: `"LiteLLM"` |
| `user_id_field` | `str` | No | Metadata field to read the end-user's Entra object ID from. Default: `"user_id"` |

## User ID resolution

The guardrail resolves the Entra user object ID in this order:

1. `metadata[user_id_field]` — the field named by `user_id_field` (default `user_id`) in the request's `metadata` dict
2. `user_api_key_dict.user_id` — the user associated with the LiteLLM API key
3. `user_api_key_dict.end_user_id` — the end-user ID set on the LiteLLM key

If none of these are present, the DLP check is **skipped** and a warning is logged.

## Enabling per request

When `default_on: false`, you can opt individual requests in or out:

```shell
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}],
    "guardrails": ["purview-prompt-dlp"],
    "metadata": {"user_id": "<entra-user-object-id>"}
  }'
```

## How it works

1. **Token acquisition** — The guardrail uses the OAuth2 client credentials grant to get a Microsoft Graph bearer token. Tokens are cached until 60 seconds before expiry.

2. **Protection scope computation** — Before each DLP check, the guardrail calls `protectionScopes/compute` for the user to retrieve the ETag representing current policy state. Results are cached per user for 1 hour (per Microsoft's recommendation). If the `processContent` response indicates policies have changed (`protectionScopeState: modified`), the cache is invalidated.

3. **Content evaluation** — The guardrail calls `processContent` with the text and an `activityMetadata.activity` of `uploadText` (prompts) or `downloadText` (responses).

4. **Block decision** — If any `policyActions` entry has `@odata.type` containing `restrictAccessAction` and `restrictionAction: "block"`, the guardrail raises an HTTP 400.

5. **Audit logging** — In all modes, guardrail results are recorded in `metadata.standard_logging_guardrail_information` and flow to configured observability backends (Langfuse, Datadog, OTEL, etc.).

## Further Reading

- [Microsoft Graph processContent API](https://learn.microsoft.com/en-us/graph/api/dataSecurityAndGovernance-processContent)
- [Microsoft Purview DLP overview](https://learn.microsoft.com/en-us/purview/dlp-learn-about-dlp)
- [Control Guardrails per API Key](./quick_start#-control-guardrails-per-api-key)
