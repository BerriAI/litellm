# ADEPT Router — End-to-End Testing Guide

This guide walks through every layer of the ADEPT system in a logical sequence.
Each stage builds on the previous one. Run them in order the first time.

---

## Prerequisites

- LiteLLM proxy running (default `http://localhost:4000`)
- PostgreSQL running and reachable
- A master key set (e.g. `LITELLM_MASTER_KEY=sk-1234`)
- `curl` and `jq` available in your terminal

Set a convenience alias for all curl commands:

```bash
export LITELLM_URL=http://localhost:4000
export LITELLM_KEY=sk-1234
```

---

## Stage 1 — Basic Health and Inference

Confirm the proxy is alive and can call a model before touching ADEPT at all.

### 1.1 Health check

```bash
curl -s $LITELLM_URL/health | jq .
```

Expected: `{"status": "healthy", ...}`

### 1.2 List available models

```bash
curl -s $LITELLM_URL/v1/models \
  -H "Authorization: Bearer $LITELLM_KEY" | jq '.data[].id'
```

You should see the model names from your `config.yaml`.

### 1.3 Basic inference (non-ADEPT model)

```bash
curl -s $LITELLM_URL/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello."}]
  }' | jq '.choices[0].message.content'
```

Expected: any sensible reply. This confirms the proxy and provider credentials work.

---

## Stage 2 — ADEPT Router Config and Visibility

### 2.0 Install the ADEPT extra

ADEPT's Postgres-backed template store requires `sqlalchemy` and `psycopg2-binary`,
which are shipped as an opt-in extra. Install them alongside the proxy:

```bash
pip install "litellm[proxy,adept]"
```

Without this extra, the proxy will raise `ImportError` on startup as soon as an
`adept_router: true` deployment is loaded.

### 2.1 Sample config.yaml for ADEPT

Add a deployment with the `adept_router_*` params:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: invoice-slm          # trained SLM (leave as default model initially)
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
      adept_router: true
      adept_router_default_model: gpt-4o-mini
      adept_router_pg_host: localhost
      adept_router_pg_port: 5432
      adept_router_pg_user: litellm
      adept_router_pg_password: yourpassword
      adept_router_pg_database: adept
      adept_router_tag_prefix: ""           # or "tool:" for namespaced tags
      adept_router_conversations_threshold: 5
      adept_router_trainer_url: ""          # set to trainer URL when ready
```

Restart LiteLLM after saving the config.

### 2.2 Confirm ADEPT router is registered

On startup you should see in the LiteLLM logs:

```
AdeptRouter: initialized PostgreSQL template store.
```

If you see a Postgres connection error instead, check your `adept_router_pg_*` credentials.

### 2.3 Confirm callback is wired

```bash
# In a Python shell or script:
python3 -c "
import litellm
from litellm import Router
# load your config...
# Check:
adept_instances = [cb for cb in litellm.callbacks if type(cb).__name__ == 'AdeptRouter']
print('AdeptRouter callbacks registered:', len(adept_instances))
"
```

Expected: `AdeptRouter callbacks registered: 1` (one per ADEPT deployment).

---

## Stage 3 — First ADEPT Inference (Cold — No Template Yet)

### 3.1 Send a tagged prompt (simulating a tool call)

This uses your ADEPT-enabled model name. The first request will:
- **Miss** the template store (no template exists yet)
- Fall back to the default model
- Store a new template + conversation in Postgres

```bash
curl -s $LITELLM_URL/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "invoice-slm",
    "messages": [
      {
        "role": "system",
        "content": "You are an invoice processor. Extract the total amount from the invoice."
      },
      {
        "role": "user",
        "content": "<invoice_text>Invoice #INV-001\nDate: 2024-01-15\nTotal: $1,234.56\nVendor: Acme Corp</invoice_text>"
      }
    ]
  }' | jq '{model: .model, content: .choices[0].message.content}'
```

Expected: a response from the default model (gpt-4o-mini). Check the logs for:

```
AdeptRouter: no template match, falling back to gpt-4o-mini
AdeptRouter: stored interaction.
```

### 3.2 Verify the template was stored in Postgres

```bash
psql -U litellm -d adept -c "SELECT id, template_hash, target_model FROM adept_templates LIMIT 5;"
```

You should see one row. The `target_model` will be empty (`""`) — it gets populated after training.

```bash
psql -U litellm -d adept -c "SELECT id, template_id FROM adept_conversations LIMIT 5;"
```

You should see one conversation row linked to the template.

---

## Stage 4 — Template Matching (Same Tool, Different Data)

### 4.1 Send the same tool's prompt with different runtime values

The system prompt is the same. The user message has the same XML structure but different content.
ADEPT should strip the tag values and match the skeleton to the stored template.

```bash
curl -s $LITELLM_URL/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "invoice-slm",
    "messages": [
      {
        "role": "system",
        "content": "You are an invoice processor. Extract the total amount from the invoice."
      },
      {
        "role": "user",
        "content": "<invoice_text>Invoice #INV-002\nDate: 2024-02-20\nTotal: $9,876.00\nVendor: Beta Ltd</invoice_text>"
      }
    ]
  }' | jq '{model: .model, content: .choices[0].message.content}'
```

Logs should show:
```
Matched template <template_id>
AdeptRouter: matched template <template_id>, routing to gpt-4o-mini
```

Even though the data is different, it matched the same template (same skeleton structure).

### 4.2 Verify the conversation count increased

```bash
psql -U litellm -d adept -c \
  "SELECT template_id, COUNT(*) as conversations FROM adept_conversations GROUP BY template_id;"
```

Expected: 2 conversations for the same template_id.

### 4.3 Different system prompt = different template

Send a different tool (different system prompt). This should create a **new** template:

```bash
curl -s $LITELLM_URL/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "invoice-slm",
    "messages": [
      {
        "role": "system",
        "content": "You are a contract reviewer. Summarize the key terms."
      },
      {
        "role": "user",
        "content": "<contract_text>This agreement is between Party A and Party B effective 2024-03-01.</contract_text>"
      }
    ]
  }' | jq .model'
```

```bash
psql -U litellm -d adept -c "SELECT COUNT(*) FROM adept_templates;"
```

Expected: 2 templates now (one per unique tool/system prompt combo).

---

## Stage 5 — Hit the Training Threshold

Send enough requests to trigger the trainer notification.
With `adept_router_conversations_threshold: 5`, the 5th, 10th, 15th... conversation triggers it.

### 5.1 Loop to threshold

```bash
for i in $(seq 1 5); do
  echo "=== Request $i ==="
  curl -s $LITELLM_URL/v1/chat/completions \
    -H "Authorization: Bearer $LITELLM_KEY" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"invoice-slm\",
      \"messages\": [
        {\"role\": \"system\", \"content\": \"You are an invoice processor. Extract the total amount from the invoice.\"},
        {\"role\": \"user\", \"content\": \"<invoice_text>Invoice #INV-00$i\nTotal: \$${i}00.00\nVendor: Vendor$i</invoice_text>\"}
      ]
    }" | jq '.choices[0].message.content' -r
  sleep 1
done
```

### 5.2 Verify conversation count

```bash
psql -U litellm -d adept -c \
  "SELECT template_id, COUNT(*) FROM adept_conversations GROUP BY template_id ORDER BY COUNT(*) DESC;"
```

### 5.3 Trainer trigger (if configured)

If `adept_router_trainer_url` is set, on the 5th request you'll see in LiteLLM logs:

```
Triggered trainer for template <template_id>
```

If no `trainer_url` is set, you'll see:

```
No trainer_url configured, skipping trainer notification
```

Both are correct — the skip is intentional when trainer is not deployed.

---

## Stage 6 — Post-Training Routing (After SLM is Trained)

Once the trainer has fine-tuned a model and loaded it into vLLM, update the template's
`target_model` in Postgres to point to the trained adapter:

```bash
psql -U litellm -d adept -c \
  "UPDATE adept_templates SET target_model = 'invoice-slm-v1' WHERE target_model = '';"
```

Then also add `invoice-slm-v1` to your LiteLLM `config.yaml` pointing to your vLLM endpoint:

```yaml
  - model_name: invoice-slm-v1
    litellm_params:
      model: openai/invoice-slm-lora
      api_base: http://localhost:8000/v1
      api_key: fake-key
```

### 6.1 Verify routing switches to SLM

```bash
curl -s $LITELLM_URL/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "invoice-slm",
    "messages": [
      {
        "role": "system",
        "content": "You are an invoice processor. Extract the total amount from the invoice."
      },
      {
        "role": "user",
        "content": "<invoice_text>Invoice #INV-010\nTotal: $500.00\nVendor: TestCo</invoice_text>"
      }
    ]
  }' | jq '{model: .model, content: .choices[0].message.content}'
```

Logs should now show:
```
AdeptRouter: matched template <template_id>, routing to invoice-slm-v1
```

### 6.2 Confirm `routed_to_slm` flag in Postgres

```bash
psql -U litellm -d adept -c \
  "SELECT additional_information->>'routed_to_slm', additional_information->>'model'
   FROM adept_conversations ORDER BY created_at DESC LIMIT 3;"
```

Expected: most recent row shows `routed_to_slm: true` and `model: invoice-slm-v1`.

### 6.3 (Optional) Test vLLM LoRA adapter directly

If you have vLLM running with a base model:

```bash
# Check vLLM health
curl http://localhost:8000/health

# List loaded models
curl http://localhost:8000/v1/models | jq '.data[].id'

# Load a LoRA adapter (requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 env var on vLLM startup)
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "invoice-slm-lora",
    "lora_path": "/path/to/lora/adapter"
  }'

# Verify the adapter appears in models list
curl http://localhost:8000/v1/models | jq '.data[].id'

# Inference through the adapter
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "invoice-slm-lora",
    "messages": [{"role": "user", "content": "Invoice #INV-001 Total: $500"}]
  }' | jq '.choices[0].message.content'
```

> **Important:** vLLM must be started with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` in its environment,
> otherwise the `/v1/load_lora_adapter` endpoint is not registered and returns 404.

---

## Stage 7 — Edge Case Tests

### 7.1 Non-tagged prompt (no XML structure)

ADEPT should still work — it just won't strip any tag values. The skeleton = the full (normalized) prompt.

```bash
curl -s $LITELLM_URL/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "invoice-slm",
    "messages": [
      {"role": "user", "content": "What is 2 + 2?"}
    ]
  }' | jq '.choices[0].message.content'
```

This creates its own template (hash of "What is {NUM} + {NUM}?"). Works fine.

### 7.2 Multi-turn conversation — only last user message used

Send a conversation with multiple turns:

```bash
curl -s $LITELLM_URL/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "invoice-slm",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"},
      {"role": "assistant", "content": "Paris."},
      {"role": "user", "content": "<query>Tell me more about Paris.</query>"}
    ]
  }' | jq '.choices[0].message.content'
```

ADEPT uses only the last `user` message for template matching.

### 7.3 Tool-result turn is skipped

If the last message is `role: tool`, ADEPT skips storing the conversation (the preceding assistant turn already captured the exchange):

```bash
curl -s $LITELLM_URL/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "invoice-slm",
    "messages": [
      {"role": "user", "content": "Run this tool"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "get_data", "arguments": "{}"}}]},
      {"role": "tool", "tool_call_id": "tc1", "content": "tool result here"}
    ]
  }' | jq '.choices[0].message.content'
```

Logs should show no "stored interaction" for this turn — that's correct behavior.

### 7.4 Special characters in Postgres password

The PG URL is URL-encoded, so passwords with `@`, `:`, `/` work. Verify by starting with:

```yaml
adept_router_pg_password: "p@ss:w/rd"
```

LiteLLM should connect without error. The encoded URL will have `p%40ss%3Aw%2Frd` in it.

---

## Quick Reference: What to Check at Each Step

| Stage | What to verify | Where to check |
|-------|---------------|----------------|
| 1 | Proxy alive, model available, basic inference works | `curl /health`, `/v1/models`, one completion |
| 2 | ADEPT registered, Postgres connected | Startup logs, `adept_templates` table exists |
| 3 | Cold miss: fallback model used, template created | Logs, `adept_templates` row |
| 4 | Same skeleton → template hit; different sys prompt → new template | Logs, `adept_templates` count |
| 5 | Conversation counter grows; trainer triggered at threshold | `adept_conversations` count, logs |
| 6 | After `target_model` updated: routes to SLM; `routed_to_slm=true` in DB | Logs, `adept_conversations.additional_information` |
| 7 | Untagged prompt, multi-turn, tool-result skip, special-char password all work | Logs, no errors |

---

## Common Problems

**Template never matches (always "no template match")**
- Check that the system prompt is identical across requests (whitespace-normalized, same content).
- Check that the XML tag names are consistent (e.g. `<invoice_text>` must match exactly).
- Query `adept_templates` — if empty, `async_log_success_event` is not firing. Check that `AdeptRouter` appears in `litellm.callbacks`.

**Conversations not stored**
- `AdeptRouter` is not registered in the callback pipeline. Look for `"AdeptRouter: stored interaction"` in logs.
- Verify startup logs show `"AdeptRouter: initialized PostgreSQL template store."`.

**Postgres connection error on startup**
- Check `adept_router_pg_*` values in config.yaml.
- Test connectivity: `psql -U <user> -h <host> -d <dbname>`.
- If password has special characters, they are URL-encoded automatically — you do not need to encode them yourself.

**vLLM `/v1/load_lora_adapter` returns 404**
- vLLM was not started with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`.
- Restart vLLM: `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 vllm serve <model> --enable-lora ...`

**Trainer never triggered**
- `adept_router_trainer_url` is empty or not set — that's fine in dev, logs will say "No trainer_url configured, skipping".
- If it IS set and still not triggered, check that `conversation_count % conversations_threshold == 0` is reached. Use `COUNT(*)` query above.
