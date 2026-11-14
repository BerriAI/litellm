The OpenAI passthrough route now resolves customer organization, team and effective member model restrictions before forwarding the request. It uses the existing customer resolver and model matcher, then checks the model again after hooks. Gateway key permissions still apply.

Actual curl proof used a controlled HTTP upstream. Before: PR head `43d55ab2b91e4cccccf014c5f813a2b70a4e89f9` merged with target `eb811913f1`, before the fix. After: the implementation linked from the current PR description. Counts below come from the upstream request log.

Setup requests, using synthetic identifiers:

```http
POST /management/v1/customer_organizations
{"customer_org_id":"proof-org","alias":"Synthetic passthrough permission proof","budget":{"max_budget":100,"allowed_models":["gpt-4.1-mini"]}}

POST /user/new
{"user_id":"proof-user","user_role":"internal_user","auto_create_key":false}

POST /key/generate
{"user_id":"proof-user","models":["gpt-4.1-mini","gpt-5.5"],"metadata":{"customer_org_id":"proof-org"}}

POST /key/generate
{"user_id":"proof-user","models":["gpt-4.1-mini"],"metadata":{"customer_org_id":"proof-org"}}
```

The first key permits both models; the second permits only mini. Credentials from both responses are omitted.

```bash
curl "$GATEWAY_BASE_URL/openai/v1/chat/completions" \
  --header "@$CUSTOMER_KEY_AB_HEADERS" \
  --header 'Content-Type: application/json' \
  --data '{"model":"gpt-5.5","messages":[{"role":"user","content":"Synthetic passthrough policy check"}]}'
```

Before: HTTP 200, one upstream call. Captured response excerpt:

```json
{
  "model": "gpt-5.5",
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120
  }
}
```

After: HTTP 403, zero upstream calls. Captured response:

```json
{
  "error": {
    "message": "The requested model 'gpt-5.5' is not in the allowed models for this customer. Check the models this customer can use and try again.",
    "type": "customer_model_access_denied",
    "param": "model",
    "code": "403"
  }
}
```

The same key still receives HTTP 200 for `gpt-4.1-mini`, with one upstream call. After widening the organization to permit both models, the mini-only key still receives HTTP 403 for `gpt-5.5`, with zero upstream calls:

```json
{
  "error": {
    "message": "The requested model 'gpt-5.5' is not available for this API key, or the model name is invalid. Check the models available to you and try again.",
    "type": "key_model_access_denied",
    "param": "model",
    "code": "403"
  }
}
```

The proof deletes its synthetic organization afterward. Registered-route regressions also cover team restrictions, member templates and overrides, hook model rewrites, model-less file retrieval, and the existing configured passthrough model contract.
