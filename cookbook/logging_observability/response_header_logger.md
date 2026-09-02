# Store upstream response headers in spend logs

Copy `response_header_logger.py` next to your proxy configuration as `custom_callbacks.py`, then register the logger:

```yaml
model_list:
  - model_name: openrouter-header-test
    litellm_params:
      model: openrouter/openai/gpt-5.6-luna
      api_key: os.environ/OPENROUTER_API_KEY
litellm_settings:
  callbacks: [custom_callbacks.proxy_handler_instance]
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  proxy_batch_write_at: 1
```

Start the proxy with this repository's code and your configuration:

```sh
uv run --no-sync litellm --config config.yaml --port 4013
```

The logger captures response headers before stream iteration, enriches successful spend logs through `async_logging_hook`, and recovers failure headers from the exception chain. A failure without recoverable headers retains the earlier capture. When a fallback response is captured, its headers replace the failed attempt’s headers. Request state stays in request metadata, so concurrent calls do not share headers on the logger instance

Send a request with existing custom metadata:

```sh
curl --fail-with-body -sS -N http://localhost:4013/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"openrouter-header-test","messages":[{"role":"user","content":"Reply exactly OK"}],"stream":true,"max_tokens":32,"metadata":{"spend_logs_metadata":{"verification_case":"stream","existing_field":"keep-me"}}}'
```

After the background write, read the stored headers:

```sh
curl --fail-with-body -sS http://localhost:4013/spend/logs \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" |
  jq '.[] | . as $row |
      (.metadata | if type == "string" then fromjson else . end).spend_logs_metadata as $m |
      {request_id: $row.request_id, case: $m.verification_case,
       existing_field: $m.existing_field, headers: $m.upstream_response_headers}'
```

Headers appear in `metadata.spend_logs_metadata.upstream_response_headers`. Provider names receive the `llm_provider-` prefix; existing custom metadata and LiteLLM's internal-header protections are preserved

Set `stream` to false for the non-streaming control. To reproduce an upstream routing failure, add `"num_retries":0,"provider":{"only":["header-verification-nonexistent-provider"],"allow_fallbacks":false}` to the request. This returned HTTP 404 in verification

The logger was verified with OpenRouter chat completions, including simultaneous streams, a non-streaming control, and an upstream 404. The unit tests also cover a headerless error after header capture and both metadata keys used by the logging pipeline. Live commands, spend-log/database readbacks, and the exact tested revisions are in [PR #39376](https://github.com/BerriAI/litellm/pull/39376)

An earlier logger without early capture lost headers on a live streaming 429. That 429 did not recur after the logger update; retention for that error shape is covered by a unit test, not a second live 429 reproduction

Unmodified v1.98.0 still needs the shared HTTP-handler propagation fix as well as the router fix. The logger alone cannot recover headers dropped before the callback. These are headers returned by OpenRouter, which may differ from those of the provider behind it. Other providers, every retry/fallback attempt, and every failure type have not been live-tested
