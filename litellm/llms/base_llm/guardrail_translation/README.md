# Request message scoping

Set `skip_assistant_message_in_guardrail` to exclude assistant messages in request history from guardrail evaluation. This excludes their text, images and tool calls from the supported request handlers, and excludes the messages from structured guardrail inputs. The model still receives the original assistant messages, and post-call guardrails still evaluate newly generated replies

Enable it for the proxy in `config.yaml`:

```yaml
litellm_settings:
  skip_assistant_message_in_guardrail: true
```

Or set it for one guardrail:

```yaml
guardrails:
  - guardrail_name: content-check
    litellm_params:
      guardrail: generic_guardrail_api
      mode: pre_call
      api_base: https://your-guardrail-api.example/check
      skip_assistant_message_in_guardrail: true
```

A per-guardrail `true` or `false` takes precedence over the global value. Omitting the per-guardrail value inherits the global setting, which defaults to `false`

The setting follows the existing request-role filters for unified Chat Completions, Anthropic Messages, Bedrock Converse passthrough and Lakera v2. The unified Responses API handler does not currently implement these request-role filters
