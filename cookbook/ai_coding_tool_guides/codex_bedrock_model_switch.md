# Switch Bedrock OpenAI models without replaying incompatible reasoning

Bedrock can reject encrypted reasoning when a Responses API conversation changes models, even within GPT-5.6. A Sol response replayed to Luna can return HTTP 400 with `encrypted reasoning was created for a different account or model`, while replaying the same input to Sol succeeds

Enable encrypted-content affinity on the proxy before starting the conversation:

```yaml
model_list:
  - model_name: sol
    litellm_params:
      model: bedrock_mantle/openai.gpt-5.6-sol
      aws_region_name: us-east-2
      api_key: os.environ/AWS_BEARER_TOKEN_BEDROCK
  - model_name: luna
    litellm_params:
      model: bedrock_mantle/openai.gpt-5.6-luna
      aws_region_name: us-east-2
      api_key: os.environ/AWS_BEARER_TOKEN_BEDROCK

router_settings:
  optional_pre_call_checks:
    - encrypted_content_affinity
```

For SigV4 authentication, omit `api_key` and configure the usual AWS credentials. The callback also recognizes Bedrock OpenAI model IDs routed through an OpenAI-compatible upstream, such as `openai/us.openai.gpt-5.6-sol`

Continue replaying the response output items in their original order. LiteLLM tags encrypted items with their originating deployment and removes the tracking wrapper before forwarding to the provider. Codex can omit item IDs because the encrypted-content wrapper also carries the origin

When the selected route has one underlying Bedrock OpenAI model, LiteLLM removes encrypted reasoning from other known Bedrock OpenAI models outside that route. It preserves readable reasoning summaries, messages, assistant phases, function calls, and function outputs. Reasoning from the selected model remains available, including in histories containing several models or when switching back to an earlier model. `reasoning.context`, effort, and summary settings are not overridden

Calls that stay on the same model retain encrypted reasoning. Aliases of the same underlying model retain it when their encryption boundary matches. If the originating deployment remains in the requested model group but is unavailable, the existing affinity/cooldown behavior applies; an unavailable deployment is not treated as an explicit model switch

The tradeoff occurs only for incompatible reasoning: the destination model must reconstruct any analysis that existed only in that reasoning. The HTTP client's original history remains unchanged, so switching back can reuse the earlier model's compatible items. This does not make encrypted reasoning portable between models

## Existing conversations and other opaque state

The proxy cannot infer the origin of an untagged encrypted item. Conversations started before affinity was enabled need a fresh conversation with a visible handoff, or client-side removal of the incompatible reasoning. Retain stable deployment IDs and do not repurpose an existing ID for another upstream model while clients still reference it

This mitigation targets reasoning items. Compaction items and other opaque state are preserved and may have their own compatibility restrictions. Routes containing several underlying models retain the existing deployment-affinity behavior

## Why not set `current_turn` globally?

Direct Bedrock Runtime tests on 2026-09-18 returned the following results with fixed credentials in `us-east-2` and `store: false`:

| Replay | Default `all_turns` | `current_turn` | Reasoning removed |
| --- | --- | --- | --- |
| Sol to Luna, next user turn | 400 | 200 | 200 |
| Luna to Sol, next user turn | 400 | 200 | 200 |
| Luna to Terra, next user turn | 400 | 200 | 200 |
| Terra to Luna, next user turn | 400 | 200 | 200 |
| Sol to Luna, active tool loop | 400 | 400 | 200 |
| Luna to Sol, active tool loop | 400 | 400 | 200 |

Same-model controls returned 200. A reasoning item from a Codex conversation reproduced the same Luna-to-Terra failure. These observations conflict with the [documented Sol/Terra/Luna reasoning reuse](https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-across-calls); the provider's internal cause remains unconfirmed

`current_turn` excludes reasoning from completed user turns, even when the model stays the same. It also leaves incompatible reasoning in the payload and still fails for a switch during the active tool loop. Selective removal preserves compatible reasoning without imposing that global loss of continuity

The [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference) and schema checked on 2026-09-18 expose effort and summary controls, but no documented `reasoning.context` override. This mitigation is configured in LiteLLM

The direct Mantle reproduction was reported separately. The Runtime tests above were independently executed; Mantle retesting was blocked by missing `bedrock-mantle:CreateInference` permission
