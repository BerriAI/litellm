# Jev and Laya decision model classifiers

Select **Decision Model** in the auto-router editor, then choose **Jev (TypeSafe)** or **Laya (self-hosted)**. Both use System One choice questions to select a configured complexity tier

The gateway administrator configures the connections. Existing Jev `api_base` and `api_key` settings retain their meaning. Laya uses separate `laya_api_base` and `laya_api_key` settings, so switching providers and saving keeps both connections available

Add this fragment to an auto-router's `litellm_params`, replacing the tier destinations with your existing gateway model names:

```yaml
model: auto_router/complexity_router
complexity_router_config:
  classifier_type: jev
  jev_classifier_config:
    provider: laya
    model: multilingual
    laya_api_base: http://localhost:8000
    timeout_ms: 10000
  tiers:
    SIMPLE: efficient-model
    MEDIUM: balanced-model
    COMPLEX: capable-model
    REASONING: reasoning-model
```

Run the upstream [`laya-serve`](https://github.com/NandhaKishorM/laya) server at the configured address. The gateway appends `/v1/systemone` to `laya_api_base`. For authenticated servers, set `laya_api_key` to the server's bearer token, using the gateway's standard `os.environ/<VARIABLE>` configuration reference if desired. An omitted Laya key sends no authorization header and never inherits the Jev key

Laya supports `multilingual`, `english`, and `typed-decisions`. The default is `multilingual`. Preload the selected checkpoint in the Laya server to avoid a cold model download on the first classification request

To use Jev, select Jev in the editor or set `provider: typesafe` and `model: jev-latest`. Jev uses `api_base` and `api_key`, with the existing `TYPESAFE_API_BASE` and `TYPESAFE_API_KEY` defaults. An explicit Jev endpoint requires its own key

The three self-hosted Laya checkpoints have zero provider token charges in the shipped model registry. This does not account for your server's compute costs. Operators can override the registry pricing for their deployment; routing decisions and usage logs use the same selected checkpoint and rate
