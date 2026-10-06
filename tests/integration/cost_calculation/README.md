# Cost tracking integration tests

The cases in `chat_completions/` send one request through the proxy to a deployment registered for that test. Each `CostTrackingTestCase` stores the deployment parameters without `api_base`, the endpoint and request sent to LiteLLM, the scripted provider response, the expected status and optional response cost header, and the flat expected spend-log fields

`expected_spend_log` includes spend, prompt and completion tokens, and every field present in `metadata.cost_breakdown`. The runner compares the complete key set, compares numeric values with a relative tolerance of `1e-6`, and compares nonnumeric values exactly

Each case owns an upstream scenario and `/model/new` deployment with a unique name. The shared `provider` fixture cannot be used because the cost group runs under pytest-xdist with eight workers. Base cases live in `chat_completions/bases/`, basic tests in `basic/`, and feature cases in `streaming/` or `cache_read/`; feature cases use `dataclasses.replace` and name the base they derive from

`spend_rollups/` covers team, user and end-user attribution with three requests from each base case. The other cost cases remain in `cost_tracking_cases.json` and continue through the legacy harness pending migration
