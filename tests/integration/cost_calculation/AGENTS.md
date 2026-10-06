# tests/integration/cost_calculation

The chat-completion pilot uses literal `CostTrackingTestCase` values and a straight-line runner that checks the proxy response and the complete spend-log row. Expected values come from the checked-in test cost map and scripted provider responses

Keep one complete base case per model in `chat_completions/bases/<provider>.py`. Put basic tests in `basic/` and changed request/response scenarios in `streaming/` or `cache_read/`. Name feature cases `<MODEL>_<SCENARIO>_TEST_CASE`, derive them with `dataclasses.replace`, and import the base under its own name. The shared `provider` fixture is serial-only and cannot be used here because the cost group runs with eight xdist workers; register a unique upstream scenario and `/model/new` deployment per test instead

Only migrate cases selected for this pilot. All other cases stay in `cost_tracking_cases.json` and the legacy harness
