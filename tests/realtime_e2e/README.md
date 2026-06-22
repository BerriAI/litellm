# Realtime e2e tests

End-to-end tests for the proxy realtime WebSocket endpoint. They connect to an
externally running litellm proxy over a real WebSocket, hit real provider APIs,
and assert the proxy normalizes every provider into the OpenAI GA realtime event
schema.

Two layers:

- Layer 1 (raw WebSocket) is the source of truth. `test_realtime_conversation.py`
  and `test_realtime_tool_calling.py` speak the GA protocol directly and assert
  precise event sequences, delta/transcript consistency, usage, and the full
  tool-call round-trip (call -> tool result -> follow-up that uses the result).
- Layer 2 (pipecat) is a realism smoke. `test_realtime_pipecat_smoke.py` drives
  the proxy through pipecat's GA `OpenAIRealtimeLLMService` to exercise the
  audio/function-call wiring. Coarse assertions only; skipped unless pipecat is
  installed.

All tests carry the `realtime_e2e` marker and skip cleanly when the proxy is
unreachable or a provider's credentials are absent, so they never run in the
default unit suite.

## 1. Start the proxy

```bash
python litellm/proxy/proxy_cli.py \
  --config tests/realtime_e2e/realtime_e2e_config.yaml \
  --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log
```

Edit `realtime_e2e_config.yaml` so the upstream model ids match what your
accounts can access. The `model_name` aliases must stay in sync with
`providers.py`.

## 2. Set the env contract

```bash
export LITELLM_PROXY_WS_URL=ws://0.0.0.0:4000     # default
export LITELLM_PROXY_URL=http://0.0.0.0:4000      # default, used for health check
export LITELLM_PROXY_API_KEY=sk-1234              # default (config master_key)
```

Provider credentials are read by the proxy from its own environment. A provider
test skips when its `required_env` (see `providers.py`) is missing, so set only
the keys for the providers you want to exercise (e.g. `OPENAI_API_KEY`).

## 3. Run

```bash
# Layer 1 only (deterministic, recommended default)
poetry run pytest tests/realtime_e2e -m realtime_e2e \
  --ignore=tests/realtime_e2e/test_realtime_pipecat_smoke.py

# A single provider
poetry run pytest tests/realtime_e2e/test_realtime_tool_calling.py -k openai

# Layer 2 (install pipecat first)
poetry run pip install "pipecat-ai[openai]"
poetry run pytest tests/realtime_e2e/test_realtime_pipecat_smoke.py -m realtime_e2e
```

## Notes

- These tests cost real money; they are intentionally excluded from the default
  unit run.
- pipecat tool calling over the realtime service has been flaky upstream
  (pipecat-ai/pipecat#2544). If a Layer 2 test fails but the matching Layer 1
  tool test passes, suspect pipecat, not litellm.
- Provider realtime model ids drift; confirm the ids in
  `realtime_e2e_config.yaml` against current provider docs before a run.
