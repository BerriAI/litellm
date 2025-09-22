# Codex Agent (Experimental, Env‑Gated)

Integrate an experimental “codex‑agent” via the LiteLLM Router for iterative, tool‑using workflows.
It is opt‑in and disabled by default. Use through Router like any other model.

- Provider slug: `codex-agent` (alias: `codex_cli_agent`)
- Status: Experimental; disabled by default (env‑gated)

## Enabling (Environment‑Gated)

Set the feature flag to opt‑in explicitly:

```bash
export LITELLM_ENABLE_CODEX_AGENT=1
```

Add a Router model entry that points to the provider alias (no client changes):

```python
from litellm import Router
router = Router(model_list=[
  {"model_name": "codex-agent-1", "litellm_params": {"model": "codex-agent/mini"}},
])
```

## Usage

```python
from litellm import Router
import os

os.environ["LITELLM_ENABLE_CODEX_AGENT"] = "1"
router = Router(model_list=[
  {"model_name": "codex-agent-1", "litellm_params": {"model": "codex-agent/mini"}},
])

resp = await router.acompletion(
  model="codex-agent-1",
  messages=[{"role": "user", "content": "Plan steps and use tools as needed."}],
)
print(resp.choices[0].message.content)
```

## Notes

- Experimental surface; subject to change.
- Off by default; enable only via `LITELLM_ENABLE_CODEX_AGENT=1`.
- Use through Router like any other model; keep CI guarded by the flag.

### Disable

Unset the flag or remove the Router model entry:

```bash
unset LITELLM_ENABLE_CODEX_AGENT
```

Follow the provider guide for repo conventions: https://docs.litellm.ai/docs/provider_registration/