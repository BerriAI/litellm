# Codex Agent (CLI)

Run the Codex headless CLI from LiteLLM and stream JSONL output—useful for iterative, tool-using workflows.

- Provider slug: `codex-agent` (alias: `codex_cli_agent`)
- Status: Experimental; disabled by default (env-gated)

## Setup

1. Install Codex CLI and authenticate (`codex login`).
2. Set environment variables:
   - `LITELLM_ENABLE_CODEX_AGENT=1` (enable provider)
   - `LITELLM_CODEX_BINARY_PATH=/abs/path/to/codex` (recommended)

## Usage

```python
from litellm import completion
import os

os.environ["LITELLM_ENABLE_CODEX_AGENT"] = "1"
os.environ["LITELLM_CODEX_BINARY_PATH"] = "/absolute/path/to/codex"

resp = completion(
  model="codex-agent/gpt-5",  # maps to --model gpt-5
  messages=[{"role": "user", "content": "plan steps"}],
  optional_params={
    "extra_body": {
      "codex_args": ["--json", "--max-iterations", "5"],
      "codex_sandbox": "read-only",
      "codex_approval_mode": "never"
    }
  },
)
print(resp.choices[0].message.content)
```

## Parameters (extra_body)

- `codex_cli_model`: maps to `--model`
- `codex_args`: list of CLI args passed through as-is
- `codex_sandbox`: `read-only` | `workspace-write` | `danger-full-access`
- `codex_approval_mode`: `never` (default) | `ask` | `always`
- Timeouts: `codex_first_byte_seconds`, `codex_idle_timeout_seconds`, `codex_max_run_seconds`
- Images: `codex_images` (list of file paths)
- Working dir persistence: `working_dir_persistence_key`

## Safety defaults

- `sandbox=read-only`
- `approval_mode=never`
- “YOLO” / full access disabled unless explicitly allowed by env

## Notes

- This provider runs a local CLI, not an HTTP endpoint.
- Follow the provider guide for repo conventions: https://docs.litellm.ai/docs/provider_registration/