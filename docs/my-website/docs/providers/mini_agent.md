# Mini-Agent Provider

The mini-agent adapter exposes the experimental MCP-based agent loop through the
regular LiteLLM `completion`/`acompletion` APIs.

## Enable the provider

1. Export `LITELLM_ENABLE_MINI_AGENT=1` **before** importing `litellm`. The
   package auto-registers the provider when the flag is present.
2. Optionally set `LITELLM_DEFAULT_CODE_MODEL` or `LITELLM_DEFAULT_MODEL` to the
   base LLM you want the agent to delegate to. You can override the base model
   per call with `litellm_params["target_model"]`.
3. (Docker execution) Start the bundled stack (mini-agent shim + codex sidecar + Ollama) and point the provider at it:

   ```bash
   docker compose -f local/docker/compose.agents.yml up --build -d
   export LITELLM_MINI_AGENT_DOCKER_CONTAINER=litellm-mini-agent
   ```

## Register custom handlers programmatically

The fork adds a helper that takes care of wiring a provider into the global
registry:

```python title="register a custom provider"
from litellm.llms.custom_llm import register_custom_provider
from litellm.llms.mini_agent import MiniAgentLLM

register_custom_provider("mini-agent", MiniAgentLLM)
```

The helper accepts either the handler class or an instance and updates
`litellm.custom_provider_map`, `provider_list`, and `_custom_providers` in one
step.

## Required Python packages

Mini-agent executes user-supplied code snippets when you call `exec_python`.
Several scenarios rely on scientific tooling, so the fork pins these packages in
`requirements.txt`:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

If you build slim containers around this fork, make sure those dependencies are
installed so scenarios such as `mini_agent_live.py` continue to succeed.

## Sample Router configuration

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "mini-agent",
            "litellm_params": {
                "model": "mini-agent",
                "custom_llm_provider": "mini-agent",
                "target_model": "ollama/gpt-4.1-mini",
                "allowed_languages": ["python", "rust", "go", "javascript"],
                "max_iterations": 4,
            },
        }
    ]
)
```

By default the provider allows Python, Rust, Go, and JavaScript snippets (mapped to
the command prefixes `python`, `cargo`, `go`, `node`). Override via `allowed_languages`
 or the `LITELLM_MINI_AGENT_LANGUAGES` environment variable to tighten or extend the list.

The provider exposes a minimal tool catalog by default (`exec_python`,
`exec_shell`, `research_echo`, `compress_runs`). Avoid instructing the model to
call arbitrary tool names; the adapter will reject them and continue the loop.
