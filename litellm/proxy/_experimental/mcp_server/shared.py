from __future__ import annotations

"""
Shared helpers for LiteLLM MCP servers (HTTP/SSE and stdio).

Contains small, local convenience tools registration so both transports expose
the same minimal helpful surface (e.g., model selection advice).
"""
from typing import Any

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.tool_registry import (
    MCPTool,
    global_mcp_tool_registry,
)


def register_default_local_tools(registry: Any | None = None) -> None:
    """
    Register small local convenience tools that don't require external servers.

    Currently adds:
    - model.advice: concise guidance on choosing a model given task/latency/cost/context hints.
    """
    reg = registry or global_mcp_tool_registry

    def _model_advice_handler(
        task_description: str = "",
        max_context_tokens: int | None = None,
        latency_sensitivity: str = "normal",
        cost_sensitivity: str = "normal",
    ) -> str:
        task = (task_description or "").lower()
        latency = (latency_sensitivity or "normal").lower()
        cost = (cost_sensitivity or "normal").lower()
        hints: list[str] = []

        # Long context / multimodal → Gemini 2.5 Flash
        if any(
            k in task
            for k in [
                "ingest",
                "summarize long",
                "long pdf",
                "1m",
                "million",
                "context window",
                "multimodal",
                "image",
            ]
        ) or (max_context_tokens is not None and max_context_tokens > 200_000):
            hints.append(
                "gemini/gemini-2.5-flash — best for very long context (~1M) and multimodal."
            )

        # Deep reasoning / complex code → chutes (middle ground) or codex-agent (SOTA)
        if any(
            k in task
            for k in [
                "deep reasoning",
                "tool chain",
                "plan",
                "refactor",
                "complex code",
                "unit tests",
                "long-form reasoning",
            ]
        ):
            if cost in ("low", "normal"):
                hints.append(
                    "chutes/* (DeepSeek ~72B) — middle ground for strong reasoning at lower cost than SOTA."
                )
            else:
                hints.append(
                    "codex-agent/* — SOTA/enterprise routes for complex reasoning and tool use."
                )

        # Quick / local / low-latency → Ollama
        if any(k in task for k in ["quick", "cheap", "local", "offline", "sandbox"]) or latency == "low":
            hints.append(
                "ollama_chat/<local-tag> — fast and local for short tasks and iteration."
            )

        if not hints:
            hints = [
                "Local quick: ollama_chat/<tag>.",
                "Middle ground reasoning: chutes/* (DeepSeek ~72B).",
                "SOTA/enterprise: codex-agent/*.",
                "Large context: gemini/gemini-2.5-flash.",
            ]

        rec = "\n- ".join(hints)
        return (
            "Model selection guide:\n"
            "- Use ollama_chat/<tag> for fast/local short tasks.\n"
            "- Use chutes/* (DeepSeek ~72B) as middle ground for strong reasoning.\n"
            "- Use codex-agent/* for SOTA/enterprise routes and complex tool use.\n"
            "- Use gemini/gemini-2.5-flash for ~1M token context or multimodal.\n\n"
            f"Based on your input, consider:\n- {rec}"
        )

    try:
        reg.register_tool(
            name="model.advice",
            description=(
                "Return concise guidance on choosing a model: ollama (local), chutes (DeepSeek ~72B), "
                "codex-agent (SOTA/enterprise), or gemini-2.5-flash (~1M context)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "What are you trying to do?",
                    },
                    "max_context_tokens": {
                        "type": "integer",
                        "description": "Estimated context window needed (tokens).",
                    },
                    "latency_sensitivity": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "default": "normal",
                    },
                    "cost_sensitivity": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "default": "normal",
                        "description": "Budget/quality tilt: low/normal favors chutes or ollama; high favors SOTA via codex-agent.",
                    },
                },
                "required": [],
            },
            handler=_model_advice_handler,
        )
    except Exception as e:
        verbose_logger.debug(f"Default MCP tools registration failed: {e}")

    # LLM chat tool – frictionless model usage via MCP
    # Agents can call any LiteLLM-supported model, e.g.,
    #  - gemini/gemini-2.5-flash
    #  - codex-agent/mini (requires LITELLM_ENABLE_CODEX_AGENT=1)
    #  - ollama_chat/<tag>
    #  - chutes/<vendor>/<model> (mapped to OpenAI-compatible route via env)
    async def _llm_chat_handler(
        model: str,
        messages: list[dict],
        stream: bool | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        custom_llm_provider: str | None = None,
        **kwargs: Any,
    ) -> str:
        import os
        import json as _json
        import litellm

        params: dict[str, Any] = {}
        if api_base is not None:
            params["api_base"] = api_base
        if api_key is not None:
            params["api_key"] = api_key
        if custom_llm_provider is not None:
            params["custom_llm_provider"] = custom_llm_provider

        # codex-agent/* mapping: map to OpenAI-compatible with CODEX_AGENT_* env
        if isinstance(model, str) and model.startswith("codex-agent/"):
            try:
                remainder = model.split("/", 1)[1] if "/" in model else model
            except Exception:
                remainder = model
            ca_api_base = os.getenv("CODEX_AGENT_API_BASE")
            ca_api_key = os.getenv("CODEX_AGENT_API_KEY") or os.getenv("CODEX_AGENT_API_TOKEN") or "sk-stub"
            if ca_api_base and "api_base" not in params:
                params["api_base"] = ca_api_base
            if ca_api_key and "api_key" not in params:
                params["api_key"] = ca_api_key
            params.setdefault("custom_llm_provider", "openai")
            model = remainder

        # chutes/* mapping: map to OpenAI-compatible with CHUTES_* env
        if isinstance(model, str) and model.startswith("chutes/"):
            try:
                remainder = model.split("/", 1)[1] if "/" in model else model
            except Exception:
                remainder = model
            ch_api_base = os.getenv("CHUTES_API_BASE") or os.getenv("CHUTES_BASE")
            ch_api_key = os.getenv("CHUTES_API_KEY") or os.getenv("CHUTES_API_TOKEN")
            if ch_api_base and "api_base" not in params:
                params["api_base"] = ch_api_base
            if ch_api_key and "api_key" not in params:
                params["api_key"] = ch_api_key
            params.setdefault("custom_llm_provider", "openai")
            model = remainder

        # Non-streaming by default for MCP tool; ignore stream unless true
        do_stream = bool(stream)
        if do_stream:
            # For now, coalesce streamed chunks into a single string
            full = []
            async for chunk in await litellm.astream_completion(
                model=model, messages=messages, **params, **kwargs
            ):
                try:
                    delta = (
                        getattr(chunk.choices[0].delta, "content", None)
                        or getattr(getattr(chunk.choices[0], "message", {}), "content", None)
                        or ""
                    )
                except Exception:
                    delta = ""
                if delta:
                    full.append(delta)
            return "".join(full)
        else:
            out = await litellm.acompletion(model=model, messages=messages, **params, **kwargs)
            # Return the assistant content if present
            try:
                content = (
                    getattr(getattr(out.choices[0], "message", {}), "content", None)
                    or getattr(out.choices[0], "text", None)
                    or ""
                )
            except Exception:
                content = ""
            if isinstance(content, (list, dict)):
                return _json.dumps(content)
            return str(content or "")

    try:
        reg.register_tool(
            name="llm.chat",
            description=(
                "Call any LiteLLM-supported model with OpenAI-style messages. "
                "Examples: model='gemini/gemini-2.5-flash', 'codex-agent/mini', 'ollama_chat/qwen3:8b', 'chutes/vendor/model'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Target model id"},
                    "messages": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "OpenAI-style messages array",
                    },
                    "stream": {"type": "boolean", "default": False},
                    "api_base": {"type": "string"},
                    "api_key": {"type": "string"},
                    "custom_llm_provider": {"type": "string"},
                },
                "required": ["model", "messages"],
            },
            handler=_llm_chat_handler,
        )
    except Exception as e:
        verbose_logger.debug(f"llm.chat tool registration failed: {e}")
