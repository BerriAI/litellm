from __future__ import annotations

"""Custom provider exposing the experimental mini-agent via LiteLLM's Router."""

import asyncio
import json
import os
from typing import Tuple, List, Dict, Any

from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.utils import ModelResponse

try:
    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
        AgentConfig,
        LocalMCPInvoker,
        DockerMCPInvoker,
        arun_mcp_mini_agent,
    )
except Exception as exc:  # pragma: no cover - import guard
    AgentConfig = None  # type: ignore
    LocalMCPInvoker = None  # type: ignore
    DockerMCPInvoker = None  # type: ignore
    arun_mcp_mini_agent = None  # type: ignore


DEFAULT_LANGS = ["python", "rust", "go", "javascript"]

LANGUAGE_PREFIXES = {
    "python": ("python", "python3"),
    "rust": ("cargo", "rustc"),
    "go": ("go",),
    "bash": ("bash", "sh"),
    "shell": ("sh", "bash"),
    "javascript": ("node", "npm", "npx"),
    "typescript": ("ts-node", "npx ts-node"),
    "deno": ("deno",),
}
DEFAULT_MAX_ITER = int(os.getenv("LITELLM_MINI_AGENT_MAX_ITERATIONS", "6"))
DEFAULT_MAX_SECONDS = float(os.getenv("LITELLM_MINI_AGENT_MAX_SECONDS", "180"))


def _gather_allowed_languages(optional_params: dict) -> Tuple[str, ...]:
    langs = optional_params.get("allowed_languages") or optional_params.get("mini_agent_allowed_languages")
    if isinstance(langs, str):
        langs = [part.strip() for part in langs.split(",") if part.strip()]
    if not langs:
        env_val = os.getenv("LITELLM_MINI_AGENT_LANGUAGES")
        if env_val:
            langs = [part.strip() for part in env_val.split(",") if part.strip()]
    if not langs:
        langs = list(DEFAULT_LANGS)
    # Normalize: lowercase, strip, unique preserving order
    normalized = []
    seen = set()
    for item in langs:
        token = str(item).strip()
        if not token:
            continue
        token = token.lower()
        if token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return tuple(normalized)


def _expand_shell_prefixes(languages: Tuple[str, ...]) -> Tuple[str, ...]:
    prefixes: List[str] = []
    def _add_prefix(value: str) -> None:
        token = value.strip()
        if not token:
            return
        if token not in prefixes:
            prefixes.append(token)

    for lang in languages:
        mapping = LANGUAGE_PREFIXES.get(lang, (lang,))
        if isinstance(mapping, str):
            mapping = (mapping,)
        for prefix in mapping:
            _add_prefix(prefix)
    # Always allow echo for trivial smoke tests / readiness probes
    _add_prefix("echo")
    return tuple(prefixes)


def _pick_base_model(optional_params: dict) -> str:
    target = optional_params.get("target_model") or optional_params.get("mini_agent_model")
    if not target:
        target = (
            os.getenv("LITELLM_DEFAULT_CHUTES_MODEL")
            or os.getenv("LITELLM_DEFAULT_CODE_MODEL")
            or os.getenv("LITELLM_DEFAULT_MODEL")
        )
    if not target:
        raise CustomLLMError(status_code=400, message="mini-agent requires 'target_model' or defaults; none found")
    return str(target)


class MiniAgentLLM(CustomLLM):
    """Expose the mini-agent loop as a LiteLLM provider."""

    def _build_config(self, model: str, optional_params: dict) -> Tuple[AgentConfig, Tuple[str, ...]]:
        if AgentConfig is None:
            raise CustomLLMError(status_code=500, message="mini-agent components unavailable")

        base_model = optional_params.get("mini_agent_target") or _pick_base_model(optional_params)
        max_iterations = int(optional_params.get("max_iterations") or optional_params.get("mini_agent_max_iterations") or DEFAULT_MAX_ITER)
        max_seconds = float(optional_params.get("max_seconds") or optional_params.get("mini_agent_max_seconds") or DEFAULT_MAX_SECONDS)
        allowed_langs = _gather_allowed_languages(optional_params)

        completion_kwargs: Dict[str, Any] = {}

        provider_hint = ""
        if isinstance(base_model, str) and "/" in base_model:
            provider_hint = base_model.split("/", 1)[0].lower()
        else:
            provider_hint = str(optional_params.get("mini_agent_provider") or "").lower()

        def _cast_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _cast_int(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        temperature = optional_params.get("temperature")
        seed = optional_params.get("seed")
        response_format = optional_params.get("response_format")
        tool_choice = optional_params.get("tool_choice")

        openai_like = {"openai", "azure", "azure_openai"}

        def _maybe_set(key: str, value) -> None:
            if value is not None:
                completion_kwargs[key] = value

        if provider_hint in openai_like:
            _maybe_set("temperature", _cast_float(temperature))
            _maybe_set("seed", _cast_int(seed))
            if isinstance(response_format, dict):
                completion_kwargs["response_format"] = response_format
            if tool_choice is not None:
                completion_kwargs["tool_choice"] = tool_choice
        elif provider_hint == "ollama":
            _maybe_set("temperature", _cast_float(temperature))
        elif provider_hint.startswith("gemini"):
            _maybe_set("temperature", _cast_float(temperature))
        else:
            _maybe_set("temperature", _cast_float(temperature))

        cfg = AgentConfig(
            model=base_model,
            max_iterations=max_iterations,
            max_total_seconds=max_seconds,
            use_tools=True,
            auto_run_code_on_code_block=True,
            completion_kwargs=completion_kwargs,
        )
        return cfg, allowed_langs

    def _make_invoker(self, allowed: Tuple[str, ...], cfg: "AgentConfig", optional_params: dict):
        shell_prefixes = _expand_shell_prefixes(allowed)
        backend = (
            optional_params.get("tool_backend")
            or optional_params.get("mini_agent_backend")
            or os.getenv("LITELLM_MINI_AGENT_TOOL_BACKEND", "local")
        )
        backend = str(backend or "local").strip().lower()
        timeout = float(cfg.max_total_seconds or DEFAULT_MAX_SECONDS)
        if backend == "docker":
            if DockerMCPInvoker is None:
                raise CustomLLMError(status_code=500, message="mini-agent docker backend unavailable")
            container = (
                optional_params.get("docker_container")
                or optional_params.get("mini_agent_docker_container")
                or os.getenv("LITELLM_MINI_AGENT_DOCKER_CONTAINER")
            )
            if not container:
                raise CustomLLMError(
                    status_code=400,
                    message="mini-agent docker backend requires 'docker_container' or LITELLM_MINI_AGENT_DOCKER_CONTAINER",
                )
            return DockerMCPInvoker(
                container=str(container),
                shell_allow_prefixes=shell_prefixes,
                tool_timeout_sec=timeout,
                docker_exec_bin=os.getenv("LITELLM_MINI_AGENT_DOCKER_EXEC_BIN", "docker"),
                python_bin=os.getenv("LITELLM_MINI_AGENT_DOCKER_PYTHON", "python"),
                shell_bin=os.getenv("LITELLM_MINI_AGENT_DOCKER_SHELL", "/bin/sh"),
            )
        return LocalMCPInvoker(shell_allow_prefixes=shell_prefixes, tool_timeout_sec=timeout)

    def _to_model_response(self, model_response: ModelResponse, result) -> ModelResponse:
        content = result.final_answer or ""
        if not content:
            content = json.dumps(
                {
                    "iterations": len(result.iterations),
                    "stopped_reason": result.stopped_reason,
                }
            )
        model_response.model = result.used_model or model_response.model
        try:
            message = model_response.choices[0].message
            if hasattr(message, "role"):
                try:
                    setattr(message, "role", "assistant")
                except Exception:
                    pass
            if hasattr(message, "content"):
                setattr(message, "content", content)
            else:
                raise AttributeError
        except Exception:
            try:
                model_response.choices[0].message = {"role": "assistant", "content": content}
            except Exception:  # pragma: no cover - defensive
                model_response.choices.append(
                    type(model_response.choices[0])(message={"role": "assistant", "content": content})
                )
        try:
            setattr(model_response.choices[0], "finish_reason", result.stopped_reason or "stop")
        except Exception:
            pass
        parsed_tools: List[Dict[str, Any]] = []
        for iter_idx, iteration in enumerate(getattr(result, "iterations", []) or []):
            tool_invocations = getattr(iteration, "tool_invocations", []) or []
            for tool_idx, invocation in enumerate(tool_invocations):
                if not isinstance(invocation, dict):
                    continue
                tool_name = invocation.get("tool_name") or invocation.get("name")
                call_index = invocation.get("call_index", tool_idx)
                t_ms = invocation.get("t_ms")
                try:
                    t_ms_val = float(t_ms) if t_ms is not None else None
                except Exception:
                    t_ms_val = t_ms

                arguments_raw = invocation.get("arguments_raw")
                arguments_value = invocation.get("arguments")
                if isinstance(arguments_value, dict):
                    arguments_dict = arguments_value
                elif isinstance(arguments_value, str):
                    try:
                        arguments_dict = json.loads(arguments_value)
                    except Exception:
                        arguments_dict = None
                else:
                    arguments_dict = None

                entry: Dict[str, Any] = {
                    "iteration": iter_idx,
                    "call_index": call_index,
                    "tool_name": tool_name,
                    "tool_call_id": invocation.get("tool_call_id"),
                    "arguments_raw": arguments_raw if isinstance(arguments_raw, str) else None,
                    "arguments": arguments_dict,
                    "ok": invocation.get("ok"),
                    "rc": invocation.get("rc"),
                    "result": invocation.get("result"),
                    "stdout": invocation.get("stdout"),
                    "stderr": invocation.get("stderr"),
                    "error": invocation.get("error"),
                    "t_ms": t_ms_val,
                    "model": invocation.get("model"),
                }
                # Preserve backwards-compatible keys for downstream consumers expecting older schema
                entry.setdefault("name", tool_name)
                entry.setdefault("index", call_index)
                parsed_tools.append(entry)

        additional = {
            "mini_agent": {
                "iterations": len(result.iterations),
                "stopped_reason": result.stopped_reason,
                "conversation": result.messages,
                "parsed_tools": parsed_tools,
            }
        }
        try:
            if hasattr(model_response, "additional_kwargs"):
                model_response.additional_kwargs.update(additional)
            else:
                model_response.additional_kwargs = additional
        except Exception:
            pass
        return model_response

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base,
        custom_prompt_dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict = None,
        timeout=None,
        client=None,
    ) -> ModelResponse:
        if arun_mcp_mini_agent is None:
            raise CustomLLMError(status_code=500, message="mini-agent components unavailable")

        optional_params = optional_params or {}
        cfg, allowed = self._build_config(model, optional_params)
        invoker = self._make_invoker(allowed, cfg, optional_params)

        async def _run():
            return await arun_mcp_mini_agent(messages=messages, mcp=invoker, cfg=cfg)

        try:
            running_loop = None
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop and running_loop.is_running():
                result_container: Dict[str, Any] = {}
                error_container: Dict[str, Exception] = {}

                def _thread_runner() -> None:
                    loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(loop)
                        result_container["value"] = loop.run_until_complete(_run())
                    except Exception as exc:  # pragma: no cover
                        error_container["error"] = exc
                    finally:
                        loop.close()
                        asyncio.set_event_loop(None)

                import threading

                thread = threading.Thread(target=_thread_runner)
                thread.start()
                thread.join()
                if "error" in error_container:
                    raise error_container["error"]
                result = result_container["value"]
            else:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(_run())
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
        except CustomLLMError:
            raise
        except Exception as exc:  # pragma: no cover
            raise CustomLLMError(status_code=500, message=str(exc)) from exc

        return self._to_model_response(model_response, result)

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base,
        custom_prompt_dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict = None,
        timeout=None,
        client=None,
    ) -> ModelResponse:
        if arun_mcp_mini_agent is None:
            raise CustomLLMError(status_code=500, message="mini-agent components unavailable")

        optional_params = optional_params or {}
        cfg, allowed = self._build_config(model, optional_params)
        invoker = self._make_invoker(allowed, cfg, optional_params)

        try:
            result = await arun_mcp_mini_agent(messages=messages, mcp=invoker, cfg=cfg)
        except CustomLLMError:
            raise
        except Exception as exc:  # pragma: no cover
            raise CustomLLMError(status_code=500, message=str(exc)) from exc

        return self._to_model_response(model_response, result)


try:
    if os.getenv("LITELLM_ENABLE_MINI_AGENT", "").strip() == "1":
        from litellm.llms.custom_llm import register_custom_provider

        register_custom_provider("mini-agent", MiniAgentLLM)
except Exception:
    pass
