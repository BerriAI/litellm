from __future__ import annotations

"""Custom provider exposing the experimental mini-agent via LiteLLM's Router."""

import asyncio
import json
import os
from typing import Iterable, Tuple, List, Dict, Any

from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.utils import ModelResponse

try:
    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
        AgentConfig,
        LocalMCPInvoker,
        arun_mcp_mini_agent,
    )
except Exception as exc:  # pragma: no cover - import guard
    AgentConfig = None  # type: ignore
    LocalMCPInvoker = None  # type: ignore
    arun_mcp_mini_agent = None  # type: ignore


DEFAULT_LANGS = ["python"]
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
    return tuple(langs)


def _pick_base_model(optional_params: dict) -> str:
    target = optional_params.get("target_model") or optional_params.get("mini_agent_model")
    if not target:
        target = os.getenv("LITELLM_DEFAULT_CHUTES_MODEL") or os.getenv("LITELLM_DEFAULT_CODE_MODEL")
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

        cfg = AgentConfig(
            model=base_model,
            max_iterations=max_iterations,
            max_total_seconds=max_seconds,
            use_tools=True,
            auto_run_code_on_code_block=True,
        )
        return cfg, allowed_langs

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
            model_response.choices[0].message = {"role": "assistant", "content": content}
        except Exception:  # pragma: no cover - defensive
            model_response.choices.append(type(model_response.choices[0])(message={"role": "assistant", "content": content}))
        additional = {
            "mini_agent": {
                "iterations": len(result.iterations),
                "stopped_reason": result.stopped_reason,
                "conversation": result.messages,
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

        cfg, allowed = self._build_config(model, optional_params or {})
        invoker = LocalMCPInvoker(shell_allow_prefixes=allowed, tool_timeout_sec=cfg.max_total_seconds or DEFAULT_MAX_SECONDS)

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

        cfg, allowed = self._build_config(model, optional_params or {})
        invoker = LocalMCPInvoker(shell_allow_prefixes=allowed, tool_timeout_sec=cfg.max_total_seconds or DEFAULT_MAX_SECONDS)

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
