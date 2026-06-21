"""
Code Interpreter Interception Handler

CustomLogger that swaps the native OpenAI Responses ``code_interpreter`` tool for
a function tool, executes the code the model emits inside a sandbox, and feeds the
captured stdout back through the typed agentic loop plan.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, cast

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.integrations.code_interpreter_interception import (
    CodeInterpreterInterceptionConfig,
)
from litellm.types.integrations.custom_logger import (
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
)
from litellm.types.utils import CallTypes

LITELLM_CODE_EXECUTION_TOOL_NAME = "litellm_code_execution"
_CACHE_TTL_SECONDS = 15 * 60


def _resolve_sandbox_tool(sandbox_tool_name: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        from litellm.sandbox.sandbox_tools import resolve_sandbox_tool
    except ImportError:
        return None
    return resolve_sandbox_tool(sandbox_tool_name)


class CodeInterpreterInterceptionLogger(CustomLogger):
    """
    CustomLogger that implements transparent code-interpreter execution loops.

    Flow:
    1. Replace the native ``code_interpreter`` tool with a function tool in the
       pre-call hook so the model emits code as function-call arguments.
    2. Detect ``litellm_code_execution`` function calls in the model response.
    3. Run the emitted code in a sandbox (reused per request via call_id) and
       build a typed rerun plan that appends the function_call_output.
    """

    def __init__(
        self,
        enabled: bool = True,
        enabled_providers: Optional[List[str]] = None,
        sandbox_tool_name: Optional[str] = None,
        sandbox_config: Optional[Any] = None,
    ):
        super().__init__()
        self.enabled = enabled
        self.enabled_providers = enabled_providers
        self.sandbox_tool_name = sandbox_tool_name
        self.sandbox_config = sandbox_config
        self._container_cache_by_call_id: Dict[str, Tuple[Any, float]] = {}

    @classmethod
    def from_config_yaml(
        cls, config: CodeInterpreterInterceptionConfig
    ) -> "CodeInterpreterInterceptionLogger":
        return cls(
            enabled=bool(config.get("enabled", True)),
            enabled_providers=config.get("enabled_providers"),
            sandbox_tool_name=config.get("sandbox_tool_name"),
        )

    @staticmethod
    def initialize_from_proxy_config(
        litellm_settings: Dict[str, Any],
        callback_specific_params: Dict[str, Any],
    ) -> "CodeInterpreterInterceptionLogger":
        params: CodeInterpreterInterceptionConfig = {}
        if "code_interpreter_interception_params" in litellm_settings:
            params = litellm_settings["code_interpreter_interception_params"]
        elif "code_interpreter_interception" in callback_specific_params and isinstance(
            callback_specific_params["code_interpreter_interception"], dict
        ):
            params = cast(
                CodeInterpreterInterceptionConfig,
                callback_specific_params["code_interpreter_interception"],
            )
        return CodeInterpreterInterceptionLogger.from_config_yaml(params)

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        if not self.enabled:
            return None
        if call_type not in (CallTypes.responses, CallTypes.aresponses):
            return None
        if (
            self.enabled_providers is not None
            and self._resolve_provider(kwargs) not in self.enabled_providers
        ):
            return None

        tools = kwargs.get("tools")
        if not isinstance(tools, list):
            return None
        if not any(
            isinstance(tool, dict) and tool.get("type") == "code_interpreter"
            for tool in tools
        ):
            return None

        function_tool = {
            "type": "function",
            "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
            "description": "Execute python code in a sandbox and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        }
        kwargs["tools"] = [
            (
                function_tool
                if isinstance(tool, dict) and tool.get("type") == "code_interpreter"
                else tool
            )
            for tool in tools
        ]
        return kwargs

    def _resolve_provider(self, kwargs: Dict[str, Any]) -> Optional[str]:
        provider = kwargs.get("custom_llm_provider")
        if provider:
            return provider
        model = kwargs.get("model")
        if not isinstance(model, str):
            return None
        try:
            return litellm.get_llm_provider(model=model)[1]
        except Exception:
            return None

    async def async_should_run_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: Dict,
    ) -> Tuple[bool, Dict]:
        if not self.enabled:
            return False, {}

        tool_calls = self._extract_code_execution_tool_calls(response=response)
        if not tool_calls:
            return False, {}

        return True, {"tool_calls": tool_calls}

    async def async_build_agentic_loop_plan(
        self,
        tools: Dict,
        model: str,
        messages: List[Dict],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
    ) -> AgenticLoopPlan:
        self._prune_expired_cache()
        tool_calls = cast(List[Dict[str, Any]], tools.get("tool_calls", []))
        call_id = self._resolve_call_id(logging_obj=logging_obj, kwargs=kwargs)
        container = await self._get_or_create_container(call_id=call_id)

        container_id = getattr(container, "id", None)
        input_list = self._normalize_messages(messages)
        code_interpreter_calls = []
        for tool_call in tool_calls:
            arguments = tool_call.get("arguments", "")
            code = self._parse_code(arguments)
            stdout = await self._run_tool_call(container=container, arguments=arguments)
            input_list.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.get("call_id"),
                    "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                    "arguments": arguments,
                }
            )
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.get("call_id"),
                    "output": stdout,
                }
            )
            code_interpreter_calls.append(
                {
                    "id": f"ci_{uuid.uuid4().hex}",
                    "type": "code_interpreter_call",
                    "status": "completed",
                    "code": code,
                    "container_id": container_id,
                    "outputs": None,
                }
            )

        optional_params = anthropic_messages_optional_request_params
        request_patch = AgenticLoopRequestPatch(
            model=model,
            messages=input_list,
            tools=optional_params.get("tools"),
            optional_params={k: v for k, v in optional_params.items() if k != "tools"},
            kwargs={k: v for k, v in kwargs.items() if k != "litellm_logging_obj"},
        )

        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=request_patch,
            metadata={
                "tool_type": "code_interpreter",
                "call_id": call_id or "",
                "code_interpreter_calls": code_interpreter_calls,
            },
        )

    async def async_post_agentic_loop_response_hook(
        self, response: Any, plan: AgenticLoopPlan, kwargs: Dict
    ) -> Any:
        calls = (plan.metadata or {}).get("code_interpreter_calls") if plan else None
        if not calls:
            return response

        is_dict = isinstance(response, dict)
        output = (
            response.get("output") if is_dict else getattr(response, "output", None)
        )
        if not isinstance(output, list):
            return response

        def _item_type(item: Any) -> Any:
            return (
                item.get("type")
                if isinstance(item, dict)
                else getattr(item, "type", None)
            )

        insert_at = next(
            (i for i, item in enumerate(output) if _item_type(item) == "message"),
            len(output),
        )
        new_output = output[:insert_at] + list(calls) + output[insert_at:]
        if is_dict:
            response["output"] = new_output
        else:
            response.output = new_output
        return response

    @staticmethod
    def _parse_code(arguments: str) -> str:
        try:
            return json.loads(arguments).get("code", "") if arguments else ""
        except (json.JSONDecodeError, TypeError, AttributeError):
            return ""

    async def _run_tool_call(self, container: Any, arguments: str) -> str:
        try:
            code = json.loads(arguments).get("code", "") if arguments else ""
        except (json.JSONDecodeError, TypeError):
            return "[invalid tool arguments: could not parse code]"

        result = await self._run_code(container=container, code=code)
        if getattr(result, "error", None):
            error = result.error
            message = (
                error.get("value") or error.get("name")
                if isinstance(error, dict)
                else str(error)
            )
            return f"[execution error] {message}"
        return getattr(result, "stdout", "") or ""

    async def _get_or_create_container(self, call_id: Optional[str]) -> Any:
        if call_id:
            cached = self._container_cache_by_call_id.get(call_id)
            if cached is not None:
                return cached[0]

        container = await self._create_container()
        if call_id:
            self._container_cache_by_call_id[call_id] = (container, time.time())
        return container

    async def _create_container(self) -> Any:
        if self.sandbox_config is not None:
            return await self.sandbox_config.acreate_sandbox()

        params = _resolve_sandbox_tool(self.sandbox_tool_name)
        if params is None:
            raise ValueError(
                "CodeInterpreterInterception: no sandbox available. Provide a "
                "sandbox_config or configure a sandbox tool resolvable via "
                "sandbox_tool_name."
            )
        return await litellm.acreate_sandbox(
            provider=params["sandbox_provider"],
            api_key=params.get("api_key"),
            api_base=params.get("api_base"),
        )

    async def _run_code(self, container: Any, code: str) -> Any:
        if self.sandbox_config is not None:
            return await self.sandbox_config.arun_code(container=container, code=code)

        params = _resolve_sandbox_tool(self.sandbox_tool_name)
        if params is None:
            raise ValueError(
                "CodeInterpreterInterception: no sandbox available to run code."
            )
        return await litellm.arun_code(
            provider=params["sandbox_provider"],
            container=container,
            code=code,
            api_key=params.get("api_key"),
        )

    def _normalize_messages(self, messages: Any) -> List[Dict[str, Any]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, list):
            return list(messages)
        return []

    def _extract_code_execution_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        if isinstance(response, dict):
            output = response.get("output", [])
        else:
            output = getattr(response, "output", []) or []
        if not isinstance(output, list):
            return []

        return [
            {
                "call_id": (
                    item.get("call_id")
                    if isinstance(item, dict)
                    else getattr(item, "call_id", None)
                ),
                "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                "arguments": (
                    item.get("arguments")
                    if isinstance(item, dict)
                    else getattr(item, "arguments", "")
                ),
            }
            for item in output
            if self._is_code_execution_call(item)
        ]

    def _is_code_execution_call(self, item: Any) -> bool:
        if isinstance(item, dict):
            return (
                item.get("type") == "function_call"
                and item.get("name") == LITELLM_CODE_EXECUTION_TOOL_NAME
            )
        return (
            getattr(item, "type", None) == "function_call"
            and getattr(item, "name", None) == LITELLM_CODE_EXECUTION_TOOL_NAME
        )

    def _prune_expired_cache(self) -> None:
        now = time.time()
        self._container_cache_by_call_id = {
            call_id: (container, created_at)
            for call_id, (
                container,
                created_at,
            ) in self._container_cache_by_call_id.items()
            if now - created_at <= _CACHE_TTL_SECONDS
        }

    def _resolve_call_id(
        self, logging_obj: Any, kwargs: Dict[str, Any]
    ) -> Optional[str]:
        if logging_obj is not None:
            logging_call_id = getattr(logging_obj, "litellm_call_id", None)
            if isinstance(logging_call_id, str) and logging_call_id:
                return logging_call_id
        kwargs_call_id = kwargs.get("litellm_call_id")
        return kwargs_call_id if isinstance(kwargs_call_id, str) else None
