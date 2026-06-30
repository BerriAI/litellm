"""
Code Interpreter Interception Handler

CustomLogger that swaps the native OpenAI Responses ``code_interpreter`` tool for
a function tool, executes the code the model emits inside a sandbox, and feeds the
captured stdout back through the typed agentic loop plan.
"""

import json
import time
import uuid
from typing import Any, Literal, TypedDict, cast

import litellm
from pydantic import ValidationError

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.integrations.code_interpreter_interception import (
    CodeInterpreterInterceptionConfig,
)
from litellm.types.integrations.custom_logger import (
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
    CHAT_COMPLETION_AGENTIC_SURFACE,
    NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES,
    is_interception_internal_key,
)
from litellm.types.llms.openai import (
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionToolMessage,
)
from litellm.types.utils import (
    CallTypes,
    ChatCompletionMessageToolCall,
    ModelResponse,
)

LITELLM_CODE_EXECUTION_TOOL_NAME = "litellm_code_execution"
_INTERCEPTION_ACTIVE_KEY = "_code_interpreter_interception_active"
_SANDBOX_KEY = "_code_interpreter_interception_sandbox_key"
_CONVERTED_STREAM_KEY = "_code_interpreter_interception_converted_stream"
_LITELLM_METADATA_KEY = "litellm_metadata"
_CACHE_TTL_SECONDS = 15 * 60


class CodeExecutionToolCall(TypedDict, total=False):
    id: str | None
    call_id: str | None
    type: Literal["function"]
    name: str
    arguments: str


class CodeInterpreterLogOutput(TypedDict):
    type: Literal["logs"]
    logs: str


class CodeInterpreterCall(TypedDict):
    id: str
    type: Literal["code_interpreter_call"]
    status: Literal["completed"]
    code: str
    container_id: str | None
    outputs: list[CodeInterpreterLogOutput]


class CodeExecutionFunctionParameters(TypedDict):
    type: Literal["object"]
    properties: dict[str, dict[str, str]]
    required: list[str]


class ResponsesFunctionTool(TypedDict):
    type: Literal["function"]
    name: str
    description: str
    parameters: CodeExecutionFunctionParameters


class ChatCompletionFunctionDefinition(TypedDict):
    name: str
    description: str
    parameters: CodeExecutionFunctionParameters


class ChatCompletionFunctionTool(TypedDict):
    type: Literal["function"]
    function: ChatCompletionFunctionDefinition


CodeExecutionFunctionTool = ResponsesFunctionTool | ChatCompletionFunctionTool


class ResponsesFunctionToolChoice(TypedDict):
    type: Literal["function"]
    name: str


class ChatCompletionFunctionToolChoice(TypedDict):
    type: Literal["function"]
    function: dict[str, str]


CodeExecutionFunctionToolChoice = ResponsesFunctionToolChoice | ChatCompletionFunctionToolChoice


def _resolve_sandbox_tool(sandbox_tool_name: str | None) -> dict[str, Any] | None:
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
    3. Run the emitted code in a sandbox (reused per request via a server-minted
       sandbox key) and build a typed rerun plan that appends the
       function_call_output.
    """

    def __init__(
        self,
        enabled: bool = True,
        enabled_providers: list[str] | None = None,
        sandbox_tool_name: str | None = None,
        sandbox_config: Any | None = None,
    ):
        super().__init__()
        self.enabled = enabled
        self.enabled_providers = enabled_providers
        self.sandbox_tool_name = sandbox_tool_name
        self.sandbox_config = sandbox_config
        self._container_cache: dict[str, tuple[Any, dict[str, Any] | None, float]] = {}

    @classmethod
    def from_config_yaml(cls, config: CodeInterpreterInterceptionConfig) -> "CodeInterpreterInterceptionLogger":
        return cls(
            enabled=bool(config.get("enabled", True)),
            enabled_providers=config.get("enabled_providers"),
            sandbox_tool_name=config.get("sandbox_tool_name"),
        )

    @staticmethod
    def initialize_from_proxy_config(
        litellm_settings: dict[str, Any],
        callback_specific_params: dict[str, Any],
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

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, Any], call_type: CallTypes | None) -> dict | None:
        if not kwargs.get("_agentic_loop_depth"):
            kwargs.pop(_INTERCEPTION_ACTIVE_KEY, None)
            kwargs.pop(_SANDBOX_KEY, None)
            self._strip_interception_metadata(kwargs)
        if not self.enabled:
            return None
        if call_type not in (
            CallTypes.responses,
            CallTypes.aresponses,
            CallTypes.completion,
            CallTypes.acompletion,
        ):
            return None
        if self.enabled_providers is not None and self._resolve_provider(kwargs) not in self.enabled_providers:
            return None

        tools = kwargs.get("tools")
        if not isinstance(tools, list):
            return None
        if not any(isinstance(tool, dict) and tool.get("type") == "code_interpreter" for tool in tools):
            return None

        kwargs[_INTERCEPTION_ACTIVE_KEY] = True
        kwargs[_SANDBOX_KEY] = uuid.uuid4().hex
        if kwargs.get("stream"):
            kwargs["stream"] = False
            kwargs[_CONVERTED_STREAM_KEY] = True
        self._write_interception_metadata(kwargs)

        function_tool = self._get_function_tool(call_type=call_type)
        kwargs["tools"] = [
            (function_tool if isinstance(tool, dict) and tool.get("type") == "code_interpreter" else tool)
            for tool in tools
        ]
        if self._tool_choice_targets_code_interpreter(kwargs.get("tool_choice")):
            kwargs["tool_choice"] = self._get_function_tool_choice(call_type=call_type)
        return kwargs

    @staticmethod
    def _strip_interception_metadata(kwargs: dict[str, Any]) -> None:
        metadata = kwargs.get(_LITELLM_METADATA_KEY)
        if not isinstance(metadata, dict):
            return
        filtered_metadata = {
            key: value
            for key, value in metadata.items()
            if not is_interception_internal_key(key)
            and not key.startswith("_agentic_loop")
            and key != "max_agentic_loops"
        }
        if filtered_metadata:
            kwargs[_LITELLM_METADATA_KEY] = filtered_metadata
        else:
            kwargs.pop(_LITELLM_METADATA_KEY, None)

    @staticmethod
    def _write_interception_metadata(kwargs: dict[str, Any]) -> None:
        metadata = kwargs.get(_LITELLM_METADATA_KEY)
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        for key in (_INTERCEPTION_ACTIVE_KEY, _SANDBOX_KEY, _CONVERTED_STREAM_KEY):
            if key in kwargs:
                metadata[key] = kwargs[key]
        kwargs[_LITELLM_METADATA_KEY] = metadata

    @staticmethod
    def _get_function_parameters() -> CodeExecutionFunctionParameters:
        return {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        }

    def _get_function_tool(self, call_type: CallTypes | None) -> CodeExecutionFunctionTool:
        description = "Execute python code in a sandbox and return stdout."
        if call_type in (CallTypes.completion, CallTypes.acompletion):
            return {
                "type": "function",
                "function": {
                    "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                    "description": description,
                    "parameters": self._get_function_parameters(),
                },
            }
        return {
            "type": "function",
            "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
            "description": description,
            "parameters": self._get_function_parameters(),
        }

    @staticmethod
    def _get_function_tool_choice(
        call_type: CallTypes | None,
    ) -> CodeExecutionFunctionToolChoice:
        if call_type in (CallTypes.completion, CallTypes.acompletion):
            return {
                "type": "function",
                "function": {"name": LITELLM_CODE_EXECUTION_TOOL_NAME},
            }
        return {
            "type": "function",
            "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
        }

    @staticmethod
    def _tool_choice_targets_code_interpreter(tool_choice: Any) -> bool:
        if not isinstance(tool_choice, dict):
            return False
        function = tool_choice.get("function")
        return (
            tool_choice.get("type") == "code_interpreter"
            or tool_choice.get("name") == "code_interpreter"
            or tool_choice.get("name") == LITELLM_CODE_EXECUTION_TOOL_NAME
            or (isinstance(function, dict) and function.get("name") == LITELLM_CODE_EXECUTION_TOOL_NAME)
        )

    def _resolve_provider(self, kwargs: dict[str, Any]) -> str | None:
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
        messages: list[dict],
        tools: list[dict] | None,
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict,
    ) -> tuple[bool, dict]:
        if not self.enabled:
            return False, {}
        if not kwargs.get(_INTERCEPTION_ACTIVE_KEY):
            return False, {}
        if self.enabled_providers is not None and custom_llm_provider not in self.enabled_providers:
            return False, {}

        tool_calls = (
            self._extract_chat_completion_code_execution_tool_calls(response=response)
            if kwargs.get("_agentic_loop_api_surface") == CHAT_COMPLETION_AGENTIC_SURFACE
            else self._extract_code_execution_tool_calls(response=response)
        )
        if not tool_calls:
            return False, {}

        return True, {"tool_calls": tool_calls}

    async def async_build_agentic_loop_plan(
        self,
        tools: dict,
        model: str,
        messages: list[dict],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: dict,
        logging_obj: Any,
        stream: bool,
        kwargs: dict,
    ) -> AgenticLoopPlan:
        if kwargs.get("_agentic_loop_api_surface") == CHAT_COMPLETION_AGENTIC_SURFACE:
            return await self._build_chat_completion_agentic_loop_plan(
                tools=tools,
                model=model,
                messages=messages,
                optional_params=anthropic_messages_optional_request_params,
                kwargs=kwargs,
            )

        await self._prune_expired_cache()
        tool_calls = cast(list[CodeExecutionToolCall], tools.get("tool_calls", []))
        sandbox_key = kwargs.get(_SANDBOX_KEY)
        container, params = await self._get_or_create_container(cache_key=sandbox_key)

        try:
            container_id = cast(str | None, getattr(container, "id", None))
            input_list = self._normalize_messages(messages)
            code_interpreter_calls: list[CodeInterpreterCall] = []
            for tool_call in tool_calls:
                arguments = tool_call.get("arguments", "")
                code = self._parse_code(arguments)
                stdout = await self._run_tool_call(container=container, params=params, arguments=arguments)
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
                        "outputs": ([{"type": "logs", "logs": stdout}] if stdout else []),
                    }
                )
        except Exception:
            await self._delete_container_for_cache_key(sandbox_key)
            raise

        optional_params = anthropic_messages_optional_request_params
        request_patch = AgenticLoopRequestPatch(
            model=model,
            messages=input_list,
            tools=self._get_followup_tools(
                tools=optional_params.get("tools"),
                call_type=CallTypes.responses,
            ),
            optional_params=self._get_followup_optional_params(optional_params),
            kwargs=self._filter_agentic_loop_kwargs(kwargs),
        )

        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=request_patch,
            metadata={
                "tool_type": "code_interpreter",
                "sandbox_key": sandbox_key or "",
                "code_interpreter_calls": code_interpreter_calls,
            },
        )

    async def _build_chat_completion_agentic_loop_plan(
        self,
        tools: dict[str, object],
        model: str,
        messages: list[dict],
        optional_params: dict[str, object],
        kwargs: dict[str, object],
    ) -> AgenticLoopPlan:
        await self._prune_expired_cache()
        tool_calls = cast(list[CodeExecutionToolCall], tools.get("tool_calls", []))
        sandbox_key = cast(str | None, kwargs.get(_SANDBOX_KEY))
        container, params = await self._get_or_create_container(cache_key=sandbox_key)

        try:
            container_id = cast(str | None, getattr(container, "id", None))
            tool_results = [
                await self._build_chat_completion_tool_result(
                    container=container,
                    params=params,
                    tool_call=tool_call,
                    container_id=container_id,
                )
                for tool_call in tool_calls
            ]
        except Exception:
            await self._delete_container_for_cache_key(sandbox_key)
            raise
        tool_messages = [result[0] for result in tool_results]
        code_interpreter_calls = [result[1] for result in tool_results]

        request_patch = AgenticLoopRequestPatch(
            model=model,
            messages=list(messages) + [self._build_chat_completion_assistant_message(tool_calls)] + tool_messages,
            tools=self._get_followup_tools(
                tools=optional_params.get("tools"),
                call_type=CallTypes.completion,
            ),
            optional_params=self._get_followup_optional_params(optional_params),
            kwargs=self._filter_agentic_loop_kwargs(kwargs),
        )

        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=request_patch,
            metadata={
                "tool_type": "code_interpreter",
                "sandbox_key": sandbox_key or "",
                "code_interpreter_calls": code_interpreter_calls,
                "response_format": "openai",
            },
        )

    async def _build_chat_completion_tool_result(
        self,
        container: object,
        params: dict[str, Any] | None,
        tool_call: CodeExecutionToolCall,
        container_id: str | None,
    ) -> tuple[ChatCompletionToolMessage, CodeInterpreterCall]:
        arguments = tool_call.get("arguments", "")
        code = self._parse_code(arguments)
        stdout = await self._run_tool_call(container=container, params=params, arguments=arguments)
        tool_call_id = tool_call.get("id") or tool_call.get("call_id") or uuid.uuid4().hex
        return (
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": stdout,
            },
            {
                "id": f"ci_{uuid.uuid4().hex}",
                "type": "code_interpreter_call",
                "status": "completed",
                "code": code,
                "container_id": container_id,
                "outputs": [{"type": "logs", "logs": stdout}] if stdout else [],
            },
        )

    async def async_agentic_loop_cleanup_hook(self, plan: AgenticLoopPlan, kwargs: dict) -> None:
        metadata = plan.metadata or {} if plan else {}
        await self._delete_container_for_cache_key(metadata.get("sandbox_key"))

    @staticmethod
    def _filter_agentic_loop_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
        return {
            k: v
            for k, v in kwargs.items()
            if k not in {"litellm_logging_obj", "acompletion"}
            and not is_interception_internal_key(k, prefixes=NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES)
        }

    def _get_followup_tools(self, tools: object, call_type: CallTypes | None) -> list[dict[str, Any]] | None:
        if not isinstance(tools, list):
            return None
        return [
            (
                self._get_function_tool(call_type=call_type)
                if isinstance(tool, dict) and tool.get("type") == "code_interpreter"
                else tool
            )
            for tool in tools
        ]

    def _get_followup_optional_params(self, optional_params: dict[str, object]) -> dict[str, object]:
        drop_tool_choice = self._tool_choice_targets_code_interpreter(optional_params.get("tool_choice"))
        return {
            k: v for k, v in optional_params.items() if k != "tools" and not (k == "tool_choice" and drop_tool_choice)
        }

    async def async_post_agentic_loop_response_hook(self, response: Any, plan: AgenticLoopPlan, kwargs: dict) -> Any:
        metadata = plan.metadata or {} if plan else {}
        await self._delete_container_for_cache_key(metadata.get("sandbox_key"))

        calls = metadata.get("code_interpreter_calls")
        if not calls:
            return response

        is_dict = isinstance(response, dict)
        output = response.get("output") if is_dict else getattr(response, "output", None)
        if not isinstance(output, list):
            return response

        def _item_type(item: Any) -> Any:
            return item.get("type") if isinstance(item, dict) else getattr(item, "type", None)

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

    async def _run_tool_call(self, container: Any, params: dict[str, Any] | None, arguments: str) -> str:
        try:
            code = json.loads(arguments).get("code", "") if arguments else ""
        except (json.JSONDecodeError, TypeError):
            return "[invalid tool arguments: could not parse code]"

        result = await self._run_code(container=container, params=params, code=code)
        if getattr(result, "error", None):
            error = result.error
            message = error.get("value") or error.get("name") if isinstance(error, dict) else str(error)
            return f"[execution error] {message}"
        return getattr(result, "stdout", "") or ""

    async def _get_or_create_container(self, cache_key: str | None) -> tuple[Any, dict[str, Any] | None]:
        if cache_key:
            cached = self._container_cache.get(cache_key)
            if cached is not None:
                return cached[0], cached[1]

        container, params = await self._create_container()
        if cache_key:
            self._container_cache[cache_key] = (container, params, time.time())
        return container, params

    async def _create_container(self) -> tuple[Any, dict[str, Any] | None]:
        if self.sandbox_config is not None:
            return await self.sandbox_config.acreate_sandbox(), None

        params = _resolve_sandbox_tool(self.sandbox_tool_name)
        if params is None:
            raise ValueError(
                "CodeInterpreterInterception: no sandbox available. Provide a "
                "sandbox_config or configure a sandbox tool resolvable via "
                "sandbox_tool_name."
            )
        container = await litellm.acreate_sandbox(
            provider=params["sandbox_provider"],
            api_key=params.get("api_key"),
            api_base=params.get("api_base"),
        )
        return container, params

    async def _run_code(self, container: Any, params: dict[str, Any] | None, code: str) -> Any:
        if self.sandbox_config is not None:
            return await self.sandbox_config.arun_code(container=container, code=code)
        if params is None:
            raise ValueError("CodeInterpreterInterception: no sandbox available to run code.")
        return await litellm.arun_code(
            provider=params["sandbox_provider"],
            container=container,
            code=code,
            api_key=params.get("api_key"),
        )

    async def _delete_container(self, container: Any, params: dict[str, Any] | None) -> None:
        try:
            if self.sandbox_config is not None:
                await self.sandbox_config.adelete_sandbox(container=container)
                return
            if params is None:
                return
            await litellm.adelete_sandbox(
                provider=params["sandbox_provider"],
                container=container,
                api_key=params.get("api_key"),
                api_base=params.get("api_base"),
            )
        except Exception:
            verbose_logger.exception("CodeInterpreterInterception: failed to delete sandbox container")

    async def _delete_container_for_cache_key(self, cache_key: str | None) -> None:
        if not cache_key:
            return
        cached = self._container_cache.pop(cache_key, None)
        if cached is None:
            return
        await self._delete_container(container=cached[0], params=cached[1])

    def _normalize_messages(self, messages: Any) -> list[dict[str, Any]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, list):
            return list(messages)
        return []

    def _extract_code_execution_tool_calls(self, response: object) -> list[CodeExecutionToolCall]:
        if isinstance(response, dict):
            output = response.get("output", [])
        else:
            output = getattr(response, "output", []) or []
        if not isinstance(output, list):
            return []

        return [
            {
                "call_id": (item.get("call_id") if isinstance(item, dict) else getattr(item, "call_id", None)),
                "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                "arguments": (item.get("arguments") if isinstance(item, dict) else getattr(item, "arguments", "")),
            }
            for item in output
            if self._is_code_execution_call(item)
        ]

    def _extract_chat_completion_code_execution_tool_calls(
        self, response: ModelResponse | dict[str, Any]
    ) -> list[CodeExecutionToolCall]:
        model_response = self._to_model_response(response)
        if model_response is None:
            return []
        choices = model_response.choices or []
        if not choices:
            return []
        message = choices[0].message
        tool_calls = message.tool_calls or []

        return [
            normalized
            for tool_call in tool_calls
            if (normalized := self._normalize_chat_completion_tool_call(tool_call)) is not None
        ]

    @staticmethod
    def _normalize_chat_completion_tool_call(
        tool_call: ChatCompletionMessageToolCall,
    ) -> CodeExecutionToolCall | None:
        if tool_call.type != "function" or tool_call.function.name != LITELLM_CODE_EXECUTION_TOOL_NAME:
            return None

        arguments = tool_call.function.arguments
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            arguments = "" if arguments is None else str(arguments)

        return {
            "id": tool_call.id,
            "call_id": tool_call.id,
            "type": "function",
            "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
            "arguments": arguments,
        }

    @staticmethod
    def _build_chat_completion_assistant_message(
        tool_calls: list[CodeExecutionToolCall],
    ) -> ChatCompletionAssistantMessage:
        return {
            "role": "assistant",
            "tool_calls": [
                cast(
                    ChatCompletionAssistantToolCall,
                    {
                        "id": tool_call.get("id"),
                        "type": "function",
                        "function": {
                            "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                            "arguments": tool_call.get("arguments", ""),
                        },
                    },
                )
                for tool_call in tool_calls
            ],
        }

    @staticmethod
    def _to_model_response(
        response: ModelResponse | dict[str, Any],
    ) -> ModelResponse | None:
        if isinstance(response, ModelResponse):
            return response
        try:
            return ModelResponse(**response)
        except (TypeError, ValidationError):
            return None

    def _is_code_execution_call(self, item: Any) -> bool:
        if isinstance(item, dict):
            return item.get("type") == "function_call" and item.get("name") == LITELLM_CODE_EXECUTION_TOOL_NAME
        return (
            getattr(item, "type", None) == "function_call"
            and getattr(item, "name", None) == LITELLM_CODE_EXECUTION_TOOL_NAME
        )

    async def _prune_expired_cache(self) -> None:
        now = time.time()
        expired = [
            (cache_key, container, params)
            for cache_key, (
                container,
                params,
                created_at,
            ) in self._container_cache.items()
            if now - created_at > _CACHE_TTL_SECONDS
        ]
        for cache_key, container, params in expired:
            self._container_cache.pop(cache_key, None)
            await self._delete_container(container=container, params=params)
