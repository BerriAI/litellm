"""
Code Interpreter Interception Handler

CustomLogger that swaps the native OpenAI Responses ``code_interpreter`` tool for
a function tool, executes the code the model emits inside a sandbox, and feeds the
captured stdout back through the typed agentic loop plan.
"""

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeAlias, TypedDict, runtime_checkable

from pydantic import ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.base_llm.sandbox.transformation import (
    CodeExecutionResult,
    ContainerHandle,
)
from litellm.types.integrations.code_interpreter_interception import (
    CodeInterpreterInterceptionConfig,
)
from litellm.types.integrations.custom_logger import (
    CHAT_COMPLETION_AGENTIC_SURFACE,
    NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES,
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
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

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

LITELLM_CODE_EXECUTION_TOOL_NAME: Final = "litellm_code_execution"
_INTERCEPTION_ACTIVE_KEY: Final = "_code_interpreter_interception_active"
_SANDBOX_KEY: Final = "_code_interpreter_interception_sandbox_key"
_SESSION_SCOPED_KEY: Final = "_code_interpreter_interception_session_scoped"
_CONVERTED_STREAM_KEY: Final = "_code_interpreter_interception_converted_stream"
_LITELLM_METADATA_KEY: Final = "litellm_metadata"
_CACHE_TTL_SECONDS: Final = 15 * 60
_SESSION_SCOPED_PER_IDENTITY_CAP: Final = 10


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


class SandboxToolParams(TypedDict):
    sandbox_provider: str
    api_key: str | None
    api_base: str | None


class SandboxConfigProtocol(Protocol):
    async def acreate_sandbox(self) -> ContainerHandle: ...

    async def arun_code(self, *, container: ContainerHandle, code: str) -> CodeExecutionResult: ...

    async def adelete_sandbox(self, *, container: ContainerHandle) -> object: ...


@runtime_checkable
class _SupportsOutput(Protocol):
    output: object


_CachedContainer: TypeAlias = tuple[ContainerHandle, SandboxToolParams | None, float, str | None]


def _output_item_type(item: object) -> object:
    if isinstance(item, dict):
        item_mapping: Final[dict[str, object]] = item
        return item_mapping.get("type")
    return getattr(item, "type", None)


def _response_output(response: object) -> object:
    if isinstance(response, dict):
        response_mapping: Final[Mapping[str, object]] = response
        return response_mapping.get("output", [])
    return getattr(response, "output", []) or []


def _tool_call_arguments(arguments: object) -> str:
    if isinstance(arguments, str):
        return arguments
    return "" if arguments is None else str(arguments)


def _narrow_tool_call(tool_call: Mapping[str, object]) -> CodeExecutionToolCall:
    tool_call_id: Final = tool_call.get("id")
    call_id: Final = tool_call.get("call_id")
    return {
        "id": tool_call_id if isinstance(tool_call_id, str) else None,
        "call_id": call_id if isinstance(call_id, str) else None,
        "type": "function",
        "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
        "arguments": _tool_call_arguments(tool_call.get("arguments")),
    }


def _extract_session_id(kwargs: dict[str, object]) -> str | None:
    for meta_key in ("metadata", "litellm_metadata"):
        meta = kwargs.get(meta_key)
        if isinstance(meta, dict):
            metadata: dict[str, object] = meta
            sid = metadata.get("session_id")
            if sid and isinstance(sid, str):
                return sid
    return None


def _extract_identity(kwargs: Mapping[str, object]) -> str:
    identity: Final = kwargs.get("user_api_key_hash")
    return identity if isinstance(identity, str) else ""


def _resolve_sandbox_tool(sandbox_tool_name: str | None) -> SandboxToolParams | None:
    if sandbox_tool_name is None:
        return None
    try:
        from litellm.sandbox.sandbox_tools import resolve_sandbox_tool
    except ImportError:
        return None
    resolved: Final[dict[str, object] | None] = resolve_sandbox_tool(sandbox_tool_name)
    if resolved is None:
        return None
    provider: Final = resolved.get("sandbox_provider")
    api_key: Final = resolved.get("api_key")
    api_base: Final = resolved.get("api_base")
    return SandboxToolParams(
        sandbox_provider=provider if isinstance(provider, str) else "",
        api_key=api_key if isinstance(api_key, str) else None,
        api_base=api_base if isinstance(api_base, str) else None,
    )


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
        sandbox_config: SandboxConfigProtocol | None = None,
    ):
        super().__init__()
        self.enabled = enabled
        self.enabled_providers = enabled_providers
        self.sandbox_tool_name = sandbox_tool_name
        self.sandbox_config = sandbox_config
        self._container_cache: dict[str, _CachedContainer] = {}

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
        params: Final[CodeInterpreterInterceptionConfig] = (
            litellm_settings["code_interpreter_interception_params"]
            if "code_interpreter_interception_params" in litellm_settings
            else callback_specific_params["code_interpreter_interception"]
            if isinstance(callback_specific_params.get("code_interpreter_interception"), dict)
            else {}
        )
        return CodeInterpreterInterceptionLogger.from_config_yaml(params)

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, object], call_type: CallTypes | None
    ) -> dict | None:
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

        tools: Final = kwargs.get("tools")
        if not isinstance(tools, list):
            return None
        if not any(isinstance(tool, dict) and tool.get("type") == "code_interpreter" for tool in tools):
            return None

        kwargs[_INTERCEPTION_ACTIVE_KEY] = True
        session_id: Final = _extract_session_id(kwargs)
        if session_id:
            identity: Final = _extract_identity(kwargs)
            kwargs[_SANDBOX_KEY] = f"{identity}:{session_id}" if identity else session_id
            kwargs[_SESSION_SCOPED_KEY] = True
        else:
            kwargs[_SANDBOX_KEY] = uuid.uuid4().hex
        if kwargs.get("stream"):
            kwargs["stream"] = False
            kwargs[_CONVERTED_STREAM_KEY] = True
        self._write_interception_metadata(kwargs)

        function_tool: Final = self._get_function_tool(call_type=call_type)
        kwargs["tools"] = [
            (function_tool if isinstance(tool, dict) and tool.get("type") == "code_interpreter" else tool)
            for tool in tools
        ]
        if self._tool_choice_targets_code_interpreter(kwargs.get("tool_choice")):
            kwargs["tool_choice"] = self._get_function_tool_choice(call_type=call_type)
        return kwargs

    @staticmethod
    def _strip_interception_metadata(kwargs: dict[str, object]) -> None:
        metadata: Final = kwargs.get(_LITELLM_METADATA_KEY)
        if not isinstance(metadata, dict):
            return
        current_metadata: Final[dict[str, object]] = metadata
        filtered_metadata: Final = {
            key: value
            for key, value in current_metadata.items()
            if not is_interception_internal_key(key)
            and not key.startswith("_agentic_loop")
            and key != "max_agentic_loops"
            and key != _SESSION_SCOPED_KEY
        }
        if filtered_metadata:
            kwargs[_LITELLM_METADATA_KEY] = filtered_metadata
        else:
            kwargs.pop(_LITELLM_METADATA_KEY, None)

    @staticmethod
    def _write_interception_metadata(kwargs: dict[str, object]) -> None:
        existing: Final = kwargs.get(_LITELLM_METADATA_KEY)
        metadata: Final[dict[str, object]] = dict(existing) if isinstance(existing, dict) else {}
        for key in (_INTERCEPTION_ACTIVE_KEY, _SANDBOX_KEY, _SESSION_SCOPED_KEY, _CONVERTED_STREAM_KEY):
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
        description: Final = "Execute python code in a sandbox and return stdout."
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
    def _tool_choice_targets_code_interpreter(tool_choice: object) -> bool:
        if not isinstance(tool_choice, dict):
            return False
        choice: Final[dict[str, object]] = tool_choice
        function: Final = choice.get("function")
        return (
            choice.get("type") == "code_interpreter"
            or choice.get("name") == "code_interpreter"
            or choice.get("name") == LITELLM_CODE_EXECUTION_TOOL_NAME
            or (isinstance(function, dict) and function.get("name") == LITELLM_CODE_EXECUTION_TOOL_NAME)
        )

    def _resolve_provider(self, kwargs: dict[str, object]) -> str | None:
        provider: Final = kwargs.get("custom_llm_provider")
        if isinstance(provider, str) and provider:
            return provider
        model: Final = kwargs.get("model")
        if not isinstance(model, str):
            return None
        try:
            return litellm.get_llm_provider(model=model)[1]
        except Exception:
            return None

    async def async_should_run_agentic_loop(
        self,
        response: object,
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

        tool_calls: Final = (
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
        response: object,
        anthropic_messages_provider_config: object,
        anthropic_messages_optional_request_params: dict[str, object],
        logging_obj: "LiteLLMLoggingObj",
        stream: bool,
        kwargs: dict[str, object],
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
        tool_calls: Final = self._agentic_tool_calls(tools)
        sandbox_key: Final = self._extract_sandbox_key(kwargs)
        is_session: Final = bool(kwargs.get(_SESSION_SCOPED_KEY))
        identity: Final = _extract_identity(kwargs) if is_session else None
        container, params = await self._get_or_create_container(cache_key=sandbox_key, identity=identity)

        try:
            container_id: Final = self._container_id(container)
            input_list: Final = self._normalize_messages(messages)
            code_interpreter_calls: Final[list[CodeInterpreterCall]] = []
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

        optional_params: Final = anthropic_messages_optional_request_params
        request_patch: Final = AgenticLoopRequestPatch(
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
                "is_session_scoped": bool(kwargs.get(_SESSION_SCOPED_KEY)),
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
        tool_calls: Final = self._agentic_tool_calls(tools)
        sandbox_key: Final = self._extract_sandbox_key(kwargs)
        is_session: Final = bool(kwargs.get(_SESSION_SCOPED_KEY))
        identity: Final = _extract_identity(kwargs) if is_session else None
        container, params = await self._get_or_create_container(cache_key=sandbox_key, identity=identity)

        try:
            container_id: Final = self._container_id(container)
            tool_results: Final = [
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
        tool_messages: Final = [result[0] for result in tool_results]
        code_interpreter_calls: Final = [result[1] for result in tool_results]

        request_patch: Final = AgenticLoopRequestPatch(
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
                "is_session_scoped": bool(kwargs.get(_SESSION_SCOPED_KEY)),
                "code_interpreter_calls": code_interpreter_calls,
                "response_format": "openai",
            },
        )

    @staticmethod
    def _container_id(container: ContainerHandle) -> str | None:
        container_id: Final[object] = getattr(container, "id", None)
        return container_id if isinstance(container_id, str) else None

    @staticmethod
    def _agentic_tool_calls(tools: dict[str, object]) -> list[CodeExecutionToolCall]:
        tool_calls: Final = tools.get("tool_calls")
        if not isinstance(tool_calls, list):
            return []
        items: Final[list[object]] = tool_calls
        return [_narrow_tool_call(item) for item in items if isinstance(item, dict)]

    @staticmethod
    def _extract_sandbox_key(kwargs: dict[str, object]) -> str | None:
        sandbox_key: Final = kwargs.get(_SANDBOX_KEY)
        return sandbox_key if isinstance(sandbox_key, str) else None

    async def _build_chat_completion_tool_result(
        self,
        container: ContainerHandle,
        params: SandboxToolParams | None,
        tool_call: CodeExecutionToolCall,
        container_id: str | None,
    ) -> tuple[ChatCompletionToolMessage, CodeInterpreterCall]:
        arguments: Final = tool_call.get("arguments", "")
        code: Final = self._parse_code(arguments)
        stdout: Final = await self._run_tool_call(container=container, params=params, arguments=arguments)
        tool_call_id: Final = tool_call.get("id") or tool_call.get("call_id") or uuid.uuid4().hex
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
        metadata: Final[dict[str, object]] = plan.metadata or {} if plan else {}
        if metadata.get("is_session_scoped"):
            return
        await self._delete_container_for_cache_key(self._metadata_sandbox_key(metadata))

    @staticmethod
    def _metadata_sandbox_key(metadata: Mapping[str, object]) -> str | None:
        sandbox_key: Final = metadata.get("sandbox_key")
        return sandbox_key if isinstance(sandbox_key, str) else None

    @staticmethod
    def _filter_agentic_loop_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
        return {
            k: v
            for k, v in kwargs.items()
            if k not in {"litellm_logging_obj", "acompletion"}
            and not is_interception_internal_key(k, prefixes=NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES)
        }

    def _get_followup_tools(self, tools: object, call_type: CallTypes | None) -> list[dict[str, object]] | None:
        if not isinstance(tools, list):
            return None
        return [
            (
                dict(self._get_function_tool(call_type=call_type))
                if isinstance(tool, dict) and tool.get("type") == "code_interpreter"
                else tool
            )
            for tool in tools
        ]

    def _get_followup_optional_params(self, optional_params: dict[str, object]) -> dict[str, object]:
        drop_tool_choice: Final = self._tool_choice_targets_code_interpreter(optional_params.get("tool_choice"))
        return {
            k: v for k, v in optional_params.items() if k != "tools" and not (k == "tool_choice" and drop_tool_choice)
        }

    async def async_post_agentic_loop_response_hook(
        self, response: object, plan: AgenticLoopPlan, kwargs: dict
    ) -> object:
        metadata: Final[dict[str, object]] = plan.metadata or {} if plan else {}
        if not metadata.get("is_session_scoped"):
            await self._delete_container_for_cache_key(self._metadata_sandbox_key(metadata))

        calls: Final = metadata.get("code_interpreter_calls")
        if not calls or not isinstance(calls, list):
            return response

        if isinstance(response, dict):
            response_mapping: Final[dict[str, object]] = response
            merged_mapping_output: Final = self._merge_code_interpreter_calls(response_mapping.get("output"), calls)
            if merged_mapping_output is not None:
                response_mapping["output"] = merged_mapping_output
            return response

        if not isinstance(response, _SupportsOutput):
            return response
        merged_attr_output: Final = self._merge_code_interpreter_calls(response.output, calls)
        if merged_attr_output is not None:
            response.output = merged_attr_output
        return response

    @staticmethod
    def _merge_code_interpreter_calls(output: object, calls: Sequence[object]) -> list[object] | None:
        if not isinstance(output, list):
            return None
        items: Final[list[object]] = output
        insert_at: Final = next(
            (i for i, item in enumerate(items) if _output_item_type(item) == "message"),
            len(items),
        )
        return items[:insert_at] + list(calls) + items[insert_at:]

    @staticmethod
    def _parse_code(arguments: str) -> str:
        try:
            return json.loads(arguments).get("code", "") if arguments else ""
        except (json.JSONDecodeError, TypeError, AttributeError):
            return ""

    async def _run_tool_call(self, container: ContainerHandle, params: SandboxToolParams | None, arguments: str) -> str:
        try:
            code: Final = json.loads(arguments).get("code", "") if arguments else ""
        except (json.JSONDecodeError, TypeError):
            return "[invalid tool arguments: could not parse code]"

        result: Final = await self._run_code(container=container, params=params, code=code)
        if getattr(result, "error", None):
            error: Final = result.error
            message: Final = error.get("value") or error.get("name") if isinstance(error, dict) else str(error)
            return f"[execution error] {message}"
        return getattr(result, "stdout", "") or ""

    async def _get_or_create_container(
        self,
        cache_key: str | None,
        identity: str | None = None,
    ) -> tuple[ContainerHandle, SandboxToolParams | None]:
        if cache_key:
            cached: Final = self._container_cache.get(cache_key)
            if cached is not None:
                self._container_cache[cache_key] = (cached[0], cached[1], time.time(), cached[3])
                return cached[0], cached[1]

        container, params = await self._create_container()
        if cache_key:
            if identity is not None:
                await self._evict_lru_session_if_over_cap(identity)
            self._container_cache[cache_key] = (container, params, time.time(), identity)
        return container, params

    async def _evict_lru_session_if_over_cap(self, identity: str) -> None:
        identity_entries: Final = [(k, v) for k, v in self._container_cache.items() if v[3] == identity]
        if len(identity_entries) < _SESSION_SCOPED_PER_IDENTITY_CAP:
            return
        lru_key, lru_entry = min(identity_entries, key=lambda item: item[1][2])
        self._container_cache.pop(lru_key, None)
        await self._delete_container(container=lru_entry[0], params=lru_entry[1])

    async def _create_container(self) -> tuple[ContainerHandle, SandboxToolParams | None]:
        if self.sandbox_config is not None:
            return await self.sandbox_config.acreate_sandbox(), None

        params: Final = _resolve_sandbox_tool(self.sandbox_tool_name)
        if params is None:
            raise ValueError(
                "CodeInterpreterInterception: no sandbox available. Provide a "
                "sandbox_config or configure a sandbox tool resolvable via "
                "sandbox_tool_name."
            )
        container: Final = await litellm.acreate_sandbox(
            provider=params["sandbox_provider"],
            api_key=params.get("api_key"),
            api_base=params.get("api_base"),
        )
        return container, params

    async def _run_code(
        self, container: ContainerHandle, params: SandboxToolParams | None, code: str
    ) -> CodeExecutionResult:
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

    async def _delete_container(self, container: ContainerHandle, params: SandboxToolParams | None) -> None:
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
        cached: Final = self._container_cache.pop(cache_key, None)
        if cached is None:
            return
        await self._delete_container(container=cached[0], params=cached[1])

    def _normalize_messages(self, messages: object) -> list[dict[str, object]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, list):
            return list(messages)
        return []

    def _extract_code_execution_tool_calls(self, response: object) -> list[CodeExecutionToolCall]:
        output: Final = _response_output(response)
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

    def _extract_chat_completion_code_execution_tool_calls(self, response: object) -> list[CodeExecutionToolCall]:
        model_response: Final = self._to_model_response(response)
        if model_response is None:
            return []
        choices: Final = model_response.choices or []
        if not choices:
            return []
        message: Final = choices[0].message
        tool_calls: Final = message.tool_calls or []

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
        tool_calls: Sequence[CodeExecutionToolCall],
    ) -> ChatCompletionAssistantMessage:
        assistant_tool_calls: Final[list[ChatCompletionAssistantToolCall]] = [
            {
                "id": tool_call.get("id"),
                "type": "function",
                "function": {
                    "name": LITELLM_CODE_EXECUTION_TOOL_NAME,
                    "arguments": tool_call.get("arguments", ""),
                },
            }
            for tool_call in tool_calls
        ]
        return {
            "role": "assistant",
            "tool_calls": assistant_tool_calls,
        }

    @staticmethod
    def _to_model_response(response: object) -> ModelResponse | None:
        if isinstance(response, ModelResponse):
            return response
        if not isinstance(response, dict):
            return None
        response_fields: Final[dict[str, object]] = response
        try:
            return ModelResponse(**response_fields)
        except (TypeError, ValidationError):
            return None

    def _is_code_execution_call(self, item: object) -> bool:
        if isinstance(item, dict):
            item_mapping: Final[dict[str, object]] = item
            return (
                item_mapping.get("type") == "function_call"
                and item_mapping.get("name") == LITELLM_CODE_EXECUTION_TOOL_NAME
            )
        item_type: Final[object] = getattr(item, "type", None)
        item_name: Final[object] = getattr(item, "name", None)
        return item_type == "function_call" and item_name == LITELLM_CODE_EXECUTION_TOOL_NAME

    async def _prune_expired_cache(self) -> None:
        now: Final = time.time()
        expired: Final = [
            (cache_key, container, params)
            for cache_key, (container, params, last_accessed, *_) in self._container_cache.items()
            if now - last_accessed > _CACHE_TTL_SECONDS
        ]
        for cache_key, container, params in expired:
            self._container_cache.pop(cache_key, None)
            await self._delete_container(container=container, params=params)
