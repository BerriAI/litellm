import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from fastapi import HTTPException
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # decorator is untyped in custom_guardrail
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.obsidian_policy.client import (
    ObsidianClient,
    PolicyDecision,
    Stage,
    ToolEvent,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.llms.base import HiddenParams
from litellm.types.mcp import MCPPostCallResponseObject
from litellm.types.proxy.guardrails.guardrail_hooks.obsidian_policy import ObsidianPolicyGuardrailConfigModel
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache

_EventHook: TypeAlias = GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None
_OBJECT_MAPPING: Final = TypeAdapter(Mapping[str, object])
_OBJECTS: Final = TypeAdapter(Sequence[object])
_BLOB_KEYS: Final = frozenset({"data", "blob"})


class _LoggingRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    litellm_call_id: str | None = None
    model_call_details: Mapping[str, object] = Field(default_factory=dict)


class _ToolCall(BaseModel):
    """The MCP tool call as LiteLLM records it in ``mcp_tool_call_metadata``, read by both stages."""

    model_config = ConfigDict(frozen=True)

    name: str = ""
    arguments: Mapping[str, object] = Field(default_factory=dict)
    mcp_server_name: str = ""
    namespaced_tool_name: str = ""
    mcp_session_id: str = ""

    def server_name(self, fallback: str) -> str:
        prefix, slash, _ = self.namespaced_tool_name.partition("/")
        return self.mcp_server_name or (prefix if slash else "") or fallback


class _CallerMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    user_api_key_user_id: str | None = None
    user_api_key_user_email: str | None = None


def _mapping(value: object) -> Mapping[str, object]:
    try:
        return _OBJECT_MAPPING.validate_python(value)
    except ValidationError:
        return MappingProxyType({})


def _objects(value: object) -> Sequence[object]:
    try:
        return _OBJECTS.validate_python(value) if isinstance(value, (list, tuple)) else ()
    except ValidationError:
        return ()


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _logging_ref(logging_obj: object) -> _LoggingRef:
    try:
        return _LoggingRef.model_validate(logging_obj, from_attributes=True)
    except ValidationError:
        return _LoggingRef()


def _tool_call(recorded: Mapping[str, object]) -> _ToolCall:
    return _ToolCall(
        name=_text(recorded.get("name")),
        arguments=_mapping(recorded.get("arguments")),
        mcp_server_name=_text(recorded.get("mcp_server_name")),
        namespaced_tool_name=_text(recorded.get("namespaced_tool_name")),
        mcp_session_id=_text(recorded.get("mcp_session_id")),
    )


def _requested_guardrails(kwargs: Mapping[str, object], metadata: Mapping[str, object]) -> Sequence[object]:
    """Merge the request, key and team guardrail selections the way the pre-call path does."""
    sources: Final = (
        kwargs.get("guardrails"),
        metadata.get("guardrails"),
        _mapping(kwargs.get("litellm_metadata")).get("guardrails"),
        _mapping(metadata.get("user_api_key_metadata")).get("guardrails"),
        _mapping(metadata.get("user_api_key_team_metadata")).get("guardrails"),
    )
    merged: Final = tuple(chain.from_iterable(_objects(source) for source in sources))
    return tuple(item for index, item in enumerate(merged) if item not in merged[:index])


def _without_blobs(value: object) -> object:
    """Drop base64 payloads (images, audio, resource blobs) and keep their size, so policies see text only."""
    original: Final = value
    if isinstance(value, Mapping):
        try:
            mapping: Final = _OBJECT_MAPPING.validate_python(value)
        except ValidationError:
            return original
        kept: Final = {key: _without_blobs(item) for key, item in mapping.items() if key not in _BLOB_KEYS}
        sizes: Final = {
            f"{key}_bytes": len(item) for key, item in mapping.items() if key in _BLOB_KEYS and isinstance(item, str)
        }
        return {**kept, **sizes}
    if isinstance(value, list):
        return [_without_blobs(item) for item in _OBJECTS.validate_python(value)]
    return value


def _tool_response_payload(response_obj: MCPPostCallResponseObject) -> Mapping[str, object]:
    return {
        "content": [
            _without_blobs(item.model_dump(mode="json", exclude_none=True, by_alias=True))
            for item in response_obj.mcp_tool_call_response
        ]
    }


def _denial_summary(decision: PolicyDecision) -> str:
    if decision.via_fail_mode:
        return f"Obsidian policy check unavailable: {decision.message}"
    reason: Final = f": {decision.message}" if decision.message else ""
    reference: Final = f" (evaluation_id {decision.evaluation_id})" if decision.evaluation_id else ""
    return f"Blocked by Obsidian policy{reason}{reference}"


def _denial_detail(guardrail_name: str | None, stage: Stage, decision: PolicyDecision) -> Mapping[str, object]:
    return {
        "error": _denial_summary(decision),
        "guardrail": guardrail_name,
        "stage": stage,
        "message": decision.message,
        "evaluation_id": decision.evaluation_id,
        "matched_policies": [policy.model_dump(exclude_none=True) for policy in decision.matched_policies],
    }


def _mark_tool_error(original_response: object) -> None:
    """The dispatcher only flags an error itself when structured output existed, so flag the real result here."""
    if not isinstance(original_response, CallToolResult):
        return
    try:
        original_response.is_error = True  # rebind-ok: the dispatcher reads is_error off the caller's CallToolResult
    except (AttributeError, TypeError, ValueError):
        verbose_proxy_logger.debug("Obsidian policy: could not mark the MCP result as an error")


class ObsidianPolicyGuardrail(CustomGuardrail):
    def __init__(
        self,
        client: ObsidianClient,
        guardrail_name: str,
        event_hook: _EventHook = None,
        default_on: bool = False,
    ) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.__init__ takes untyped **kwargs
            guardrail_name=guardrail_name,
            supported_event_hooks=self.get_supported_event_hooks(),
            event_hook=event_hook,
            default_on=default_on,
        )
        self._client: Final = client

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: overrides CustomGuardrail
        return [GuardrailEventHooks.pre_mcp_call, GuardrailEventHooks.post_mcp_call]

    @staticmethod
    def get_config_model() -> type[ObsidianPolicyGuardrailConfigModel]:
        return ObsidianPolicyGuardrailConfigModel

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth | None,
        cache: "DualCache",
        data: dict[str, object],  # mutable-ok: overrides CustomLogger.async_pre_call_hook
        call_type: str,
    ) -> Exception | str | dict[str, object] | None:  # mutable-ok: overrides CustomLogger.async_pre_call_hook
        if call_type != CallTypes.call_mcp_tool.value:
            return data
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_mcp_call) is not True:  # pyright: ignore[reportUnknownMemberType]  # data is untyped in CustomGuardrail
            return data

        logging_ref: Final = _logging_ref(data.get("litellm_logging_obj"))
        recorded: Final = _tool_call(_mapping(logging_ref.model_call_details.get("mcp_tool_call_metadata")))
        decision: Final = await self._client.evaluate(
            ToolEvent(
                stage="PreToolUse",
                server_name=recorded.server_name(_text(data.get("mcp_server_name"))),
                tool_name=recorded.name or _text(data.get("mcp_tool_name")),
                arguments=_mapping(data.get("mcp_arguments")) or recorded.arguments,
                session_id=recorded.mcp_session_id or logging_ref.litellm_call_id or str(uuid.uuid4()),
                user_id=user_api_key_dict.user_id
                if user_api_key_dict
                else _text(data.get("user_api_key_user_id")) or None,
                user_email=user_api_key_dict.user_email if user_api_key_dict else None,
            )
        )
        if decision.allowed:
            return data
        raise HTTPException(status_code=403, detail=_denial_detail(self.guardrail_name, "PreToolUse", decision))

    async def async_post_mcp_tool_call_hook(
        self,
        kwargs: dict[str, object],  # mutable-ok: overrides CustomLogger.async_post_mcp_tool_call_hook
        response_obj: MCPPostCallResponseObject,
        start_time: datetime,
        end_time: datetime,
    ) -> MCPPostCallResponseObject | None:
        # The dispatcher treats an exception here as "no change" and forwards the original
        # tool output, so a crash must be turned into a fail-mode decision instead.
        try:
            decision: Final = await self._post_decision(kwargs, response_obj)
        except Exception as exc:  # noqa: BLE001  # the dispatcher contract makes raising equivalent to allowing
            verbose_proxy_logger.exception("Obsidian policy: PostToolUse evaluation raised")
            failure: Final = self._client.failure("error", f"guardrail raised {type(exc).__name__}")
            return None if failure.allowed else self._deny_post(kwargs, response_obj, failure)
        if decision is None or decision.allowed:
            return None
        return self._deny_post(kwargs, response_obj, decision)

    async def _post_decision(
        self, kwargs: Mapping[str, object], response_obj: MCPPostCallResponseObject
    ) -> PolicyDecision | None:
        metadata: Final = _mapping(_mapping(kwargs.get("litellm_params")).get("metadata"))
        request_data: Final = {"metadata": metadata, "guardrails": list(_requested_guardrails(kwargs, metadata))}
        if self.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.post_mcp_call) is not True:  # pyright: ignore[reportUnknownMemberType]  # data is untyped in CustomGuardrail
            return None

        recorded: Final = _tool_call(_mapping(kwargs.get("mcp_tool_call_metadata")))
        caller: Final = _CallerMetadata.model_validate(metadata)
        return await self._client.evaluate(
            ToolEvent(
                stage="PostToolUse",
                server_name=recorded.server_name(""),
                tool_name=recorded.name,
                arguments=recorded.arguments,
                session_id=recorded.mcp_session_id or _text(kwargs.get("litellm_call_id")) or str(uuid.uuid4()),
                user_id=caller.user_api_key_user_id,
                user_email=caller.user_api_key_user_email,
                tool_response=_tool_response_payload(response_obj),
            )
        )

    def _deny_post(
        self, kwargs: Mapping[str, object], response_obj: object, decision: PolicyDecision
    ) -> MCPPostCallResponseObject:
        _mark_tool_error(kwargs.get("original_response"))
        hidden: Final = getattr(response_obj, "hidden_params", None)
        return MCPPostCallResponseObject(
            mcp_tool_call_response=[
                TextContent(type="text", text=json.dumps(_denial_detail(self.guardrail_name, "PostToolUse", decision)))
            ],
            hidden_params=hidden if isinstance(hidden, HiddenParams) else HiddenParams(),
        )
