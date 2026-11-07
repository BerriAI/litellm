import json
import uuid
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from fastapi import HTTPException
from mcp.types import TextContent
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.obsidian_policy.client import (
    ObsidianClient,
    PolicyDecision,
    Stage,
    ToolEvent,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.obsidian_policy import ObsidianPolicyGuardrailConfigModel
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.types.mcp import MCPPostCallResponseObject

_EventHook: TypeAlias = GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None
_OBJECT_MAPPING: Final = TypeAdapter(Mapping[str, object])


class _ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str = ""
    arguments: Mapping[str, object] = MappingProxyType({})
    mcp_server_name: str | None = None
    mcp_session_id: str | None = None


class _CallerMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    user_api_key_user_id: str | None = None
    user_api_key_user_email: str | None = None


class _CallRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    litellm_call_id: str | None = None


def _call_id(logging_obj: object) -> str | None:
    try:
        return _CallRef.model_validate(logging_obj, from_attributes=True).litellm_call_id
    except ValidationError:
        return None


def _mapping(value: object) -> Mapping[str, object]:
    try:
        return _OBJECT_MAPPING.validate_python(value)
    except ValidationError:
        return MappingProxyType({})


def _denial_detail(guardrail_name: str | None, stage: Stage, decision: PolicyDecision) -> Mapping[str, object]:
    return {
        "error": "Blocked by Obsidian policy",
        "guardrail": guardrail_name,
        "stage": stage,
        "message": decision.message,
        "evaluation_id": decision.evaluation_id,
        "matched_policies": [policy.model_dump(exclude_none=True) for policy in decision.matched_policies],
    }


class ObsidianPolicyGuardrail(CustomGuardrail):
    def __init__(
        self,
        client: ObsidianClient,
        guardrail_name: str,
        event_hook: _EventHook = None,
        default_on: bool = False,
    ) -> None:
        super().__init__(
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
        user_api_key_dict: UserAPIKeyAuth,
        cache: "DualCache",
        data: dict[str, object],  # mutable-ok: overrides CustomLogger.async_pre_call_hook
        call_type: str,
    ) -> Exception | str | dict[str, object] | None:  # mutable-ok: overrides CustomLogger.async_pre_call_hook
        if call_type != CallTypes.call_mcp_tool.value:
            return data
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_mcp_call) is not True:
            return data

        call_id: Final = _call_id(data.get("litellm_logging_obj"))
        decision: Final = await self._client.evaluate(
            ToolEvent(
                stage="PreToolUse",
                server_name=str(data.get("mcp_server_name") or ""),
                tool_name=str(data.get("mcp_tool_name") or ""),
                arguments=_mapping(data.get("mcp_arguments")),
                session_id=str(call_id or uuid.uuid4()),
                user_id=user_api_key_dict.user_id,
                user_email=user_api_key_dict.user_email,
            )
        )
        if decision.allowed:
            return data
        raise HTTPException(status_code=403, detail=_denial_detail(self.guardrail_name, "PreToolUse", decision))

    async def async_post_mcp_tool_call_hook(
        self,
        kwargs: dict[str, object],  # mutable-ok: overrides CustomLogger.async_post_mcp_tool_call_hook
        response_obj: "MCPPostCallResponseObject",
        start_time: datetime,
        end_time: datetime,
    ) -> "MCPPostCallResponseObject | None":
        litellm_params: Final = _mapping(kwargs.get("litellm_params"))
        metadata: Final = _mapping(litellm_params.get("metadata"))
        request_data: Final = {
            "metadata": metadata,
            **({"guardrails": kwargs["guardrails"]} if "guardrails" in kwargs else {}),
        }
        if self.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.post_mcp_call) is not True:
            return None

        tool_call: Final = _ToolCall.model_validate(_mapping(kwargs.get("mcp_tool_call_metadata")))
        caller: Final = _CallerMetadata.model_validate(metadata)
        decision: Final = await self._client.evaluate(
            ToolEvent(
                stage="PostToolUse",
                server_name=tool_call.mcp_server_name or "",
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                session_id=tool_call.mcp_session_id or str(kwargs.get("litellm_call_id") or uuid.uuid4()),
                user_id=caller.user_api_key_user_id,
                user_email=caller.user_api_key_user_email,
                tool_response=_tool_response_payload(response_obj),
            )
        )
        if decision.allowed:
            return None
        return response_obj.model_copy(
            update={
                "mcp_tool_call_response": [
                    TextContent(
                        type="text",
                        text=json.dumps(_denial_detail(self.guardrail_name, "PostToolUse", decision)),
                    )
                ]
            }
        )


def _tool_response_payload(response_obj: "MCPPostCallResponseObject") -> Mapping[str, object]:
    return {
        "content": [
            item.model_dump(mode="json", exclude_none=True, by_alias=True)
            for item in response_obj.mcp_tool_call_response
        ]
    }
