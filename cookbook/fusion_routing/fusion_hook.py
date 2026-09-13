from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Protocol, Union

from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import get_session_id_from_request_data
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

FusionRole = Literal["main", "sidekick"]


@dataclass(frozen=True, slots=True)
class FusionModelGroup:
    main_model: str
    sidekick_model: str

    def model_for_role(self, role: FusionRole) -> str:
        return self.main_model if role == "main" else self.sidekick_model


class FusionClassifier(Protocol):
    async def classify(self, data: Dict[str, Any], session_id: str) -> FusionRole: ...


def _last_tool_call_name(data: Dict[str, Any]) -> Optional[str]:
    messages: List[Dict[str, Any]] = data.get("messages") or []
    for message in reversed(messages):
        tool_calls: List[Dict[str, Any]] = message.get("tool_calls") or []
        for tool_call in tool_calls:
            return tool_call.get("function", {}).get("name")
    return None


class ToolNameFusionClassifier:
    def __init__(self, sidekick_tools: FrozenSet[str]) -> None:
        self._sidekick_tools = sidekick_tools

    async def classify(self, data: Dict[str, Any], session_id: str) -> FusionRole:
        tool_name = _last_tool_call_name(data)
        return "sidekick" if tool_name in self._sidekick_tools else "main"


class FusionRoutingHook(CustomLogger):
    def __init__(self, group: FusionModelGroup, classifier: FusionClassifier) -> None:
        super().__init__()
        self._group = group
        self._classifier = classifier

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Dict[str, Any],
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, Dict[str, Any]]]:
        session_id = get_session_id_from_request_data(data)
        if session_id is None:
            return None

        role = await self._classifier.classify(data, session_id)
        metadata: Dict[str, Any] = {
            **(data.get("metadata") or {}),
            "fusion_role": role,
            "fusion_original_model": data["model"],
        }
        return {**data, "model": self._group.model_for_role(role), "metadata": metadata}


fusion_hook_instance = FusionRoutingHook(
    group=FusionModelGroup(main_model="fusion-main", sidekick_model="fusion-sidekick"),
    classifier=ToolNameFusionClassifier(sidekick_tools=frozenset({"read_file", "grep", "run_tests", "apply_diff"})),
)
