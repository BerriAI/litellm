from typing import Any, Dict, FrozenSet, Optional, cast

import pytest

from cookbook.fusion_routing.fusion_hook import (
    FusionModelGroup,
    FusionRole,
    FusionRoutingHook,
    ToolNameFusionClassifier,
)
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth

USER_API_KEY_DICT = UserAPIKeyAuth()
CACHE = DualCache()


class StubClassifier:
    def __init__(self, role: FusionRole) -> None:
        self._role = role

    async def classify(self, data: Dict[str, Any], session_id: str) -> FusionRole:
        return self._role


def _make_hook(role: FusionRole) -> FusionRoutingHook:
    group = FusionModelGroup(main_model="fusion-main", sidekick_model="fusion-sidekick")
    return FusionRoutingHook(group=group, classifier=StubClassifier(role))


async def _route(hook: FusionRoutingHook, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    result = await hook.async_pre_call_hook(USER_API_KEY_DICT, CACHE, data, "completion")
    return cast(Optional[Dict[str, Any]], result)


async def test_routes_to_sidekick_model():
    hook = _make_hook("sidekick")
    data = {"model": "fusion-main", "litellm_session_id": "task-1", "messages": []}

    result = await _route(hook, data)

    assert result is not None
    assert result["model"] == "fusion-sidekick"
    assert result["metadata"]["fusion_role"] == "sidekick"
    assert result["metadata"]["fusion_original_model"] == "fusion-main"


async def test_routes_to_main_model():
    hook = _make_hook("main")
    data = {"model": "fusion-sidekick", "litellm_session_id": "task-1", "messages": []}

    result = await _route(hook, data)

    assert result is not None
    assert result["model"] == "fusion-main"
    assert result["metadata"]["fusion_role"] == "main"


async def test_noop_without_session_id():
    hook = _make_hook("sidekick")
    data = {"model": "fusion-main", "messages": []}

    result = await _route(hook, data)

    assert result is None


async def test_preserves_existing_metadata():
    hook = _make_hook("sidekick")
    data = {
        "model": "fusion-main",
        "litellm_session_id": "task-1",
        "messages": [],
        "metadata": {"user_id": "u-1"},
    }

    result = await _route(hook, data)

    assert result is not None
    assert result["metadata"]["user_id"] == "u-1"
    assert result["metadata"]["fusion_role"] == "sidekick"


@pytest.mark.parametrize(
    "sidekick_tools,tool_name,expected_role",
    [
        (frozenset({"read_file", "grep"}), "read_file", "sidekick"),
        (frozenset({"read_file", "grep"}), "edit_file", "main"),
        (frozenset({"read_file", "grep"}), None, "main"),
    ],
)
async def test_tool_name_classifier(
    sidekick_tools: FrozenSet[str], tool_name: Optional[str], expected_role: FusionRole
):
    classifier = ToolNameFusionClassifier(sidekick_tools=sidekick_tools)
    messages = [{"role": "assistant", "tool_calls": [{"function": {"name": tool_name}}]}] if tool_name else []

    role = await classifier.classify({"messages": messages}, session_id="task-1")

    assert role == expected_role
