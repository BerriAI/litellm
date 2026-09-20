"""Function-level parity: the port's `marked` against the legacy hook's Messages-route function
on the same bodies. The legacy hook never runs on the native route, so there is no call to share."""

import copy

import pytest

from litellm.callbacks_v1.builtin import anthropic_cache_control as port
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from tests.test_litellm.callbacks_v1.builtin.support import golden, plain

EPHEMERAL = {"type": "ephemeral"}
TURNS = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
    {"role": "user", "content": [{"type": "text", "text": "third"}, {"type": "text", "text": "fourth"}]},
]
ALREADY_MARKED = [
    {"role": "user", "content": [{"type": "text", "text": "kept", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}
]
FIVE_USERS = [{"role": "user", "content": f"turn {index}"} for index in range(5)]

CASES = {
    "system string": ({"system": "be brief", "messages": TURNS}, [("system", None, None)]),
    "system blocks": (
        {"system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "messages": TURNS},
        [("system", None, None)],
    ),
    "no system to mark": ({"messages": TURNS}, [("system", None, None)]),
    "every user turn": ({"messages": TURNS}, [("user", None, None)]),
    "last message by index": ({"messages": TURNS}, [(None, -1, None)]),
    "index out of range": ({"messages": TURNS}, [(None, 9, None)]),
    "ttl": ({"system": "be brief", "messages": TURNS}, [("system", None, "1h"), (None, -1, "5m")]),
    "client breakpoint is kept": ({"messages": ALREADY_MARKED}, [("user", None, None)]),
    "stops at four breakpoints": (
        {"system": "be brief", "messages": FIVE_USERS},
        [("system", None, None), ("user", None, None)],
    ),
}


def legacy(body: dict, points: list[tuple]) -> dict:
    injection_points = [
        {"location": "message", "role": role, "index": index, "control": {**EPHEMERAL, **({"ttl": ttl} if ttl else {})}}
        for role, index, ttl in points
    ]
    messages, system, _ = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
        copy.deepcopy(body["messages"]),
        copy.deepcopy(body.get("system")),
        injection_points,  # pyright: ignore[reportArgumentType]  # plain dicts stand in for the legacy TypedDicts
    )
    return {**body, "messages": messages, **({"system": system} if "system" in body else {})}


@pytest.mark.parametrize("name", CASES)
def test_the_port_marks_a_messages_body_exactly_as_the_legacy_hook_does(name: str) -> None:
    body, points = CASES[name]
    config = port.Config(points=tuple(port.InjectionPoint(role, index, ttl) for role, index, ttl in points))
    original = copy.deepcopy(body)

    assert plain(port.AnthropicCacheControl(config).marked(body)) == legacy(body, points)
    assert body == original


def test_the_patch_covers_only_the_configured_providers() -> None:
    request = {**golden("request.sending")["event"], "body": {"system": "be brief", "messages": TURNS}}
    config = port.Config(points=(port.InjectionPoint(role="system"),))

    interceptor = port.AnthropicCacheControl(config)

    assert interceptor.payload(request) is None  # pyright: ignore[reportArgumentType]  # the golden event plus a Messages body
    patch = interceptor.payload({**request, "custom_llm_provider": "anthropic"})  # pyright: ignore[reportArgumentType]  # the golden event plus a Messages body

    assert patch is not None and set(patch) == {"body"}
    assert plain(patch["body"])["system"] == [{"type": "text", "text": "be brief", "cache_control": EPHEMERAL}]  # pyright: ignore[reportIndexIssue, reportTypedDictNotRequiredAccess, reportCallIssue, reportArgumentType]  # the patch body is the Messages mapping


def test_a_body_that_is_not_a_messages_request_is_left_alone() -> None:
    config = port.Config(points=(port.InjectionPoint(role="user"),))
    assert port.AnthropicCacheControl(config).marked({"input": "hello"}) is None
    assert port.AnthropicCacheControl(port.Config(points=())).marked({"messages": TURNS}) is None
