from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.semantic_cache_gate import gated_cache_overrides
from litellm.proxy.route_llm_request import _merge_router_settings_override
from litellm.proxy.utils import ProxyLogging
from litellm.types.caching import (
    DEFAULT_CACHING_SUPPORTED_CALL_TYPES,
    CachingSupportedCallTypes,
    LiteLLMCacheType,
)

ALLOWED_TEAM = "team-eval"
OTHER_TEAM = "team-other"


class _TeamAllowlist(CustomLogger):
    def __init__(self, teams: frozenset[str]) -> None:
        super().__init__()
        self.teams = teams
        self.seen: list[tuple[str | None, object, str]] = []

    async def async_should_semantic_cache(
        self, user_api_key_dict: UserAPIKeyAuth, data: Mapping[str, object], call_type: str
    ) -> bool:
        self.seen.append((user_api_key_dict.team_id, data.get("model"), call_type))
        return user_api_key_dict.team_id in self.teams


class _RaisingGate(CustomLogger):
    async def async_should_semantic_cache(
        self, user_api_key_dict: UserAPIKeyAuth, data: Mapping[str, object], call_type: str
    ) -> bool:
        raise RuntimeError("allowlist service down")


class _TruthyGate(CustomLogger):
    async def async_should_semantic_cache(  # pyright: ignore[reportIncompatibleMethodOverride]  # operator hooks are untyped Python
        self, user_api_key_dict: UserAPIKeyAuth, data: Mapping[str, object], call_type: str
    ) -> str:
        return "yes"


def _semantic_cache(
    supported_call_types: tuple[CachingSupportedCallTypes, ...] = DEFAULT_CACHING_SUPPORTED_CALL_TYPES,
) -> Cache:
    return Cache(
        type=LiteLLMCacheType.VALKEY_SEMANTIC,
        host="localhost",
        port="6379",
        similarity_threshold=0.8,
        supported_call_types=list(supported_call_types),
    )


def _request(**cache_controls: object) -> Mapping[str, object]:
    return {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What color is the sky?"}],
        **({"cache": cache_controls} if cache_controls else {}),
    }


def _key(team_id: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-hashed", team_id=team_id)


@pytest.mark.asyncio
async def test_a_team_the_gate_rejects_skips_semantic_lookup_and_store_even_when_the_client_opts_in() -> None:
    gate: Final = _TeamAllowlist(frozenset({ALLOWED_TEAM}))
    data: Final = _request(**{"use-cache": True, "ttl": 60})

    result: Final = await gated_cache_overrides(
        data=data, user_api_key_dict=_key(OTHER_TEAM), call_type="acompletion", gates=(gate,), cache=_semantic_cache()
    )

    assert result == {"cache": {"use-cache": True, "ttl": 60, "no-cache": True, "no-store": True}}
    assert gate.seen == [(OTHER_TEAM, "gpt-4o-mini", "acompletion")]


@pytest.mark.asyncio
async def test_a_team_the_gate_allows_keeps_its_own_cache_controls() -> None:
    data: Final = _request(ttl=60)

    result: Final = await gated_cache_overrides(
        data=data,
        user_api_key_dict=_key(ALLOWED_TEAM),
        call_type="acompletion",
        gates=(_TeamAllowlist(frozenset({ALLOWED_TEAM})),),
        cache=_semantic_cache(),
    )

    assert result == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("cache", [None, Cache(type=LiteLLMCacheType.LOCAL)], ids=["no-cache", "exact-match-cache"])
async def test_the_gate_is_never_consulted_without_a_semantic_cache(cache: Cache | None) -> None:
    gate: Final = _TeamAllowlist(frozenset())
    data: Final = _request()

    result: Final = await gated_cache_overrides(
        data=data, user_api_key_dict=_key(OTHER_TEAM), call_type="acompletion", gates=(gate,), cache=cache
    )

    assert result == {}
    assert gate.seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["acreate_batch", "avector_store_search", "aspeech", "atranscription", "arerank"])
async def test_the_gate_is_never_consulted_for_a_route_that_cannot_reach_the_cache(route: str) -> None:
    gate: Final = _TeamAllowlist(frozenset())

    result: Final = await gated_cache_overrides(
        data=_request(), user_api_key_dict=_key(OTHER_TEAM), call_type=route, gates=(gate,), cache=_semantic_cache()
    )

    assert result == {}
    assert gate.seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "served_call_type"),
    [
        ("aresponses", "acompletion"),
        ("acompletion", "aresponses"),
        ("anthropic_messages", "aresponses"),
        ("agenerate_content", "acompletion"),
        ("agenerate_content_stream", "acompletion"),
        ("acreate_interaction", "aresponses"),
        ("_aresponses_websocket", "aresponses"),
        ("asend_message", "acompletion"),
    ],
)
async def test_the_gate_covers_routes_that_reach_the_cache_through_a_bridge_whatever_call_types_it_serves(
    route: str, served_call_type: CachingSupportedCallTypes
) -> None:
    gate: Final = _TeamAllowlist(frozenset())

    result: Final = await gated_cache_overrides(
        data=_request(),
        user_api_key_dict=_key(OTHER_TEAM),
        call_type=route,
        gates=(gate,),
        cache=_semantic_cache(supported_call_types=(served_call_type,)),
    )

    assert result == {"cache": {"no-cache": True, "no-store": True}}
    assert gate.seen == [(OTHER_TEAM, "gpt-4o-mini", route)]


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", [_RaisingGate(), _TruthyGate()], ids=["raises", "returns-non-bool"])
async def test_a_gate_that_fails_or_answers_anything_but_true_skips_the_semantic_cache(gate: CustomLogger) -> None:
    result: Final = await gated_cache_overrides(
        data=_request(),
        user_api_key_dict=_key(ALLOWED_TEAM),
        call_type="aresponses",
        gates=(gate,),
        cache=_semantic_cache(),
    )

    assert result == {"cache": {"no-cache": True, "no-store": True}}


@pytest.mark.asyncio
@pytest.mark.parametrize("deny_first", [True, False])
async def test_one_rejecting_gate_skips_the_semantic_cache_whatever_the_others_answer(deny_first: bool) -> None:
    allow: Final = _TeamAllowlist(frozenset({ALLOWED_TEAM}))
    deny: Final = _TeamAllowlist(frozenset())

    result: Final = await gated_cache_overrides(
        data=_request(),
        user_api_key_dict=_key(ALLOWED_TEAM),
        call_type="anthropic_messages",
        gates=(deny, allow) if deny_first else (allow, deny),
        cache=_semantic_cache(),
    )

    assert result == {"cache": {"no-cache": True, "no-store": True}}


@pytest.mark.asyncio
async def test_only_callbacks_that_override_the_hook_act_as_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    gate: Final = _TeamAllowlist(frozenset({ALLOWED_TEAM}))
    monkeypatch.setattr(litellm, "callbacks", [CustomLogger(), gate])

    gates: Final = ProxyLogging.callback_capabilities().semantic_cache_gates
    result: Final = await gated_cache_overrides(
        data=_request(),
        user_api_key_dict=_key(OTHER_TEAM),
        call_type="acompletion",
        gates=gates,
        cache=_semantic_cache(),
    )

    assert gates == (gate,)
    assert result == {"cache": {"no-cache": True, "no-store": True}}


@pytest.mark.asyncio
async def test_a_rejected_team_cannot_reopen_the_semantic_cache_through_its_own_fallback_entries() -> None:
    data: Final = {
        **_request(ttl=60),
        "fallbacks": [{"model": "gpt-4o", "cache": {"ttl": 5}}, "gpt-4.1-mini", {"model": "gpt-4.1"}],
        "context_window_fallbacks": [{"gpt-4o-mini": ["gpt-4o"], "cache": {}}],
        "content_policy_fallbacks": [{"model": "gpt-4o", "cache": None}],
    }

    result: Final = await gated_cache_overrides(
        data=data,
        user_api_key_dict=_key(OTHER_TEAM),
        call_type="acompletion",
        gates=(_TeamAllowlist(frozenset({ALLOWED_TEAM})),),
        cache=_semantic_cache(),
    )

    assert result == {
        "cache": {"ttl": 60, "no-cache": True, "no-store": True},
        "fallbacks": [
            {"model": "gpt-4o", "cache": {"ttl": 5, "no-cache": True, "no-store": True}},
            "gpt-4.1-mini",
            {"model": "gpt-4.1"},
        ],
        "context_window_fallbacks": [{"gpt-4o-mini": ["gpt-4o"], "cache": {"no-cache": True, "no-store": True}}],
        "content_policy_fallbacks": [{"model": "gpt-4o", "cache": {"no-cache": True, "no-store": True}}],
    }


@pytest.mark.asyncio
async def test_a_rejected_team_cannot_reopen_the_semantic_cache_through_grouped_or_nested_fallback_targets() -> None:
    skip: Final = {"no-cache": True, "no-store": True}
    data: Final = {
        **_request(),
        "fallbacks": [
            {
                "gpt-4o-mini": [
                    "gpt-4.1-mini",
                    {"model": "gpt-4o", "cache": {}},
                    {"model": "gpt-4.1", "fallbacks": [{"model": "gpt-5", "cache": {"ttl": 5}}]},
                ]
            }
        ],
        "router_settings_override": {"content_policy_fallbacks": [{"gpt-4o-mini": [{"model": "gpt-4o", "cache": {}}]}]},
    }

    overrides: Final = await gated_cache_overrides(
        data=data,
        user_api_key_dict=_key(OTHER_TEAM),
        call_type="acompletion",
        gates=(_TeamAllowlist(frozenset({ALLOWED_TEAM})),),
        cache=_semantic_cache(),
    )

    assert overrides["fallbacks"] == [
        {
            "gpt-4o-mini": [
                "gpt-4.1-mini",
                {"model": "gpt-4o", "cache": skip},
                {"model": "gpt-4.1", "fallbacks": [{"model": "gpt-5", "cache": {"ttl": 5, **skip}}]},
            ]
        }
    ]
    assert overrides["router_settings_override"] == {
        "content_policy_fallbacks": [{"gpt-4o-mini": [{"model": "gpt-4o", "cache": skip}]}]
    }


@pytest.mark.asyncio
async def test_a_rejected_team_cannot_reopen_the_semantic_cache_through_router_settings_fallbacks() -> None:
    data: Final = {
        **_request(),
        "router_settings_override": {"fallbacks": [{"model": "gpt-4o", "cache": {}}], "num_retries": 2},
    }

    overrides: Final = await gated_cache_overrides(
        data=data,
        user_api_key_dict=_key(OTHER_TEAM),
        call_type="acompletion",
        gates=(_TeamAllowlist(frozenset({ALLOWED_TEAM})),),
        cache=_semantic_cache(),
    )
    routed: Final = {**data, **overrides}
    _ = _merge_router_settings_override(routed)

    assert routed["fallbacks"] == [{"model": "gpt-4o", "cache": {"no-cache": True, "no-store": True}}]
    assert routed["num_retries"] == 2
