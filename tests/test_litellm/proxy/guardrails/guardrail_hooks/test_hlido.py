from types import SimpleNamespace
from typing import Any, List

import httpx
import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.hlido import (
    HlidoGuardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
)
from litellm.types.guardrails import SupportedGuardrailIntegrations

_ENDPOINT = "https://hlido.test/v1/agents/some-agent"


def _resp(body: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("GET", _ENDPOINT),
    )


def _agent_body(score: int = 90, tier: str = "VITAL") -> dict:
    return {
        "slug": "some-agent",
        "name": "Some Agent",
        "score": score,
        "tier": tier,
        "evidence_url": "https://hlido.test/reviews/some-agent/",
    }


class FakeHandler:
    def __init__(self, items: List[Any]):
        self._items = list(items)
        self.calls: List[SimpleNamespace] = []

    async def get(self, *, url, headers=None, timeout=None, **kwargs):
        self.calls.append(SimpleNamespace(url=url, headers=headers, timeout=timeout))
        if not self._items:
            raise AssertionError("FakeHandler ran out of programmed responses")
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _make_guardrail(handler: FakeHandler, **kwargs) -> HlidoGuardrail:
    defaults = {
        "api_base": "https://hlido.test",
        "slugs": ("some-agent",),
        "async_handler": handler,
        "guardrail_name": "hlido-trust",
        "event_hook": "pre_call",
        "default_on": True,
    }
    defaults.update(kwargs)
    return HlidoGuardrail(**defaults)


def test_registries_expose_hlido():
    assert (
        guardrail_initializer_registry[SupportedGuardrailIntegrations.HLIDO.value]
        is not None
    )
    assert (
        guardrail_class_registry[SupportedGuardrailIntegrations.HLIDO.value]
        is HlidoGuardrail
    )


@pytest.mark.asyncio
async def test_allows_agent_above_min_score():
    handler = FakeHandler([_resp(_agent_body(score=90))])
    guardrail = _make_guardrail(handler, min_score=60)

    await guardrail._check_request({})

    assert len(handler.calls) == 1
    assert handler.calls[0].url == _ENDPOINT


@pytest.mark.asyncio
async def test_blocks_agent_below_min_score():
    handler = FakeHandler([_resp(_agent_body(score=42, tier="FADING"))])
    guardrail = _make_guardrail(handler, min_score=60)

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail._check_request({})

    assert "some-agent" in str(excinfo.value)
    assert "42" in str(excinfo.value)


@pytest.mark.asyncio
async def test_blocks_agent_outside_allowed_tiers():
    handler = FakeHandler([_resp(_agent_body(score=90, tier="FADING"))])
    guardrail = _make_guardrail(handler, allowed_tiers=("VITAL", "STRONG"))

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail._check_request({})

    assert "FADING" in str(excinfo.value)


@pytest.mark.asyncio
async def test_unverified_agent_allowed_by_default():
    handler = FakeHandler([_resp({"error": "not_found"}, status_code=404)])
    guardrail = _make_guardrail(handler)

    await guardrail._check_request({})


@pytest.mark.asyncio
async def test_unverified_agent_blocked_when_configured():
    handler = FakeHandler([_resp({"error": "not_found"}, status_code=404)])
    guardrail = _make_guardrail(handler, on_unverified="block")

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail._check_request({})

    assert "no independent review" in str(excinfo.value)


@pytest.mark.asyncio
async def test_api_failure_allows_by_default():
    handler = FakeHandler(
        [httpx.ConnectError("boom", request=httpx.Request("GET", _ENDPOINT))]
    )
    guardrail = _make_guardrail(handler)

    await guardrail._check_request({})


@pytest.mark.asyncio
async def test_api_failure_blocks_when_configured():
    handler = FakeHandler(
        [httpx.ConnectError("boom", request=httpx.Request("GET", _ENDPOINT))]
    )
    guardrail = _make_guardrail(handler, on_error="block")

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail._check_request({})

    assert "API unreachable" in str(excinfo.value)


@pytest.mark.asyncio
async def test_request_metadata_slugs_are_checked():
    handler = FakeHandler(
        [
            _resp(_agent_body()),
            _resp(
                {
                    "slug": "weak-agent",
                    "score": 10,
                    "tier": "WEAK",
                    "evidence_url": "https://hlido.test/reviews/weak-agent/",
                },
            ),
        ]
    )
    guardrail = _make_guardrail(handler, min_score=60)

    with pytest.raises(GuardrailRaisedException) as excinfo:
        await guardrail._check_request({"metadata": {"hlido_slugs": ["weak-agent"]}})

    assert "weak-agent" in str(excinfo.value)
    assert [c.url for c in handler.calls] == [
        _ENDPOINT,
        "https://hlido.test/v1/agents/weak-agent",
    ]


@pytest.mark.asyncio
async def test_lookup_is_cached_between_checks():
    handler = FakeHandler([_resp(_agent_body())])
    guardrail = _make_guardrail(handler, min_score=60)

    await guardrail._check_request({})
    await guardrail._check_request({})

    assert len(handler.calls) == 1


@pytest.mark.asyncio
async def test_no_slugs_makes_no_api_calls():
    handler = FakeHandler([])
    guardrail = _make_guardrail(handler, slugs=None)

    await guardrail._check_request({})

    assert handler.calls == []


@pytest.mark.asyncio
async def test_api_key_sent_as_bearer_when_configured():
    handler = FakeHandler([_resp(_agent_body())])
    guardrail = _make_guardrail(handler, api_key="hlk_live_test")

    await guardrail._check_request({})

    assert handler.calls[0].headers["Authorization"] == "Bearer hlk_live_test"
