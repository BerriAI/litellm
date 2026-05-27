"""
Regression tests for LIT-2579: when multiple post-call guardrails are
configured and the first one raises (blocks the response), every other
guardrail must still run so its standard_logging_guardrail_information entry
is recorded in request_data["metadata"]. Without this, spend logs / Langfuse
/ DataDog only show the triggering guardrail and ops cannot see which other
guardrails passed (or also fired) on the same request.

The fix is in ProxyLogging.post_call_success_hook in litellm/proxy/utils.py:
the loop captures the first exception and re-raises it only after every
guardrail callback has run.
"""

import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks


class _BlockingPostCallGuardrail(CustomGuardrail):
    """First guardrail in the chain - blocks the response."""

    def __init__(self, name: str = "blocking-guardrail"):
        super().__init__(
            guardrail_name=name,
            supported_event_hooks=[GuardrailEventHooks.post_call],
            event_hook=GuardrailEventHooks.post_call,
            default_on=True,
        )

    @log_guardrail_information
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        raise HTTPException(
            status_code=400,
            detail={"error": f"{self.guardrail_name} policy violation"},
        )


class _PassingPostCallGuardrail(CustomGuardrail):
    """Subsequent guardrail in the chain - passes (no violation)."""

    def __init__(self, name: str = "passing-guardrail"):
        super().__init__(
            guardrail_name=name,
            supported_event_hooks=[GuardrailEventHooks.post_call],
            event_hook=GuardrailEventHooks.post_call,
            default_on=True,
        )
        self.calls = 0

    @log_guardrail_information
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        self.calls += 1
        return None


def _make_inputs():
    request_data = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {},
    }
    response = litellm.ModelResponse(
        id="chatcmpl-test",
        choices=[
            {
                "message": {"role": "assistant", "content": "hello"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        model="fake-model",
        object="chat.completion",
    )
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test")
    return request_data, response, user_api_key_dict


@pytest.mark.asyncio
async def test_blocking_guardrail_does_not_stop_subsequent_guardrails_from_recording_metadata(
    monkeypatch,
):
    """
    LIT-2579: A blocking post-call guardrail must not prevent subsequent
    post-call guardrails from running. Every guardrail's outcome must land
    in request_data["metadata"]["standard_logging_guardrail_information"],
    and the blocking guardrail's exception must still surface to the caller.
    """
    blocking = _BlockingPostCallGuardrail(name="blocking-guardrail")
    passing = _PassingPostCallGuardrail(name="passing-guardrail")
    monkeypatch.setattr(litellm, "callbacks", [blocking, passing])

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    request_data, response, user_api_key_dict = _make_inputs()

    with pytest.raises(HTTPException) as exc_info:
        await proxy_logging.post_call_success_hook(
            data=request_data,
            response=response,
            user_api_key_dict=user_api_key_dict,
        )

    # 1. The blocking guardrail's exception surfaces to the caller.
    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value.detail, dict)
    assert exc_info.value.detail.get("guardrail_name") == "blocking-guardrail"

    # 2. The passing guardrail actually ran (this is the bug fix).
    assert passing.calls == 1, (
        "passing-guardrail.async_post_call_success_hook was not invoked after "
        "blocking-guardrail raised - the post-call loop bailed early."
    )

    # 3. BOTH guardrails recorded entries in metadata.
    entries = request_data["metadata"].get("standard_logging_guardrail_information")
    assert isinstance(entries, list)
    names = [e.get("guardrail_name") for e in entries]
    assert names == ["blocking-guardrail", "passing-guardrail"], (
        f"expected both guardrails to record metadata in order, got {names!r}"
    )
    statuses = {e["guardrail_name"]: e.get("guardrail_status") for e in entries}
    assert statuses["blocking-guardrail"] == "guardrail_intervened"
    assert statuses["passing-guardrail"] == "success"


@pytest.mark.asyncio
async def test_two_blocking_guardrails_record_both_and_raise_first(monkeypatch):
    """
    If two guardrails both raise, every guardrail still gets an entry in
    metadata, and the FIRST exception is the one surfaced to the caller.
    """
    first = _BlockingPostCallGuardrail(name="first-blocking")
    second = _BlockingPostCallGuardrail(name="second-blocking")
    monkeypatch.setattr(litellm, "callbacks", [first, second])

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    request_data, response, user_api_key_dict = _make_inputs()

    with pytest.raises(HTTPException) as exc_info:
        await proxy_logging.post_call_success_hook(
            data=request_data,
            response=response,
            user_api_key_dict=user_api_key_dict,
        )

    # First-blocking is what surfaces.
    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value.detail, dict)
    assert exc_info.value.detail.get("guardrail_name") == "first-blocking"

    # Both raised guardrails have intervened entries.
    entries = request_data["metadata"].get("standard_logging_guardrail_information")
    assert isinstance(entries, list)
    statuses = {e["guardrail_name"]: e["guardrail_status"] for e in entries}
    assert statuses == {
        "first-blocking": "guardrail_intervened",
        "second-blocking": "guardrail_intervened",
    }


@pytest.mark.asyncio
async def test_no_blocking_guardrail_passes_through_normally(monkeypatch):
    """
    Sanity check: with two passing guardrails, no exception is raised and
    both record their metadata entries (this verifies the fix didn't break
    the happy path).
    """
    a = _PassingPostCallGuardrail(name="passing-a")
    b = _PassingPostCallGuardrail(name="passing-b")
    monkeypatch.setattr(litellm, "callbacks", [a, b])

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    request_data, response, user_api_key_dict = _make_inputs()

    await proxy_logging.post_call_success_hook(
        data=request_data,
        response=response,
        user_api_key_dict=user_api_key_dict,
    )

    # Both guardrails ran.
    assert a.calls == 1 and b.calls == 1

    entries = request_data["metadata"].get("standard_logging_guardrail_information")
    assert isinstance(entries, list)
    names = [e["guardrail_name"] for e in entries]
    assert names == ["passing-a", "passing-b"]
    for entry in entries:
        assert entry["guardrail_status"] == "success"

