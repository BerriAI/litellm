import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

from litellm.integrations.otel.model.spans import SpanRole
from litellm.integrations.otel.runtime import traced


@pytest.mark.asyncio
async def test_decorator_is_passthrough_when_v2_disabled():
    calls = []

    @traced("identity.unit-test", role=SpanRole.SERVICE)
    async def work(x):
        calls.append(x)
        return x * 2

    result = await work(5)
    assert result == 10
    assert calls == [5]


@pytest.mark.asyncio
async def test_attrs_callback_invoked_with_result():
    captured = {}

    @traced(
        "identity.attrs-test",
        role=SpanRole.SERVICE,
        attrs=lambda result: captured.setdefault("result", result) or {},
    )
    async def work():
        return {"principal_kind": "api_key"}

    await work()
    assert captured["result"] == {"principal_kind": "api_key"}


@pytest.mark.asyncio
async def test_attrs_exception_is_swallowed():
    @traced(
        "identity.attrs-error",
        role=SpanRole.SERVICE,
        attrs=lambda result: 1 / 0,
    )
    async def work():
        return "ok"

    assert await work() == "ok"


@pytest.mark.asyncio
async def test_db_call_role_does_not_open_a_new_span():
    """DB_CALL spans piggyback on the current span instead of starting one.

    With V2 disabled there is no current span; the decorator should still
    run the function and swallow any attribute-application errors.
    """

    @traced("identity.db-test", role=SpanRole.DB_CALL)
    async def work():
        return "value"

    assert await work() == "value"


def test_decorator_rejects_sync_functions():
    with pytest.raises(TypeError):
        @traced("identity.sync-not-allowed", role=SpanRole.SERVICE)
        def sync_fn():
            return None
