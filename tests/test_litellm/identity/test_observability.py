import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

from litellm.integrations.otel.model.spans import SpanRole
from litellm.integrations.otel.runtime import (
    _apply_span_attrs,
    _resolve_attrs,
    traced,
)


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


def test_resolve_attrs_passes_result_positionally_when_builder_lacks_result_param():
    seen = {}

    def builder(value):
        seen["value"] = value
        return {"identity.kind": "api_key"}

    out = _resolve_attrs(builder, args=("a",), kwargs={"k": "v"}, result="RESULT")

    assert seen["value"] == "RESULT"
    assert out == {"identity.kind": "api_key"}


def test_resolve_attrs_returns_empty_without_a_builder():
    assert _resolve_attrs(None, args=(), kwargs={}, result="x") == {}


class _RecordingSpan:
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, key, value):
        self.attrs[key] = value


def test_apply_span_attrs_sets_non_null_values_and_skips_none():
    span = _RecordingSpan()
    _apply_span_attrs(span, {"a": 1, "b": None, "c": "x"})
    assert span.attrs == {"a": 1, "c": "x"}


def test_apply_span_attrs_is_noop_without_span_or_attrs():
    _apply_span_attrs(None, {"a": 1})
    _apply_span_attrs(_RecordingSpan(), None)


def test_apply_span_attrs_swallows_set_attribute_errors():
    class _Boom:
        def set_attribute(self, key, value):
            raise RuntimeError("span closed")

    _apply_span_attrs(_Boom(), {"a": 1})
