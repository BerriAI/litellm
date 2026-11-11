"""
Tests for VERIA-53: PromQL string-literal quoting in
``get_daily_spend_from_prometheus``.

PromQL string literals follow Go's escape rules
(https://prometheus.io/docs/prometheus/latest/querying/basics/). JSON's
quoting is a strict subset of Go's, so ``json.dumps`` produces a literal
Prometheus parses identically.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_quote_safe_input_round_trips():
    from litellm.integrations.prometheus_helpers.prometheus_api import (
        _quote_promql_string_literal,
    )

    assert _quote_promql_string_literal("sk-abc123") == '"sk-abc123"'
    assert _quote_promql_string_literal("hash:deadbeef") == '"hash:deadbeef"'


def test_quote_escapes_double_quote():
    from litellm.integrations.prometheus_helpers.prometheus_api import (
        _quote_promql_string_literal,
    )

    # A bare double quote would otherwise terminate the label matcher and
    # let the attacker append `, foo="..."} or sum(...)`.
    assert _quote_promql_string_literal('hello"injected') == '"hello\\"injected"'


def test_quote_escapes_backslash():
    from litellm.integrations.prometheus_helpers.prometheus_api import (
        _quote_promql_string_literal,
    )

    assert _quote_promql_string_literal('a\\"b') == '"a\\\\\\"b"'


def test_quote_escapes_newlines_and_control_chars():
    """Beyond the security minimum, the canonical Go/JSON escape also
    handles control characters that would otherwise produce an invalid
    PromQL string literal."""
    from litellm.integrations.prometheus_helpers.prometheus_api import (
        _quote_promql_string_literal,
    )

    assert _quote_promql_string_literal("a\nb") == '"a\\nb"'
    assert _quote_promql_string_literal("a\tb") == '"a\\tb"'
    assert _quote_promql_string_literal("a\rb") == '"a\\rb"'


@pytest.mark.asyncio
async def test_get_daily_spend_does_not_pass_raw_quote_into_query():
    from litellm.integrations.prometheus_helpers import prometheus_api

    captured = {}

    class _FakeResponse:
        def json(self):
            return {"data": {"result": []}}

    async def _capture(url, params):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse()

    fake_client = MagicMock()
    fake_client.get = AsyncMock(side_effect=_capture)

    with patch.object(prometheus_api, "PROMETHEUS_URL", "http://prom.example"):
        with patch.object(prometheus_api, "async_http_handler", fake_client):
            await prometheus_api.get_daily_spend_from_prometheus(
                api_key='sk-victim"} or sum(other_metric{a="b'
            )

    rendered_query = captured["params"]["query"]
    # The legitimate matcher framing must still be intact: one outer
    # `delta()` window, one inner `hashed_api_key="..."` matcher.
    assert rendered_query.startswith(
        'sum(delta(litellm_spend_metric_total{hashed_api_key="'
    )
    assert rendered_query.endswith('"}[1d]))')

    # Every injected `"` from the attacker payload appears as `\"` so the
    # PromQL parser treats them as literal characters inside the matcher
    # value, never as the terminator that would let the rest parse as
    # PromQL syntax.
    inner = rendered_query[
        len('sum(delta(litellm_spend_metric_total{hashed_api_key="') : -len('"}[1d]))')
    ]
    assert '"' not in inner.replace('\\"', "")


@pytest.mark.asyncio
async def test_get_daily_spend_with_no_api_key_uses_unfiltered_query():
    from litellm.integrations.prometheus_helpers import prometheus_api

    captured = {}

    class _FakeResponse:
        def json(self):
            return {"data": {"result": []}}

    async def _capture(url, params):
        captured["params"] = params
        return _FakeResponse()

    fake_client = MagicMock()
    fake_client.get = AsyncMock(side_effect=_capture)

    with patch.object(prometheus_api, "PROMETHEUS_URL", "http://prom.example"):
        with patch.object(prometheus_api, "async_http_handler", fake_client):
            await prometheus_api.get_daily_spend_from_prometheus(api_key=None)

    assert captured["params"]["query"] == "sum(delta(litellm_spend_metric_total[1d]))"


@pytest.mark.asyncio
async def test_get_daily_spend_legitimate_hashed_key_unchanged():
    """A normal hex hashed_api_key flows through `json.dumps` as itself
    plus the surrounding quotes — no spurious escaping that would break
    real lookups."""
    from litellm.integrations.prometheus_helpers import prometheus_api

    captured = {}

    class _FakeResponse:
        def json(self):
            return {"data": {"result": []}}

    async def _capture(url, params):
        captured["params"] = params
        return _FakeResponse()

    fake_client = MagicMock()
    fake_client.get = AsyncMock(side_effect=_capture)

    legit_key = "a" * 64  # 64-char hex-ish hashed key
    with patch.object(prometheus_api, "PROMETHEUS_URL", "http://prom.example"):
        with patch.object(prometheus_api, "async_http_handler", fake_client):
            await prometheus_api.get_daily_spend_from_prometheus(api_key=legit_key)

    assert (
        captured["params"]["query"]
        == f'sum(delta(litellm_spend_metric_total{{hashed_api_key="{legit_key}"}}[1d]))'
    )
