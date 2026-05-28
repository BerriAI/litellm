"""
Regression tests for LIT-3391.

Bedrock returns ``x-amzn-RequestId`` on every ApplyGuardrail response. Prior to
this fix the response headers were discarded; downstream loggers (Langfuse,
Datadog, OTEL, the Logs UI) had no way to correlate a LiteLLM request to the
provider-side guardrail execution.
"""

from contextlib import ExitStack
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockGuardrail,
)


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"x-amzn-RequestId": "abc-123"}, "abc-123"),
        ({"X-AMZN-REQUESTID": "abc-123"}, "abc-123"),
        ({"x-amz-request-id": "legacy-99"}, "legacy-99"),
        ({"x-amzn-RequestId": "canonical", "x-amz-request-id": "legacy"}, "canonical"),
        ({"content-type": "application/json"}, None),
        ({}, None),
    ],
)
def test_extract_request_id_dict(headers, expected):
    h = httpx.Headers(headers)
    assert BedrockGuardrail._extract_bedrock_request_id(h) == expected


def test_extract_request_id_none_input():
    assert BedrockGuardrail._extract_bedrock_request_id(None) is None


def test_extract_request_id_empty_string_skipped():
    h = httpx.Headers({"x-amzn-RequestId": ""})
    assert BedrockGuardrail._extract_bedrock_request_id(h) is None


def test_extract_request_id_never_raises():
    class Boom:
        def get(self, *a, **k):
            raise RuntimeError("simulated httpx malfunction")

    assert BedrockGuardrail._extract_bedrock_request_id(Boom()) is None


def _bare_guardrail():
    return BedrockGuardrail.__new__(BedrockGuardrail)


def test_build_tracing_detail_includes_request_id():
    g = _bare_guardrail()
    response = {"action": "GUARDRAIL_INTERVENED", "assessments": []}
    headers = httpx.Headers({"x-amzn-RequestId": "req-xyz"})
    td = g._build_tracing_detail(response, response_headers=headers)
    assert td.get("provider_request_id") == "req-xyz"
    assert td.get("guardrail_action") == "GUARDRAIL_INTERVENED"


def test_build_tracing_detail_omits_request_id_when_headers_absent():
    g = _bare_guardrail()
    response = {"action": "NONE", "assessments": []}
    td = g._build_tracing_detail(response)
    assert "provider_request_id" not in td


def test_build_tracing_detail_omits_request_id_when_header_missing():
    g = _bare_guardrail()
    response = {"action": "NONE", "assessments": []}
    td = g._build_tracing_detail(
        response,
        response_headers=httpx.Headers({"content-type": "application/json"}),
    )
    assert "provider_request_id" not in td


class _IntegrationGuardrail:
    def __enter__(self):
        self._stack = ExitStack()
        self._stack.enter_context(
            patch.object(
                BedrockGuardrail,
                "_load_credentials",
                return_value=(MagicMock(), "us-east-1"),
            )
        )
        prep = self._stack.enter_context(patch.object(BedrockGuardrail, "_prepare_request"))
        p = MagicMock()
        p.url = "https://bedrock-runtime.us-east-1.amazonaws.com/guardrail/test/version/1/apply"
        p.body = b"{}"
        p.headers = {"Content-Type": "application/json"}
        prep.return_value = p
        self.g = BedrockGuardrail(
            guardrailIdentifier="test-id",
            guardrailVersion="1",
            aws_region_name="us-east-1",
            guardrail_name="test_guardrail",
            event_hook="pre_call",
        )
        return self.g

    def __exit__(self, *exc):
        self._stack.close()


def _fake_response(status, headers, body):
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.headers = httpx.Headers(headers)
    r.json = MagicMock(return_value=body)
    r.text = str(body)
    return r


@pytest.mark.asyncio
async def test_make_bedrock_api_request_success_records_request_id():
    with _IntegrationGuardrail() as g:
        fake = _fake_response(
            200,
            {"x-amzn-RequestId": "success-request-id-001"},
            {"action": "NONE", "assessments": []},
        )
        g.async_handler = MagicMock()
        g.async_handler.post = AsyncMock(return_value=fake)
        rd: Dict[str, Any] = {"metadata": {}}
        await g.make_bedrock_api_request(
            source="INPUT",
            messages=[{"role": "user", "content": "hello"}],
            request_data=rd,
        )
        slg = rd["metadata"]["standard_logging_guardrail_information"]
        assert len(slg) == 1
        assert slg[0].get("provider_request_id") == "success-request-id-001"
        assert slg[0].get("guardrail_action") == "NONE"


@pytest.mark.asyncio
async def test_make_bedrock_api_request_http_error_records_request_id():
    with _IntegrationGuardrail() as g:
        err = _fake_response(
            400,
            {"x-amzn-RequestId": "err-request-id-042"},
            {"message": "ValidationException: bad input"},
        )
        raised = httpx.HTTPStatusError("400", request=MagicMock(), response=err)
        g.async_handler = MagicMock()
        g.async_handler.post = AsyncMock(side_effect=raised)
        rd: Dict[str, Any] = {"metadata": {}}
        with pytest.raises(Exception):
            await g.make_bedrock_api_request(
                source="INPUT",
                messages=[{"role": "user", "content": "hello"}],
                request_data=rd,
            )
        slg = rd["metadata"]["standard_logging_guardrail_information"]
        assert len(slg) == 1
        assert slg[0].get("guardrail_status") == "guardrail_failed_to_respond"
        assert slg[0].get("provider_request_id") == "err-request-id-042"


@pytest.mark.asyncio
async def test_make_bedrock_api_request_success_omits_request_id_when_header_missing():
    with _IntegrationGuardrail() as g:
        fake = _fake_response(
            200,
            {"content-type": "application/json"},
            {"action": "NONE", "assessments": []},
        )
        g.async_handler = MagicMock()
        g.async_handler.post = AsyncMock(return_value=fake)
        rd: Dict[str, Any] = {"metadata": {}}
        await g.make_bedrock_api_request(
            source="INPUT",
            messages=[{"role": "user", "content": "hello"}],
            request_data=rd,
        )
        slg = rd["metadata"]["standard_logging_guardrail_information"]
        assert len(slg) == 1
        assert not slg[0].get("provider_request_id")
