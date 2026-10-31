from typing import Final

import httpx
import pytest
import respx
from pydantic import JsonValue, TypeAdapter, ValidationError

import litellm

_REQUEST: Final = TypeAdapter(dict[str, JsonValue])


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


@pytest.mark.asyncio
@pytest.mark.parametrize("base", ["https://gateway.example", "https://gateway.example/v1/"])
async def test_apply_returns_native_findings_using_shared_gateway_settings(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, base: str
) -> None:
    monkeypatch.setattr(litellm, "api_base", base)
    monkeypatch.setattr(litellm, "api_key", "test-key")
    verdict: Final = {"is_grounded": False, "reason": "The source gives a different height.", "source_ids": []}
    route: Final = respx_mock.post("https://gateway.example/guardrails/apply_guardrail").respond(json=verdict)
    messages: Final = [{"role": "user", "content": "The tower is 330 metres tall."}]

    result: Final = await litellm.guardrails.apply("grounding", response="450 metres", messages=messages)

    assert result == verdict
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-key"
    assert _REQUEST.validate_json(route.calls.last.request.content) == {
        "guardrail_name": "grounding",
        "text": "450 metres",
        "input_type": "response",
        "response_format": "native",
        "messages": messages,
    }


@pytest.mark.asyncio
async def test_apply_infers_input_direction_and_preserves_empty_values(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post("https://gateway.example/guardrails/apply_guardrail").respond(json={"safe": True})

    result: Final = await litellm.guardrails.apply(
        "safety", input="", messages=[], metadata={}, api_base="https://gateway.example", api_key="explicit-key"
    )

    assert result == {"safe": True}
    assert route.calls.last.request.headers["Authorization"] == "Bearer explicit-key"
    assert _REQUEST.validate_json(route.calls.last.request.content) == {
        "guardrail_name": "safety",
        "text": "",
        "input_type": "request",
        "response_format": "native",
        "messages": [],
        "metadata": {},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [{}, {"input": "prompt", "response": "answer"}])
async def test_apply_requires_exactly_one_text(texts: dict[str, str], respx_mock: respx.MockRouter) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        await litellm.guardrails.apply(
            "grounding", input=texts.get("input"), response=texts.get("response"), api_base="https://gateway.example"
        )
    assert not respx_mock.calls


@pytest.mark.asyncio
async def test_apply_requires_a_gateway_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "api_base", None)
    with pytest.raises(ValueError, match="gateway URL"):
        await litellm.guardrails.apply("grounding", response="answer")


@pytest.mark.asyncio
async def test_apply_surfaces_guardrail_http_failures(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://gateway.example/guardrails/apply_guardrail").respond(
        422, json={"error": {"message": "Guardrail did not report structured native findings."}}
    )
    with pytest.raises(httpx.HTTPStatusError) as failure:
        await litellm.guardrails.apply("grounding", response="answer", api_base="https://gateway.example")
    assert failure.value.response.status_code == 422


@pytest.mark.asyncio
async def test_apply_rejects_an_unstructured_response(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://gateway.example/guardrails/apply_guardrail").respond(json="allow")
    with pytest.raises(ValidationError):
        await litellm.guardrails.apply("grounding", response="answer", api_base="https://gateway.example")
