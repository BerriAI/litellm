from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.hosted_vllm.count_tokens.token_counter import HostedVLLMTokenCounter
from litellm.llms.vllm.common_utils import VLLMModelInfo
from litellm.types.utils import TokenCountResponse

API_BASE: Final = "https://vllm.example.com/benchflow/qwen3-06b"
COUNT_URL: Final = f"{API_BASE}/v1/messages/count_tokens"


async def _async_count(
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
    api_key: object = None,
) -> tuple[TokenCountResponse | None, respx.Route]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm_params: Final[dict[str, object]] = {"api_base": API_BASE}
    if api_key is not None:
        litellm_params["api_key"] = api_key
    with respx.mock:
        route: Final = respx.post(COUNT_URL).mock(return_value=response)
        result: Final = await HostedVLLMTokenCounter().count_tokens(
            model_to_use="Qwen/Qwen3-0.6B",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": litellm_params},
            request_model="vllm-counted",
        )
        return result, route


def test_registered_for_the_hosted_vllm_provider_only() -> None:
    counter: Final = VLLMModelInfo().get_token_counter()

    assert isinstance(counter, HostedVLLMTokenCounter)
    assert counter.should_use_token_counting_api(custom_llm_provider="hosted_vllm")
    assert not counter.should_use_token_counting_api(custom_llm_provider="vllm")
    assert not counter.should_use_token_counting_api(custom_llm_provider="openai")
    assert not counter.should_use_token_counting_api(custom_llm_provider=None)


@pytest.mark.asyncio
async def test_reports_the_backend_count_from_the_count_tokens_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, route = await _async_count(monkeypatch, httpx.Response(200, json={"input_tokens": 574}))

    assert result is not None
    assert result.total_tokens == 574
    assert result.tokenizer_type == "hosted_vllm_api"
    assert result.error is False
    assert route.called


@pytest.mark.asyncio
async def test_keeps_the_backend_payload_and_model_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: Final = {"input_tokens": 12, "context_management": {"original_input_tokens": 12}}
    result, _ = await _async_count(monkeypatch, httpx.Response(200, json=payload))

    assert result is not None
    assert result.request_model == "vllm-counted"
    assert result.model_used == "Qwen/Qwen3-0.6B"
    assert result.original_response == payload


@pytest.mark.asyncio
async def test_sends_no_authorization_header_without_a_configured_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, route = await _async_count(monkeypatch, httpx.Response(200, json={"input_tokens": 3}))

    assert "authorization" not in {key.lower() for key in route.calls[0].request.headers}


@pytest.mark.asyncio
async def test_sends_the_configured_key_as_a_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _, route = await _async_count(monkeypatch, httpx.Response(200, json={"input_tokens": 3}), api_key="backend-key")

    assert route.calls[0].request.headers.get("authorization") == "Bearer backend-key"


@pytest.mark.asyncio
async def test_a_backend_without_the_route_falls_back_to_the_local_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = await _async_count(monkeypatch, httpx.Response(404, json={"detail": "Not Found"}))

    assert result is None


@pytest.mark.asyncio
async def test_a_missing_route_answered_as_plain_text_also_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = await _async_count(monkeypatch, httpx.Response(404, text="Not Found"))

    assert result is None


@pytest.mark.asyncio
async def test_a_backend_rejecting_the_model_is_not_masked_by_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected: Final = httpx.Response(
        404,
        json={"type": "error", "error": {"type": "NotFoundError", "message": "The model `Qwen/X` does not exist."}},
    )
    result, _ = await _async_count(monkeypatch, rejected)

    assert result is not None
    assert result.error is True
    assert result.status_code == 404
    assert "does not exist" in (result.error_message or "")


@pytest.mark.asyncio
async def test_a_backend_auth_failure_is_not_masked_by_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = await _async_count(
        monkeypatch,
        httpx.Response(401, json={"type": "error", "error": {"type": "AuthenticationError", "message": "bad key"}}),
        api_key="wrong-key",
    )

    assert result is not None
    assert result.error is True
    assert result.status_code == 401


@pytest.mark.asyncio
async def test_missing_api_base_uses_the_local_estimate() -> None:
    result: Final = await HostedVLLMTokenCounter().count_tokens(
        model_to_use="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "hi"}],
        contents=None,
        deployment={"litellm_params": {"model": "hosted_vllm/Qwen/Qwen3-0.6B"}},
        request_model="vllm-counted",
    )

    assert result is None


@pytest.mark.asyncio
async def test_no_messages_uses_the_local_estimate() -> None:
    result: Final = await HostedVLLMTokenCounter().count_tokens(
        model_to_use="Qwen/Qwen3-0.6B",
        messages=None,
        contents=None,
        deployment={"litellm_params": {"api_base": API_BASE}},
        request_model="vllm-counted",
    )

    assert result is None
