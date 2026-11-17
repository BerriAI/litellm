from collections.abc import Callable, Iterator
from typing import Final, Literal

import httpx
import openai
import pytest
import respx

import litellm
from litellm import Router


@pytest.fixture
def fresh_openai_clients(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


def _answer(reply: int | Literal["refused"]) -> Callable[[httpx.Request], httpx.Response]:
    def answer(request: httpx.Request) -> httpx.Response:
        if reply == "refused":
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(reply, json={"error": {"message": f"upstream {reply}", "type": "x", "code": None}})

    return answer


@pytest.mark.parametrize(
    ("reply", "provider_calls"),
    [(502, 3), (503, 3), (504, 3), ("refused", 1)],
)
def test_internal_server_error_retry_policy_governs_only_errors_the_router_named_internal_server_error(
    fresh_openai_clients: None, reply: int | Literal["refused"], provider_calls: int
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "chat",
                "litellm_params": {"model": "openai/gpt-5.6", "api_key": "sk-fake", "max_retries": 0},
            }
        ],
        num_retries=2,
        retry_after=0,
        retry_policy={"InternalServerErrorRetries": 0},
    )

    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_answer(reply))
        with pytest.raises(openai.APIError):
            router.completion(model="chat", messages=[{"role": "user", "content": "hi"}])

    assert route.call_count == provider_calls
