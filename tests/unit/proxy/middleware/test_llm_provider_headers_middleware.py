import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from litellm.proxy.middleware.llm_provider_headers_middleware import LLMProviderHeadersMiddleware

PROVIDER_HEADERS = {"llm_provider-x-upstream-debug": "marker", "llm_provider-retry-after": "7"}
OWN_HEADERS = {"x-litellm-call-id": "call-1", "retry-after": "2"}


def _client(include: bool, streaming: bool, status_code: int) -> TestClient:
    headers = {**PROVIDER_HEADERS, **OWN_HEADERS}

    async def reply(request: Request) -> Response:
        if streaming:
            return StreamingResponse(iter([b"data: a\n\n", b"data: b\n\n"]), headers=headers, status_code=status_code)
        return JSONResponse({"ok": True}, headers=headers, status_code=status_code)

    app = Starlette(routes=[Route("/v1/chat/completions", reply, methods=["POST"])])
    app.add_middleware(LLMProviderHeadersMiddleware, include=lambda: include)
    return TestClient(app)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status_code", [200, 429])
def test_opt_out_drops_every_provider_header_and_keeps_the_rest(streaming: bool, status_code: int) -> None:
    response = _client(include=False, streaming=streaming, status_code=status_code).post("/v1/chat/completions")

    assert response.status_code == status_code
    assert [name for name in response.headers.keys() if name.lower().startswith("llm_provider-")] == []
    assert {name: response.headers[name] for name in OWN_HEADERS} == OWN_HEADERS
    assert response.content == (b"data: a\n\ndata: b\n\n" if streaming else b'{"ok":true}')


@pytest.mark.parametrize("streaming", [False, True])
def test_provider_headers_pass_through_by_default(streaming: bool) -> None:
    response = _client(include=True, streaming=streaming, status_code=200).post("/v1/chat/completions")

    assert {name: response.headers[name] for name in PROVIDER_HEADERS} == PROVIDER_HEADERS
