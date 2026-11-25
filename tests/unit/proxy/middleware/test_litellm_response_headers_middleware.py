from typing import Final

import pytest
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from litellm.proxy.middleware.litellm_response_headers_middleware import LiteLLMResponseHeadersMiddleware


@pytest.mark.parametrize("include", [True, False])
@pytest.mark.parametrize("status_code", [200, 400])
@pytest.mark.parametrize("streaming", [True, False])
def test_header_opt_out_preserves_body_and_other_headers(include: bool, status_code: int, streaming: bool) -> None:
    app: Final = FastAPI()
    app.add_middleware(LiteLLMResponseHeadersMiddleware, include_litellm_headers=lambda: include)

    @app.get("/reply")
    def reply() -> Response:
        headers: Final = {"X-LiteLLM-Call-ID": "call", "x-litellm-priority": "1", "retry-after": "2"}
        if streaming:
            return StreamingResponse(iter([b"hello", b" world"]), headers=headers, status_code=status_code)
        return Response(b"hello world", headers=headers, status_code=status_code)

    with TestClient(app) as client:
        response: Final = client.get("/reply")
    assert response.status_code == status_code
    assert response.content == b"hello world"
    assert response.headers["retry-after"] == "2"
    actual: Final = {k: v for k, v in response.headers.items() if k.startswith("x-litellm-")}
    assert actual == ({"x-litellm-call-id": "call", "x-litellm-priority": "1"} if include else {})
