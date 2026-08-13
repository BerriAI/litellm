"""Vendor §9.9: GET /v1/responses/{id} retrieve after store (LIT-4778).

Creates a stored response, retrieves it by id, and pins invalid-id error handling.
"""

from __future__ import annotations

import time

import pytest
from e2e_config import POLL_INTERVAL, POLL_TIMEOUT, unique_marker
from e2e_http import NoBody, Success, UnknownApiError, unwrap
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from pydantic import BaseModel

pytestmark = pytest.mark.e2e


class ResponsesCreateBody(BaseModel):
    model: str
    input: str
    store: bool = True
    stream: bool = False
    max_output_tokens: int = 64


class ResponsesObject(BaseModel):
    id: str
    object: str | None = None
    status: str | None = None


def _retrieve_response(proxy: ProxyClient, key: str, response_id: str) -> ResponsesObject:
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        result = proxy.transport.get(
            f"/v1/responses/{response_id}",
            headers=proxy.transport.bearer(key),
            params=NoBody(),
            response_type=ResponsesObject,
        )
        match result:
            case Success(data=response):
                return response
            case UnknownApiError(status_code=404):
                time.sleep(POLL_INTERVAL)
            case other:
                raise AssertionError(f"unexpected retrieve result: {other!r}")
    raise AssertionError(f"response {response_id!r} was not retrievable within {POLL_TIMEOUT}s")


class TestResponsesRetrieve:
    @pytest.mark.skip(
        reason="stage red: product gap (LIT-5446), retrieve returns a different id than the stored response (non-idempotent response-id re-encryption)"
    )
    @pytest.mark.covers("llm.responses.openai.basic.nonstream.works")
    def test_store_and_retrieve_by_id(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model = f"e2e-resp-store-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key = resources.key()

        created = unwrap(
            proxy.transport.post(
                "/v1/responses",
                headers=proxy.transport.bearer(key),
                json=ResponsesCreateBody(
                    model=model,
                    input=f"Say pong. {unique_marker()}",
                    store=True,
                ),
                response_type=ResponsesObject,
            )
        )
        assert created.id, f"create returned no id: {created}"
        assert created.object == "response"
        assert created.status == "completed"

        retrieved = _retrieve_response(proxy, key, created.id)
        assert retrieved.id == created.id
        assert retrieved.object == "response"
        assert retrieved.status == "completed"

    @pytest.mark.skip(
        reason="stage red: product gap (LIT-5447), retrieving an unknown response id returns 400 (model=None) instead of 404"
    )
    @pytest.mark.covers("llm.responses.openai.input_validation.nonstream.works")
    def test_invalid_response_id_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        key = resources.key()
        get_result = proxy.transport.get(
            "/v1/responses/resp_00000000000000000000000000000000",
            headers=proxy.transport.bearer(key),
            params=NoBody(),
            response_type=ResponsesObject,
        )
        match get_result:
            case Success():
                pytest.fail("invalid response id must not succeed")
            case UnknownApiError(status_code=404):
                return
            case other:
                pytest.fail(f"invalid response id expected 404, got {other!r}")
