"""Vendor §9.9: GET /v1/responses/{id} retrieve after store (LIT-4778).

Creates a stored response, retrieves it by id, and pins invalid-id error handling.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import NoBody, Success, UnknownApiError, unwrap
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient

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


class TestResponsesRetrieve:
    @pytest.mark.covers("llm.responses.openai.basic.nonstream.works")
    def test_store_and_retrieve_by_id(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
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
        assert created.object in (None, "response")
        assert created.status in (None, "completed", "in_progress", "queued")

        retrieved = unwrap(
            proxy.transport.get(
                f"/v1/responses/{created.id}",
                headers=proxy.transport.bearer(key),
                params=NoBody(),
                response_type=ResponsesObject,
            )
        )
        assert retrieved.id == created.id

    @pytest.mark.covers("llm.responses.openai.input_validation.nonstream.works")
    def test_invalid_response_id_returns_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-resp-badid-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key = resources.key()
        get_result = proxy.transport.get(
            "/v1/responses/invalid-id",
            headers=proxy.transport.bearer(key),
            params=NoBody(),
            response_type=ResponsesObject,
        )
        match get_result:
            case Success():
                pytest.fail("invalid response id must not succeed")
            case UnknownApiError(status_code=status):
                assert status in (400, 404, 500), (
                    f"invalid id expected 404/500-ish, got {status}"
                )
            case _:
                return
