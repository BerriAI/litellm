"""Vendor §9.9: GET /v1/responses/{id} retrieve after store (LIT-4778).

Creates a stored response, retrieves it by id, and pins invalid-id error handling.
"""

from __future__ import annotations

import time
from typing import Final, Literal

import openai
import pytest
from e2e_config import POLL_INTERVAL, POLL_TIMEOUT, unique_marker
from e2e_http import NoBody, Success, UnknownApiError, unwrap
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from responses_helpers import AZURE_OPENAI_BACKEND, azure_openai_params
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInputMessageItem,
    ResponseInputText,
    ResponseQueuedEvent,
)
from proxy_client import ProxyClient
from pydantic import BaseModel
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

OPENAI_BACKEND: Final = "openai/gpt-5.5"
OPENAI_MINI_BACKEND: Final = "openai/gpt-4o-mini"
LONG_TASK: Final = "Write a numbered list counting from 1 to 400, one number per line, with a short word after each."
CANCELLABLE_STATUSES: Final = frozenset({"queued", "in_progress"})


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
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(OPENAI_MINI_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_store_and_retrieve_by_id(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model = f"e2e-resp-store-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(model=OPENAI_MINI_BACKEND, api_key="os.environ/OPENAI_API_KEY"),
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
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
        )
    )
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


def _retrieve_until_gone(client: openai.OpenAI, response_id: str) -> openai.APIStatusError:
    deadline: Final = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        try:
            client.responses.retrieve(response_id)
        except openai.APIStatusError as error:
            return error
        time.sleep(POLL_INTERVAL)
    raise AssertionError(f"response {response_id!r} was still retrievable {POLL_TIMEOUT}s after delete")


def _register_openai(proxy: ProxyClient, resources: ResourceManager, prefix: str) -> str:
    return _register_response_deployment(proxy, resources, "openai", prefix)


def _register_response_deployment(
    proxy: ProxyClient,
    resources: ResourceManager,
    deployment: Literal["openai", "azure"],
    prefix: str,
) -> str:
    model: Final = f"{prefix}-{unique_marker()}"
    params: Final = (
        azure_openai_params()
        if deployment == "azure"
        else LiteLLMParamsBody(model=OPENAI_BACKEND, api_key="os.environ/OPENAI_API_KEY")
    )
    model_id: Final = proxy.create_model(model, params)
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


def _deployment_param(deployment: Literal["openai", "azure"], provider: Provider, backend: str, mode: Mode) -> object:
    return pytest.param(
        deployment,
        id=deployment,
        marks=meta(
            Subject(
                domain=Domain.LLM_TRANSLATION,
                route=Route.RESPONSES,
                providers=(provider,),
                models=(backend,),
                mode=mode,
            )
        ),
    )


def _input_texts(item: object) -> tuple[str, ...]:
    if not isinstance(item, ResponseInputMessageItem):
        return ()
    return tuple(part.text for part in item.content if isinstance(part, ResponseInputText))


@pytest.mark.provider_live
class TestStoredResponseLifecycle:
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(OPENAI_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_input_items_list_the_stored_prompt(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register_openai(proxy, resources, "e2e-resp-items")
        client = sdk.openai(resources.key())
        marker = unique_marker()

        created = client.responses.create(
            model=model, input=f"Reply with one word. {marker}", store=True, extra_body=NO_PROXY_CACHE
        )
        items = client.responses.input_items.list(created.id, limit=20, order="desc").data

        texts = tuple(text for item in items for text in _input_texts(item))
        assert any(marker in text for text in texts), f"input_items did not list the stored prompt: {items!r}"

    @pytest.mark.parametrize(
        "deployment",
        [
            _deployment_param("openai", Provider.OPENAI, OPENAI_BACKEND, Mode.NONSTREAM),
            _deployment_param("azure", Provider.AZURE, AZURE_OPENAI_BACKEND, Mode.NONSTREAM),
        ],
    )
    def test_deleted_response_is_no_longer_retrievable(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
        deployment: Literal["openai", "azure"],
    ) -> None:
        model: Final = _register_response_deployment(proxy, resources, deployment, "e2e-resp-delete")
        client: Final = sdk.openai(resources.key())

        created: Final = client.responses.create(
            model=model, input=f"Reply with one word. {unique_marker()}", store=True, extra_body=NO_PROXY_CACHE
        )
        retrieved: Final = client.responses.retrieve(created.id)
        assert retrieved.status == "completed", f"stored response not retrievable as completed: {retrieved!r}"
        assert retrieved.output_text == created.output_text, (
            f"retrieved output changed: created={created.output_text!r}, retrieved={retrieved.output_text!r}"
        )

        client.responses.delete(created.id)

        gone: Final = _retrieve_until_gone(client, created.id)
        assert 400 <= gone.status_code < 500, f"retrieve after delete expected a 4xx: {gone!r}"

    @pytest.mark.parametrize(
        "deployment",
        [
            _deployment_param("openai", Provider.OPENAI, OPENAI_BACKEND, Mode.STREAM),
            _deployment_param("azure", Provider.AZURE, AZURE_OPENAI_BACKEND, Mode.STREAM),
        ],
    )
    def test_streamed_response_can_be_deleted(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
        deployment: Literal["openai", "azure"],
    ) -> None:
        model: Final = _register_response_deployment(proxy, resources, deployment, "e2e-resp-delete-stream")
        client: Final = sdk.openai(resources.key())
        events: Final = tuple(
            client.responses.create(
                model=model,
                input=f"Reply with one word. {unique_marker()}",
                store=True,
                stream=True,
                extra_body=NO_PROXY_CACHE,
            )
        )
        completed: Final = next((event for event in events if isinstance(event, ResponseCompletedEvent)), None)
        assert completed is not None, f"stream did not complete: {events!r}"
        response_id: Final = completed.response.id

        client.responses.delete(response_id)
        gone: Final = _retrieve_until_gone(client, response_id)
        assert 400 <= gone.status_code < 500, f"retrieve after streamed delete expected a 4xx: {gone!r}"


@pytest.mark.provider_live
class TestBackgroundResponseCancel:
    @pytest.mark.parametrize(
        "deployment",
        [
            _deployment_param("openai", Provider.OPENAI, OPENAI_BACKEND, Mode.NONSTREAM),
            _deployment_param("azure", Provider.AZURE, AZURE_OPENAI_BACKEND, Mode.NONSTREAM),
        ],
    )
    def test_cancel_background_response(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
        deployment: Literal["openai", "azure"],
    ) -> None:
        model: Final = _register_response_deployment(proxy, resources, deployment, "e2e-resp-cancel")
        client: Final = sdk.openai(resources.key())

        created: Final = client.responses.create(
            model=model, input=f"{LONG_TASK} {unique_marker()}", background=True, extra_body=NO_PROXY_CACHE
        )
        assert created.status in CANCELLABLE_STATUSES, f"background response was not queued: {created.status}"

        cancelled: Final = client.responses.cancel(created.id)
        assert cancelled.status == "cancelled", f"cancel did not stop the response: {cancelled.status}"

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(OPENAI_BACKEND,),
            mode=Mode.STREAM,
        )
    )
    def test_cancel_background_streaming_response_by_streamed_id(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register_openai(proxy, resources, "e2e-resp-cancel-stream")
        client = sdk.openai(resources.key())

        stream = client.responses.create(
            model=model,
            input=f"{LONG_TASK} {unique_marker()}",
            background=True,
            stream=True,
            extra_body=NO_PROXY_CACHE,
        )
        response_id = next(
            (event.response.id for event in stream if isinstance(event, (ResponseCreatedEvent, ResponseQueuedEvent))),
            None,
        )
        stream.close()
        assert response_id, "background stream advertised no response id before the first output"

        cancelled = client.responses.cancel(response_id)
        assert cancelled.status == "cancelled", f"cancel by streamed id did not stop the response: {cancelled.status}"
