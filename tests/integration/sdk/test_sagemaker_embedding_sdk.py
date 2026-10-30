import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.sagemaker_embedding import (
    ACCESS_KEY,
    COMPONENT,
    COMPONENT_HEADER,
    JSON_OBJECT,
    NON_STRING_MODEL_ID,
    REGION,
    SECRET_KEY,
    SERVED_MODEL,
    endpoint,
    invocations_target,
    litellm_params,
    openai_route,
    plain_route,
    respond,
)
from integration._support.wire import Request, Wire, wire_server
from pydantic import JsonValue

import litellm

_AWS: Final[dict[str, str]] = {
    "aws_access_key_id": ACCESS_KEY,
    "aws_secret_access_key": SECRET_KEY,
    "aws_region_name": REGION,
}


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(respond) as served:
        yield served


@pytest.fixture(autouse=True)
def _runtime_at_the_peer(wire: Wire, monkeypatch: pytest.MonkeyPatch) -> None:
    wire.drain()
    monkeypatch.setenv("AWS_ENDPOINT_URL_SAGEMAKER_RUNTIME", wire.url)
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")


def _text() -> str:
    return f"integration embedding {uuid.uuid4().hex}"


def _only_request(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert len(received) == 1, [(request.method, request.target) for request in received]
    return received[0]


def _signed_headers(request: Request) -> tuple[str, ...]:
    authorization: Final = request.headers["authorization"]
    assert authorization.startswith(f"AWS4-HMAC-SHA256 Credential={ACCESS_KEY}/"), authorization
    signed: Final = next(part for part in authorization.split(", ") if part.startswith("SignedHeaders="))
    return tuple(signed.removeprefix("SignedHeaders=").split(";"))


def _assert_component_request(request: Request, scenario: str, body: dict[str, JsonValue]) -> None:
    assert (request.method, request.target) == ("POST", invocations_target(endpoint(scenario))), request
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert COMPONENT_HEADER in _signed_headers(request), request.headers["authorization"]
    assert JSON_OBJECT.validate_json(request.body) == body, request.body


def _vectors(response: litellm.EmbeddingResponse) -> list[JsonValue]:
    return [row["embedding"] for row in response.data]


def test_embedding_signs_model_id_as_the_inference_component_and_reads_the_served_model(wire: Wire) -> None:
    text: Final = _text()
    response: Final = litellm.embedding(
        model=openai_route("openai"), input=[text], model_id=COMPONENT, extra_body={"model": SERVED_MODEL}, **_AWS
    )
    assert response.model == SERVED_MODEL, response
    assert _vectors(response) == [[0.25, -0.5, 1.0]], response
    assert (response.usage.prompt_tokens, response.usage.total_tokens) == (7, 7), response.usage
    _assert_component_request(_only_request(wire), "openai", {"input": [text], "model": SERVED_MODEL})


async def test_aembedding_signs_model_id_as_the_inference_component_and_reads_the_served_model(wire: Wire) -> None:
    texts: Final = [_text(), _text()]
    response: Final = await litellm.aembedding(
        model=openai_route("openai"), input=texts, model_id=COMPONENT, extra_body={"model": SERVED_MODEL}, **_AWS
    )
    assert response.model == SERVED_MODEL, response
    assert _vectors(response) == [[0.25, -0.5, 1.0], [0.25, -0.5, 2.0]], response
    _assert_component_request(_only_request(wire), "openai", {"input": texts, "model": SERVED_MODEL})


def test_embedding_names_the_endpoint_when_the_container_reports_no_model(wire: Wire) -> None:
    text: Final = _text()
    response: Final = litellm.embedding(model=openai_route("nomodel"), input=[text], model_id=COMPONENT, **_AWS)
    assert response.model == endpoint("nomodel"), response
    assert _vectors(response) == [[0.25, -0.5, 1.0]], response
    _assert_component_request(_only_request(wire), "nomodel", {"input": [text]})


def test_plain_route_keeps_the_hf_body_and_signs_the_component_header(wire: Wire) -> None:
    text: Final = _text()
    response: Final = litellm.embedding(model=plain_route("hf"), input=[text], model_id=COMPONENT, **_AWS)
    assert response.model == endpoint("hf"), response
    assert _vectors(response) == [[0.25, -0.5, 1.0]], response
    assert response.usage.prompt_tokens == 3, response.usage
    _assert_component_request(_only_request(wire), "hf", {"inputs": [text]})


def test_plain_route_without_model_id_sends_no_component_header(wire: Wire) -> None:
    text: Final = _text()
    response: Final = litellm.embedding(model=plain_route("hf"), input=[text], **_AWS)
    assert _vectors(response) == [[0.25, -0.5, 1.0]], response
    request: Final = _only_request(wire)
    assert (request.method, request.target) == ("POST", invocations_target(endpoint("hf"))), request
    assert COMPONENT_HEADER not in request.headers, dict(request.headers)
    assert COMPONENT_HEADER not in _signed_headers(request), request.headers["authorization"]
    assert JSON_OBJECT.validate_json(request.body) == {"inputs": [text]}, request.body


def test_explicit_provider_spelling_signs_model_id_as_the_inference_component(wire: Wire) -> None:
    text: Final = _text()
    response: Final = litellm.embedding(
        model=f"openai/{endpoint('openai')}",
        custom_llm_provider="sagemaker",
        input=[text],
        model_id=COMPONENT,
        extra_body={"model": SERVED_MODEL},
        **_AWS,
    )
    assert response.model == SERVED_MODEL, response
    assert _vectors(response) == [[0.25, -0.5, 1.0]], response
    _assert_component_request(_only_request(wire), "openai", {"input": [text], "model": SERVED_MODEL})


async def test_router_aembedding_forwards_the_deployment_model_id_as_the_component(wire: Wire) -> None:
    text: Final = _text()
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "sm-openai-router",
                "litellm_params": litellm_params(
                    openai_route("openai"), model_id=COMPONENT, extra_body={"model": SERVED_MODEL}
                ),
            }
        ],
        num_retries=0,
    )
    response: Final = await router.aembedding(model="sm-openai-router", input=[text])
    assert response.model == SERVED_MODEL, response
    assert _vectors(response) == [[0.25, -0.5, 1.0]], response
    _assert_component_request(_only_request(wire), "openai", {"input": [text], "model": SERVED_MODEL})


def test_embedding_rejects_a_non_string_model_id_before_calling_aws(wire: Wire) -> None:
    with pytest.raises(litellm.BadRequestError) as caught:
        litellm.embedding(model=openai_route("openai"), input=[_text()], model_id=[COMPONENT], **_AWS)
    assert caught.value.status_code == 400, caught.value
    assert NON_STRING_MODEL_ID in str(caught.value), caught.value
    assert wire.drain() == (), "AWS was called with a list-valued model_id"


def test_embedding_surfaces_the_container_model_error_with_its_status(wire: Wire) -> None:
    with pytest.raises(litellm.BadRequestError) as caught:
        litellm.embedding(model=openai_route("err424"), input=[_text()], model_id=COMPONENT, **_AWS)
    assert caught.value.status_code == 424, caught.value
    assert "unexpected field inputs" in str(caught.value), caught.value
    assert _only_request(wire).headers[COMPONENT_HEADER] == COMPONENT
