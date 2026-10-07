import json
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import httpx
import openai
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.sagemaker_embedding import (
    ACCESS_KEY,
    COMPONENT,
    COMPONENT_HEADER,
    EXPECTED_SHAPE,
    IC_NOT_ALLOWED,
    IC_REQUIRED,
    JSON_OBJECT,
    NON_STRING_MODEL_ID,
    PASS_IN_HINT,
    REMOVE_HINT,
    SERVED_MODEL,
    deployment,
    embedding_value,
    endpoint,
    invocations_target,
    openai_deployment,
    openai_route,
    plain_route,
    respond,
)
from integration._support.wire import Request, Wire, wire_server
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(240)

_NO_CACHE: Final[dict[str, JsonValue]] = {"no-cache": True}
_SPEND_ROW: Final = 'SELECT status, call_type, model_group FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s'


def _model_list(health_endpoint: str) -> list[JsonValue]:
    return [
        openai_deployment("sm-openai", "openai"),
        deployment(
            "sm-openai-explicit",
            f"openai/{endpoint('openai')}",
            custom_llm_provider="sagemaker",
            model_id=COMPONENT,
            extra_body={"model": SERVED_MODEL},
        ),
        deployment("sm-openai-bare", openai_route("openai")),
        deployment("sm-openai-no-model-id", openai_route("openai"), extra_body={"model": SERVED_MODEL}),
        openai_deployment("sm-openai-custom-key", "openai", input_type="query"),
        deployment(
            "sm-openai-pinned",
            openai_route("openai"),
            model_id=COMPONENT,
            extra_body={"model": SERVED_MODEL, "encoding_format": "float"},
        ),
        openai_deployment("sm-openai-model-id-int", "openai", model_id=7),
        openai_deployment("sm-openai-model-id-list", "openai", model_id=[COMPONENT]),
        openai_deployment("sm-openai-model-id-empty", "openai", model_id=""),
        openai_deployment("sm-openai-model-id-null", "openai", model_id=None),
        openai_deployment("sm-openai-model-id-long", "openai", model_id="c" * 5120),
        openai_deployment("sm-openai-extra-body-string", "openai", extra_body="not-an-object"),
        openai_deployment("sm-openai-extra-body-empty", "openai", extra_body={}),
        openai_deployment("sm-openai-nousage", "nousage"),
        openai_deployment("sm-openai-nomodel", "nomodel"),
        openai_deployment("sm-openai-noindex", "noindex"),
        openai_deployment("sm-openai-promptonly", "promptonly"),
        openai_deployment("sm-openai-datadict", "datadict"),
        openai_deployment("sm-openai-hfshape", "hfshape"),
        openai_deployment("sm-openai-notjson", "notjson"),
        openai_deployment("sm-openai-err400", "err400"),
        openai_deployment("sm-openai-err424", "err424"),
        openai_deployment("sm-openai-err429", "err429"),
        openai_deployment("sm-openai-err503", "err503"),
        openai_deployment("sm-openai-err500", "err500"),
        openai_deployment("sm-openai-drop", "drop"),
        deployment("sm-openai-unknown", "sagemaker/openai/unknown-embed-endpoint", model_id=COMPONENT),
        openai_deployment("sm-openai-ic", "icrequired"),
        deployment("sm-openai-ic-missing", openai_route("icrequired"), extra_body={"model": SERVED_MODEL}),
        deployment("sm-plain-noic", plain_route("noic"), model_id=COMPONENT),
        deployment("sm-plain-noic-control", plain_route("noic")),
        deployment(
            "sm-openai-health",
            f"sagemaker/openai/{health_endpoint}",
            model_id=COMPONENT,
            extra_body={"model": SERVED_MODEL},
            model_info={"mode": "embedding"},
        ),
        deployment("sm-plain", plain_route("hf"), model_id=COMPONENT),
        deployment("sm-plain-control", plain_route("hf")),
        deployment("sm-plain-drop-params", plain_route("hf"), model_id=COMPONENT, drop_params=True),
        deployment("sm-plain-dict", plain_route("hfdict"), model_id=COMPONENT),
        deployment("sm-plain-err424", plain_route("err424"), model_id=COMPONENT),
        deployment("sm-voyage", plain_route("voyage"), model_id=COMPONENT),
        deployment("sm-cohere", plain_route("cohere"), model_id=COMPONENT),
    ]


def _config(directory: Path, health_endpoint: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = _model_list(health_endpoint)
    path: Final = directory / "sagemaker-embedding.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def rig() -> Iterator[Gateway]:
    with gateway_from_environment() as gateway:
        yield gateway


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(respond) as served:
        yield served


@pytest.fixture(autouse=True)
def _drained_wire(wire: Wire) -> None:
    wire.drain()


@pytest.fixture(scope="module")
def health_endpoint() -> str:
    return endpoint(f"openai-health-{uuid.uuid4().hex[:8]}")


@pytest.fixture(scope="module")
def proxy(
    rig: Gateway, wire: Wire, health_endpoint: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("sagemaker-embedding")
    overrides: Final = {"AWS_ENDPOINT_URL_SAGEMAKER_RUNTIME": wire.url, "AWS_MAX_ATTEMPTS": "2"}
    with owned_proxy(rig, directory, overrides, config=_config(directory, health_endpoint), workers=2) as owned:
        yield owned


def _text() -> str:
    return f"integration embedding {uuid.uuid4().hex}"


def _embed(proxy: Gateway, model: str, text: JsonValue, **fields: JsonValue) -> httpx.Response:
    return proxy.request("POST", "/v1/embeddings", {"model": model, "input": text, "cache": _NO_CACHE, **fields})


def _spend_row(response: httpx.Response) -> dict[str, JsonValue]:
    call_id: Final = response.headers["x-litellm-call-id"]
    rows: Final = eventually(lambda: read_rows(_SPEND_ROW, (call_id,)), lambda found: len(found) == 1, seconds=70)
    return rows[0]


def _usage(payload: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    usage: Final = payload["usage"]
    assert isinstance(usage, dict), payload
    return tuple(usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens"))


def _only_request(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert len(received) == 1, [(request.method, request.target) for request in received]
    return received[0]


def _body(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def _forwarded_body(sent: httpx.Request, text: str) -> dict[str, JsonValue]:
    asked: Final = JSON_OBJECT.validate_json(sent.content)
    encoding: Final = {"encoding_format": asked["encoding_format"]} if "encoding_format" in asked else {}
    return {"input": [text], **encoding, "model": SERVED_MODEL}


def _signed_headers(request: Request) -> tuple[str, ...]:
    authorization: Final = request.headers["authorization"]
    assert authorization.startswith(f"AWS4-HMAC-SHA256 Credential={ACCESS_KEY}/"), authorization
    signed: Final = next(part for part in authorization.split(", ") if part.startswith("SignedHeaders="))
    return tuple(signed.removeprefix("SignedHeaders=").split(";"))


def _sdk_base_url(proxy: Gateway) -> str:
    return f"{str(proxy.client.base_url).rstrip('/')}/v1"


def _sdk(proxy: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=_sdk_base_url(proxy), api_key=proxy.key, max_retries=0, timeout=60)


def _async_sdk(proxy: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=_sdk_base_url(proxy), api_key=proxy.key, max_retries=0, timeout=60)


def test_openai_route_signs_model_id_as_the_inference_component_header_and_sends_the_openai_body(
    proxy: Gateway, wire: Wire
) -> None:
    texts: Final = [_text(), _text()]
    response: Final = _embed(proxy, "sm-openai", texts)
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["data"] == [
        {"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 1.0]},
        {"object": "embedding", "index": 1, "embedding": [0.25, -0.5, 2.0]},
    ], response.text
    assert payload["model"] == "sm-openai", response.text
    assert _usage(payload) == (7, 0, 7), response.text
    request: Final = _only_request(wire)
    assert (request.method, request.target) == ("POST", invocations_target(endpoint("openai"))), request
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert request.headers["x-amzn-sagemaker-custom-attributes"] == "accept_eula=true", dict(request.headers)
    assert request.headers["content-type"] == "application/json", dict(request.headers)
    assert COMPONENT_HEADER in _signed_headers(request), request.headers["authorization"]
    assert _body(request) == {"input": texts, "model": SERVED_MODEL}, request.body
    row: Final = _spend_row(response)
    assert row == {"status": "success", "call_type": "aembedding", "model_group": "sm-openai"}, row


@pytest.mark.parametrize(
    ("path", "model_in_body"),
    [
        ("/v1/embeddings", True),
        ("/embeddings", True),
        ("/openai/deployments/sm-openai/embeddings", False),
        ("/engines/sm-openai/embeddings", False),
    ],
)
def test_every_embeddings_alias_reaches_the_inference_component(
    proxy: Gateway, wire: Wire, path: str, model_in_body: bool
) -> None:
    text: Final = _text()
    body: Final[dict[str, JsonValue]] = {
        "input": text,
        "cache": _NO_CACHE,
        **({"model": "sm-openai"} if model_in_body else {}),
    }
    response: Final = proxy.request("POST", path, body)
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["data"] == [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 1.0]}], response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"input": text, "model": SERVED_MODEL}, request.body
    assert _spend_row(response)["status"] == "success"


def test_openai_sdk_sync_client_reads_the_vectors_in_the_encoding_it_asked_for(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    raw: Final = _sdk(proxy).embeddings.with_raw_response.create(
        model="sm-openai", input=[text], extra_body={"cache": _NO_CACHE}
    )
    created: Final = raw.parse()
    assert [row.embedding for row in created.data] == [[0.25, -0.5, 1.0]], created
    assert created.model == "sm-openai", created
    assert (created.usage.prompt_tokens, created.usage.total_tokens) == (7, 7), created
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == _forwarded_body(raw.http_request, text), request.body
    call_id: Final = raw.headers["x-litellm-call-id"]
    rows: Final = eventually(lambda: read_rows(_SPEND_ROW, (call_id,)), lambda found: len(found) == 1, seconds=70)
    assert rows[0]["status"] == "success", rows


async def test_openai_sdk_async_client_reads_the_vectors_in_the_encoding_it_asked_for(
    proxy: Gateway, wire: Wire
) -> None:
    text: Final = _text()
    raw: Final = await _async_sdk(proxy).embeddings.with_raw_response.create(
        model="sm-openai", input=[text], extra_body={"cache": _NO_CACHE}
    )
    created: Final = raw.parse()
    assert [row.embedding for row in created.data] == [[0.25, -0.5, 1.0]], created
    assert created.model == "sm-openai", created
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == _forwarded_body(raw.http_request, text), request.body
    call_id: Final = raw.headers["x-litellm-call-id"]
    rows: Final = eventually(lambda: read_rows(_SPEND_ROW, (call_id,)), lambda found: len(found) == 1, seconds=70)
    assert rows[0]["status"] == "success", rows


def test_explicit_custom_llm_provider_spelling_takes_the_openai_route(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-openai-explicit", [text])
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["model"] == "sm-openai-explicit", response.text
    request: Final = _only_request(wire)
    assert request.target == f"/endpoints/{endpoint('openai')}/invocations", request
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"input": [text], "model": SERVED_MODEL}, request.body
    assert _spend_row(response)["status"] == "success"


def test_plain_route_signs_model_id_as_the_header_and_keeps_the_hf_body(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-plain", [text])
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["data"] == [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 1.0]}], response.text
    assert payload["model"] == "sm-plain", response.text
    assert _usage(payload) == (3, 0, 3), response.text
    request: Final = _only_request(wire)
    assert request.target == f"/endpoints/{endpoint('hf')}/invocations", request
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert COMPONENT_HEADER in _signed_headers(request), request.headers["authorization"]
    assert _body(request) == {"inputs": [text]}, request.body
    assert _spend_row(response)["status"] == "success"


def test_plain_route_without_model_id_sends_no_inference_component_header(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-plain-control", [text])
    assert response.status_code == 200, response.text
    request: Final = _only_request(wire)
    assert COMPONENT_HEADER not in request.headers, dict(request.headers)
    assert COMPONENT_HEADER not in _signed_headers(request), request.headers["authorization"]
    assert _body(request) == {"inputs": [text]}, request.body
    assert _spend_row(response)["status"] == "success"


def test_plain_route_reads_the_wrapped_hf_embedding_key(proxy: Gateway, wire: Wire) -> None:
    texts: Final = [_text(), _text()]
    response: Final = _embed(proxy, "sm-plain-dict", texts)
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["data"] == [
        {"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 1.0]},
        {"object": "embedding", "index": 1, "embedding": [0.25, -0.5, 2.0]},
    ], response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"inputs": texts}, request.body


def test_voyage_endpoint_keeps_the_voyage_body_and_signs_model_id_as_the_header(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-voyage", [text])
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["data"] == [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 1.0]}], response.text
    assert payload["model"] == "sm-voyage", response.text
    assert _usage(payload) == (9, 0, 9), response.text
    request: Final = _only_request(wire)
    assert request.target == f"/endpoints/{endpoint('voyage')}/invocations", request
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"input": [text], "model": endpoint("voyage")}, request.body
    assert _spend_row(response)["status"] == "success"


def test_cohere_endpoint_keeps_the_cohere_body_and_signs_model_id_as_the_header(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-cohere", [text])
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["data"] == [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 1.0]}], response.text
    assert _usage(payload) == (6, 0, 6), response.text
    request: Final = _only_request(wire)
    assert request.target == f"/endpoints/{endpoint('cohere')}/invocations", request
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"texts": [text], "input_type": "search_document"}, request.body
    assert _spend_row(response)["status"] == "success"


def test_openai_route_forwards_dimensions_encoding_format_and_user_in_the_body(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(
        proxy, "sm-openai", [text], dimensions=3, encoding_format="float", user="integration-embedding-user"
    )
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["data"] == [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 1.0]}], response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {
        "input": [text],
        "dimensions": 3,
        "encoding_format": "float",
        "user": "integration-embedding-user",
        "model": SERVED_MODEL,
    }, request.body


def test_openai_route_without_model_id_or_extra_body_sends_the_bare_openai_body(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-openai-bare", [text])
    assert response.status_code == 200, response.text
    request: Final = _only_request(wire)
    assert COMPONENT_HEADER not in request.headers, dict(request.headers)
    assert _body(request) == {"input": [text]}, request.body


def test_openai_route_sends_a_custom_deployment_key_at_the_top_level_of_the_body(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-openai-custom-key", [text])
    assert response.status_code == 200, response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"input": [text], "input_type": "query", "model": SERVED_MODEL}, request.body


def test_deployment_extra_body_wins_over_the_callers_encoding_format(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    created: Final = _sdk(proxy).embeddings.create(
        model="sm-openai-pinned", input=[text], extra_body={"cache": _NO_CACHE}
    )
    assert [row.embedding for row in created.data] == [[0.25, -0.5, 1.0]], created
    request: Final = _only_request(wire)
    assert _body(request) == {"input": [text], "encoding_format": "float", "model": SERVED_MODEL}, request.body


def test_openai_route_returns_the_base64_string_a_raw_caller_asked_for_unchanged(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-openai", [text], encoding_format="base64")
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["data"] == [{"object": "embedding", "index": 0, "embedding": embedding_value(0, "base64")}], (
        response.text
    )
    request: Final = _only_request(wire)
    assert _body(request) == {"input": [text], "encoding_format": "base64", "model": SERVED_MODEL}, request.body


def test_openai_route_reports_zero_usage_when_the_container_sends_none(proxy: Gateway, wire: Wire) -> None:
    response: Final = _embed(proxy, "sm-openai-nousage", [_text()])
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert _usage(payload) == (0, 0, 0), response.text
    assert _only_request(wire).headers[COMPONENT_HEADER] == COMPONENT
    assert _spend_row(response)["status"] == "success"


def test_openai_route_names_the_endpoint_when_the_container_sends_no_model(proxy: Gateway, wire: Wire) -> None:
    response: Final = _embed(proxy, "sm-openai-nomodel", [_text()])
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["model"] == "sm-openai-nomodel", response.text
    assert _usage(payload) == (7, 0, 7), response.text
    assert _only_request(wire).headers[COMPONENT_HEADER] == COMPONENT


def test_openai_route_numbers_rows_the_container_left_unindexed(proxy: Gateway, wire: Wire) -> None:
    response: Final = _embed(proxy, "sm-openai-noindex", [_text(), _text()])
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["data"] == [
        {"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 1.0]},
        {"object": "embedding", "index": 1, "embedding": [0.25, -0.5, 2.0]},
    ], response.text
    assert _only_request(wire).headers[COMPONENT_HEADER] == COMPONENT


def test_openai_route_takes_total_tokens_from_prompt_tokens_when_the_container_omits_it(
    proxy: Gateway, wire: Wire
) -> None:
    response: Final = _embed(proxy, "sm-openai-promptonly", [_text()])
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert _usage(payload) == (5, 0, 5), response.text
    assert _only_request(wire).headers[COMPONENT_HEADER] == COMPONENT


@pytest.mark.parametrize("model", ["sm-openai-datadict", "sm-openai-hfshape"])
def test_openai_route_rejects_a_container_body_that_is_not_an_openai_embeddings_object(
    proxy: Gateway, wire: Wire, model: str
) -> None:
    response: Final = _embed(proxy, model, [_text()])
    assert response.status_code == 503, response.text
    assert EXPECTED_SHAPE in response.text, response.text
    assert _only_request(wire).headers[COMPONENT_HEADER] == COMPONENT
    assert _spend_row(response)["status"] == "failure"


def test_openai_route_reports_a_non_json_container_body_as_a_server_error(proxy: Gateway, wire: Wire) -> None:
    response: Final = _embed(proxy, "sm-openai-notjson", [_text()])
    assert response.status_code == 500, response.text
    assert "Expecting value" in response.text, response.text
    assert _only_request(wire).headers[COMPONENT_HEADER] == COMPONENT
    assert _spend_row(response)["status"] == "failure"


def test_container_validation_error_reaches_the_caller_as_400(proxy: Gateway, wire: Wire) -> None:
    response: Final = _embed(proxy, "sm-openai-err400", [_text()])
    assert response.status_code == 400, response.text
    assert "1 validation error detected" in response.text, response.text
    assert _only_request(wire).headers[COMPONENT_HEADER] == COMPONENT
    assert _spend_row(response)["status"] == "failure"


def test_container_model_error_reaches_the_caller_with_the_original_message(proxy: Gateway, wire: Wire) -> None:
    response: Final = _embed(proxy, "sm-openai-err424", [_text()])
    assert response.status_code == 424, response.text
    assert "unexpected field inputs" in response.text, response.text
    assert _only_request(wire).headers[COMPONENT_HEADER] == COMPONENT
    assert _spend_row(response)["status"] == "failure"


def test_plain_route_body_rejected_by_an_openai_container_reaches_the_caller(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-plain-err424", [text])
    assert response.status_code == 424, response.text
    assert "unexpected field inputs" in response.text, response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"inputs": [text]}, request.body
    assert _spend_row(response)["status"] == "failure"


@pytest.mark.parametrize(
    ("model", "status", "message"),
    [
        ("sm-openai-err429", 429, "Model is not ready for inference yet"),
        ("sm-openai-err503", 503, "The endpoint is scaling, try again later"),
        ("sm-openai-err500", 503, "An internal failure occurred"),
    ],
)
def test_container_outage_errors_reach_the_caller_and_every_attempt_carries_the_header(
    proxy: Gateway, wire: Wire, model: str, status: int, message: str
) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, model, [text])
    assert response.status_code == status, response.text
    assert message in response.text, response.text
    attempts: Final = wire.drain()
    assert len(attempts) >= 1, attempts
    assert {attempt.headers[COMPONENT_HEADER] for attempt in attempts} == {COMPONENT}, attempts
    assert {attempt.body for attempt in attempts} == {json.dumps({"input": [text], "model": SERVED_MODEL}).encode()}
    assert _spend_row(response)["status"] == "failure"


def test_dropped_connection_reaches_the_caller_as_a_server_error(proxy: Gateway, wire: Wire) -> None:
    response: Final = _embed(proxy, "sm-openai-drop", [_text()])
    assert response.status_code == 503, response.text
    assert "Connection was closed" in response.text, response.text
    attempts: Final = wire.drain()
    assert len(attempts) >= 1, attempts
    assert {attempt.headers[COMPONENT_HEADER] for attempt in attempts} == {COMPONENT}, attempts
    assert _spend_row(response)["status"] == "failure"


def test_inference_component_endpoint_answers_once_the_header_unlocks_it(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-openai-ic", [text])
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    assert payload["data"] == [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 1.0]}], response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"input": [text], "model": SERVED_MODEL}, request.body
    assert _spend_row(response)["status"] == "success"


def test_inference_component_endpoint_without_model_id_tells_the_caller_how_to_pass_it(
    proxy: Gateway, wire: Wire
) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-openai-ic-missing", [text])
    assert response.status_code == 400, response.text
    assert IC_REQUIRED in response.text, response.text
    assert PASS_IN_HINT in response.text, response.text
    request: Final = _only_request(wire)
    assert COMPONENT_HEADER not in request.headers, dict(request.headers)
    assert _body(request) == {"input": [text], "model": SERVED_MODEL}, request.body
    assert _spend_row(response)["status"] == "failure"


def test_endpoint_without_inference_components_tells_the_deployment_to_drop_model_id(
    proxy: Gateway, wire: Wire
) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-plain-noic", [text])
    assert response.status_code == 400, response.text
    assert IC_NOT_ALLOWED in response.text, response.text
    assert REMOVE_HINT in response.text, response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"inputs": [text]}, request.body
    assert _spend_row(response)["status"] == "failure"


def test_endpoint_without_inference_components_serves_a_deployment_without_model_id(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-plain-noic-control", [text])
    assert response.status_code == 200, response.text
    request: Final = _only_request(wire)
    assert COMPONENT_HEADER not in request.headers, dict(request.headers)
    assert _body(request) == {"inputs": [text]}, request.body
    assert _spend_row(response)["status"] == "success"


def test_unknown_endpoint_reaches_the_caller_as_400(proxy: Gateway, wire: Wire) -> None:
    response: Final = _embed(proxy, "sm-openai-unknown", [_text()])
    assert response.status_code == 400, response.text
    assert "Endpoint unknown-embed-endpoint of account 000000000000 not found." in response.text, response.text
    request: Final = _only_request(wire)
    assert request.target == "/endpoints/unknown-embed-endpoint/invocations", request
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _spend_row(response)["status"] == "failure"


def test_plain_route_rejects_an_unsupported_openai_param_without_drop_params(proxy: Gateway, wire: Wire) -> None:
    response: Final = _embed(proxy, "sm-plain", [_text()], dimensions=3)
    assert response.status_code == 400, response.text
    assert "does not support parameters: {'dimensions': 3}" in response.text, response.text
    assert "drop_params" in response.text, response.text
    assert wire.drain() == (), response.text
    assert _spend_row(response)["status"] == "failure"


def test_plain_route_drops_an_unsupported_openai_param_under_drop_params(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-plain-drop-params", [text], dimensions=3)
    assert response.status_code == 200, response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"inputs": [text]}, request.body


@pytest.mark.parametrize("model", ["sm-openai-model-id-int", "sm-openai-model-id-list"])
def test_a_non_string_model_id_fails_before_any_request_leaves_the_proxy(
    proxy: Gateway, wire: Wire, model: str
) -> None:
    response: Final = _embed(proxy, model, [_text()])
    assert response.status_code == 400, response.text
    assert NON_STRING_MODEL_ID in response.text, response.text
    assert wire.drain() == (), response.text
    assert _spend_row(response)["status"] == "failure"


@pytest.mark.parametrize("model", ["sm-openai-model-id-empty", "sm-openai-model-id-null"])
def test_an_empty_or_null_model_id_sends_no_inference_component_header(proxy: Gateway, wire: Wire, model: str) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, model, [text])
    assert response.status_code == 200, response.text
    request: Final = _only_request(wire)
    assert COMPONENT_HEADER not in request.headers, dict(request.headers)
    assert _body(request) == {"input": [text], "model": SERVED_MODEL}, request.body


def test_a_long_model_id_is_signed_and_forwarded_whole(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-openai-model-id-long", [text])
    assert response.status_code == 200, response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == "c" * 5120, len(request.headers[COMPONENT_HEADER])
    assert COMPONENT_HEADER in _signed_headers(request), request.headers["authorization"]
    assert _body(request) == {"input": [text], "model": SERVED_MODEL}, request.body


@pytest.mark.parametrize("model", ["sm-openai-extra-body-string", "sm-openai-extra-body-empty"])
def test_an_extra_body_that_is_not_a_populated_object_adds_nothing_to_the_body(
    proxy: Gateway, wire: Wire, model: str
) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, model, [text])
    assert response.status_code == 200, response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"input": [text]}, request.body


def test_a_caller_sent_model_id_overrides_the_deployments_inference_component(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-openai", [text], model_id="caller-chosen-component")
    assert response.status_code == 200, response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == "caller-chosen-component", dict(request.headers)
    assert _body(request) == {"input": [text], "model": SERVED_MODEL}, request.body


def test_a_caller_sent_model_id_is_forwarded_when_the_deployment_has_none(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-openai-bare", [text], model_id="caller-chosen-component")
    assert response.status_code == 200, response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == "caller-chosen-component", dict(request.headers)
    assert _body(request) == {"input": [text]}, request.body


def test_a_caller_sent_non_string_model_id_fails_before_any_request_leaves_the_proxy(
    proxy: Gateway, wire: Wire
) -> None:
    response: Final = _embed(proxy, "sm-openai-bare", [_text()], model_id=["caller-chosen-component"])
    assert response.status_code == 400, response.text
    assert NON_STRING_MODEL_ID in response.text, response.text
    assert wire.drain() == (), response.text


def test_a_caller_sent_extra_body_replaces_the_deployments_extra_body(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    response: Final = _embed(proxy, "sm-openai", [text], extra_body={"truncate": "END"})
    assert response.status_code == 200, response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"input": [text], "truncate": "END"}, request.body


def test_a_failed_call_does_not_poison_the_next_one(proxy: Gateway, wire: Wire) -> None:
    failed: Final = _embed(proxy, "sm-openai-err503", [_text()])
    assert failed.status_code == 503, failed.text
    wire.drain()
    text: Final = _text()
    response: Final = _embed(proxy, "sm-openai", [text])
    assert response.status_code == 200, response.text
    request: Final = _only_request(wire)
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    assert _body(request) == {"input": [text], "model": SERVED_MODEL}, request.body
    assert _spend_row(response)["status"] == "success"


def test_two_identical_uncached_requests_each_reach_the_container_and_log_once(proxy: Gateway, wire: Wire) -> None:
    text: Final = _text()
    first: Final = _embed(proxy, "sm-openai", [text])
    second: Final = _embed(proxy, "sm-openai", [text])
    assert (first.status_code, second.status_code) == (200, 200), (first.text, second.text)
    assert first.headers["x-litellm-call-id"] != second.headers["x-litellm-call-id"], (first.headers, second.headers)
    requests: Final = wire.drain()
    assert [_body(request) for request in requests] == [{"input": [text], "model": SERVED_MODEL}] * 2, requests
    assert {request.headers[COMPONENT_HEADER] for request in requests} == {COMPONENT}, requests
    assert (_spend_row(first)["status"], _spend_row(second)["status"]) == ("success", "success")


def test_health_check_reaches_the_container_with_the_inference_component_header(
    proxy: Gateway, wire: Wire, health_endpoint: str
) -> None:
    report: Final = proxy.get("/health", {"model": "sm-openai-health"})
    assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
    request: Final = _only_request(wire)
    assert request.target == f"/endpoints/{health_endpoint}/invocations", request
    assert request.headers[COMPONENT_HEADER] == COMPONENT, dict(request.headers)
    body: Final = _body(request)
    assert body["model"] == SERVED_MODEL, request.body
    assert isinstance(body["input"], list) and len(body["input"]) == 1, request.body
