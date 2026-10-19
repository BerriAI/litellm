import json
from datetime import datetime
from typing import Final

import google.auth
import google.auth.credentials
import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.vertex_ai.vertex_embeddings.embedding_handler import VertexEmbedding
from litellm.types.utils import EmbeddingResponse

GECKO: Final = "textembedding-gecko"
FINE_TUNED_ENDPOINT: Final = "1234567890"
BGE: Final = "bge-small-en-v1.5"
PUBLIC_CALL_PROJECT: Final = "vertex-embedding-handler-tests"
BOTH_CALL_STYLES: Final = pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
EVERY_MODEL_FAMILY: Final = pytest.mark.parametrize("model", [GECKO, FINE_TUNED_ENDPOINT, BGE])
BODIES_THAT_ARE_NOT_OBJECTS: Final = [
    [],
    [{"predictions": []}],
    ["predictions"],
    "predictions",
    "",
    7,
    403,
    2.5,
    True,
    None,
    ["vertex-secret"],
]


class _StaticGoogleCredentials(google.auth.credentials.Credentials):
    def refresh(self, request: object) -> None:
        self.token = "test-token"


@pytest.fixture
def predict_endpoint(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> respx.Route:
    monkeypatch.setattr(google.auth, "default", lambda scopes: (_StaticGoogleCredentials(), "test-project"))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    return respx_mock.post(host="us-central1-aiplatform.googleapis.com")


def _predict_response(payload: object) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode())


def _embedding_logging() -> Logging:
    return Logging(
        model=GECKO,
        messages=[],
        stream=False,
        call_type="embedding",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="embedding-call",
        function_id="embedding-function",
    )


def _gecko_prediction(values: object, token_count: int) -> dict[str, object]:
    return {"embeddings": {"values": values, "statistics": {"token_count": token_count, "truncated": False}}}


def _embedding_rows(vectors: list[object]) -> list[dict[str, object]]:
    return [{"object": "embedding", "index": index, "embedding": vector} for index, vector in enumerate(vectors)]


async def _embedding_response(
    run_async: bool, endpoint: respx.Route, model: str, payload: object, logging_obj: Logging
) -> EmbeddingResponse:
    endpoint.mock(return_value=_predict_response(payload))
    pending: Final = VertexEmbedding().embedding(
        model=model,
        input=["hello", "world"],
        print_verbose=None,
        model_response=EmbeddingResponse(),
        optional_params={},
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai",
        timeout=10.0,
        aembedding=run_async,
        vertex_project="test-project",
        vertex_location="us-central1",
        vertex_credentials=None,
        litellm_params={},
    )
    return await pending if run_async else pending


@BOTH_CALL_STYLES
@pytest.mark.parametrize(
    ("model", "payload", "prompt_tokens"),
    [
        (
            GECKO,
            {
                "predictions": [_gecko_prediction([0.1, 0.2, 0.3], 3), _gecko_prediction([1, 2.0, 3], 5)],
                "metadata": {"billableCharacterCount": 10},
            },
            8,
        ),
        (FINE_TUNED_ENDPOINT, {"predictions": [[[0.1, 0.2, 0.3]], [[1, 2.0, 3]]], "deployedModelId": "42"}, 0),
        (BGE, {"predictions": [[0.1, 0.2, 0.3], [1, 2.0, 3]], "deployedModelId": "42"}, 0),
    ],
)
async def test_vertex_embedding_returns_each_prediction_vector_exactly_as_sent(
    run_async: bool, predict_endpoint: respx.Route, model: str, payload: object, prompt_tokens: int
) -> None:
    response: Final = await _embedding_response(run_async, predict_endpoint, model, payload, _embedding_logging())

    assert (
        json.dumps(response.data),
        response.object,
        response.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        response.usage.total_tokens,
    ) == (
        json.dumps(_embedding_rows([[0.1, 0.2, 0.3], [1, 2.0, 3]])),
        "list",
        model,
        prompt_tokens,
        0,
        prompt_tokens,
    )


@BOTH_CALL_STYLES
@pytest.mark.parametrize("values", [[], "not-a-vector", None, [0.5, "x", None, True, {"a": 1}]])
async def test_vertex_embedding_keeps_prediction_values_that_are_not_float_vectors(
    run_async: bool, predict_endpoint: respx.Route, values: object
) -> None:
    response: Final = await _embedding_response(
        run_async, predict_endpoint, GECKO, {"predictions": [_gecko_prediction(values, 3)]}, _embedding_logging()
    )

    assert json.dumps(response.data) == json.dumps(_embedding_rows([values]))


@BOTH_CALL_STYLES
async def test_vertex_embedding_returns_every_vector_of_a_large_batch_in_order(
    run_async: bool, predict_endpoint: respx.Route
) -> None:
    vectors: Final[list[object]] = [[index, index + 0.5] for index in range(500)]

    response: Final = await _embedding_response(
        run_async, predict_endpoint, BGE, {"predictions": vectors}, _embedding_logging()
    )

    assert json.dumps(response.data) == json.dumps(_embedding_rows(vectors))


@BOTH_CALL_STYLES
async def test_vertex_embedding_logs_the_predict_body_as_json_text(
    run_async: bool, predict_endpoint: respx.Route
) -> None:
    payload: Final = {"predictions": [[0.1, 0.2]], "deployedModelId": "42"}
    logging_obj: Final = _embedding_logging()

    await _embedding_response(run_async, predict_endpoint, BGE, payload, logging_obj)

    assert logging_obj.model_call_details["original_response"] == json.dumps(payload)


@BOTH_CALL_STYLES
@EVERY_MODEL_FAMILY
@pytest.mark.parametrize("body", BODIES_THAT_ARE_NOT_OBJECTS)
async def test_vertex_embedding_rejects_a_predict_body_that_is_not_an_object_without_echoing_it(
    run_async: bool, predict_endpoint: respx.Route, model: str, body: object
) -> None:
    with pytest.raises(ValidationError) as rejection:
        await _embedding_response(run_async, predict_endpoint, model, body, _embedding_logging())

    rejection_text: Final = str(rejection.value)
    assert (
        [error["type"] for error in rejection.value.errors()],
        "vertex-secret" in rejection_text,
        "403" in rejection_text,
    ) == (["dict_type"], False, False), rejection_text


@BOTH_CALL_STYLES
async def test_vertex_embedding_logs_a_predict_body_that_is_not_an_object_before_rejecting_it(
    run_async: bool, predict_endpoint: respx.Route
) -> None:
    logging_obj: Final = _embedding_logging()

    with pytest.raises(ValidationError):
        await _embedding_response(run_async, predict_endpoint, GECKO, ["not", "an", "object"], logging_obj)

    assert logging_obj.model_call_details["original_response"] == ["not", "an", "object"]


@BOTH_CALL_STYLES
@EVERY_MODEL_FAMILY
@pytest.mark.parametrize(
    "payload",
    [{}, {"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}}],
)
async def test_vertex_embedding_reports_a_predict_object_without_predictions_as_a_key_error(
    run_async: bool, predict_endpoint: respx.Route, model: str, payload: object
) -> None:
    with pytest.raises(KeyError, match="predictions"):
        await _embedding_response(run_async, predict_endpoint, model, payload, _embedding_logging())


def test_vertex_embedding_call_returns_the_prediction_vectors(predict_endpoint: respx.Route) -> None:
    predict_endpoint.mock(
        return_value=_predict_response({"predictions": [_gecko_prediction([0.1, 2], 3), _gecko_prediction([3.5], 4)]})
    )

    response: Final = litellm.embedding(
        model=f"vertex_ai/{GECKO}",
        input=["hello", "world"],
        vertex_project=PUBLIC_CALL_PROJECT,
        vertex_location="us-central1",
    )

    assert (json.dumps(response.data), response.usage.prompt_tokens) == (
        json.dumps(_embedding_rows([[0.1, 2], [3.5]])),
        7,
    )


async def test_vertex_async_embedding_call_returns_the_prediction_vectors(predict_endpoint: respx.Route) -> None:
    predict_endpoint.mock(
        return_value=_predict_response({"predictions": [_gecko_prediction([0.1, 2], 3), _gecko_prediction([3.5], 4)]})
    )

    response: Final = await litellm.aembedding(
        model=f"vertex_ai/{GECKO}",
        input=["hello", "world"],
        vertex_project=PUBLIC_CALL_PROJECT,
        vertex_location="us-central1",
    )

    assert (json.dumps(response.data), response.usage.prompt_tokens) == (
        json.dumps(_embedding_rows([[0.1, 2], [3.5]])),
        7,
    )


MALFORMED_PREDICT_BODIES: Final = [
    ["vertex-secret"],
    "predictions",
    403,
    None,
    {"predictions": [_gecko_prediction([0.5], 1) for _ in range(403)] + [None]},
]


@pytest.mark.parametrize("body", MALFORMED_PREDICT_BODIES)
def test_vertex_embedding_call_maps_a_malformed_predict_body_to_a_connection_error_without_echoing_it(
    predict_endpoint: respx.Route, body: object
) -> None:
    predict_endpoint.mock(return_value=_predict_response(body))

    with pytest.raises(litellm.APIConnectionError) as failure:
        litellm.embedding(
            model=f"vertex_ai/{GECKO}",
            input=["hello", "world"],
            vertex_project=PUBLIC_CALL_PROJECT,
            vertex_location="us-central1",
        )

    assert (failure.value.status_code, "vertex-secret" in str(failure.value)) == (500, False)


@pytest.mark.parametrize("body", MALFORMED_PREDICT_BODIES)
async def test_vertex_async_embedding_call_maps_a_malformed_predict_body_to_a_connection_error_without_echoing_it(
    predict_endpoint: respx.Route, body: object
) -> None:
    predict_endpoint.mock(return_value=_predict_response(body))

    with pytest.raises(litellm.APIConnectionError) as failure:
        await litellm.aembedding(
            model=f"vertex_ai/{GECKO}",
            input=["hello", "world"],
            vertex_project=PUBLIC_CALL_PROJECT,
            vertex_location="us-central1",
        )

    assert (failure.value.status_code, "vertex-secret" in str(failure.value)) == (500, False)
