import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.replicate.chat.handler import handle_prediction_response_streaming
from litellm.llms.replicate.common_utils import ReplicateError

POLL_URL = "https://api.replicate.com/v1/predictions/test-prediction"
CREATE_URL = "https://api.replicate.com/v1/models/meta/meta-llama-3-8b-instruct/predictions"
MANY_TOKENS = [f"token-{index} " for index in range(500)]


@pytest.fixture(autouse=True)
def _poll_without_waiting(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.llms.replicate.chat.handler.REPLICATE_POLLING_DELAY_SECONDS", 0)


def _poll(*bodies: object) -> list[dict[str, object]]:
    with respx.mock:
        respx.get(POLL_URL).mock(side_effect=[httpx.Response(200, json=body) for body in bodies])
        return list(
            handle_prediction_response_streaming(
                POLL_URL,
                "replicate-test-key",
                lambda _message: None,
                headers={"Authorization": "Bearer replicate-test-key"},
                http_client=HTTPHandler(),
            )
        )


def test_streaming_poll_yields_only_the_text_added_since_the_previous_poll():
    chunks = _poll(
        {"status": "starting"},
        {"status": "processing", "output": ["Hel"]},
        {"status": "processing", "output": ["Hel", "lo"]},
        {"status": "succeeded", "output": ["Hel", "lo", ", world"], "metrics": {"predict_time": 1.5}},
    )

    assert chunks == [
        {"output": "Hel", "status": "processing"},
        {"output": "lo", "status": "processing"},
        {"output": ", world", "status": "succeeded"},
    ]


@pytest.mark.parametrize(
    ("output", "text"),
    [
        (["Hello", ", ", "world"], "Hello, world"),
        ("Hello", "Hello"),
        ({"Hel": 1, "lo": [2]}, "Hello"),
        (["12", "12.0"], "1212.0"),
        ([], ""),
        ("", ""),
        ({}, ""),
        (MANY_TOKENS, "".join(MANY_TOKENS)),
    ],
)
def test_streaming_poll_joins_the_prediction_output_the_way_str_join_does(output: object, text: str):
    assert _poll({"status": "succeeded", "output": output}) == [{"output": text, "status": "succeeded"}]


@pytest.mark.parametrize("status", [None, 1, ["succeeded"], {"state": "succeeded"}])
def test_streaming_poll_passes_an_unrecognised_status_through_and_keeps_polling(status: object):
    chunks = _poll({"status": status, "output": ["a"]}, {"status": "succeeded", "output": ["a", "b"]})

    assert chunks == [{"output": "a", "status": status}, {"output": "b", "status": "succeeded"}]


@pytest.mark.parametrize(
    "output",
    [
        None,
        5,
        0,
        1.5,
        True,
        ["a", 1],
        ["a", None],
        [["a"]],
        [{"a": 1}],
        [*MANY_TOKENS[:403], 403, *MANY_TOKENS[404:]],
    ],
)
def test_streaming_poll_reports_an_output_that_is_not_text_as_unprocessable(output: object):
    with pytest.raises(ReplicateError) as exc_info:
        _poll({"status": "succeeded", "output": output})

    assert exc_info.value.status_code == 422
    assert exc_info.value.message == f"Unable to parse response. Got={output}"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"status": "failed", "error": "boom"}, "Error: boom"),
        ({"status": "failed", "error": None}, "Error: None"),
        ({"status": "failed"}, "Error: "),
        ({"status": "failed", "error": {"detail": "bad input"}}, "Error: {'detail': 'bad input'}"),
    ],
)
def test_streaming_poll_raises_the_provider_error_of_a_failed_prediction(body: object, message: str):
    with pytest.raises(ReplicateError) as exc_info:
        _poll(body)

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == message


@pytest.mark.parametrize("body", [{}, {"output": ["a"]}])
def test_streaming_poll_raises_key_error_when_the_prediction_has_no_status(body: object):
    with pytest.raises(KeyError) as exc_info:
        _poll(body)

    assert exc_info.value.args == ("status",)


@pytest.mark.parametrize(
    "body", [[{"status": "succeeded"}], [], "Request timed out", ["secret-output", 429], 3, True, 1.5]
)
def test_streaming_poll_rejects_a_body_that_is_not_an_object_without_echoing_it(body: object):
    with pytest.raises(ValidationError) as exc_info:
        _poll(body)

    assert [(error["type"], error["loc"]) for error in exc_info.value.errors()] == [("dict_type", ())]
    assert "secret-output" not in str(exc_info.value)
    assert "429" not in str(exc_info.value)
    assert "Request timed out" not in str(exc_info.value)


def test_streaming_poll_keeps_the_chunks_yielded_before_a_malformed_body():
    chunks = []
    with respx.mock:
        respx.get(POLL_URL).mock(
            side_effect=[
                httpx.Response(200, json={"status": "processing", "output": ["a"]}),
                httpx.Response(200, json=["not", "an", "object"]),
            ]
        )
        stream = handle_prediction_response_streaming(
            POLL_URL, "replicate-test-key", lambda _message: None, headers={}, http_client=HTTPHandler()
        )
        chunks.append(next(stream))
        with pytest.raises(ValidationError):
            next(stream)

    assert chunks == [{"output": "a", "status": "processing"}]


def test_streaming_poll_retries_after_a_non_200_status():
    with respx.mock:
        respx.get(POLL_URL).mock(
            side_effect=[
                httpx.Response(503, text="busy"),
                httpx.Response(200, json={"status": "succeeded", "output": ["ok"]}),
            ]
        )
        chunks = list(
            handle_prediction_response_streaming(
                POLL_URL, "replicate-test-key", lambda _message: None, headers={}, http_client=HTTPHandler()
            )
        )

    assert chunks == [{"output": "ok", "status": "succeeded"}]


def _stream_completion(*poll_bodies: object) -> list[str | None]:
    with respx.mock:
        respx.post(CREATE_URL).mock(return_value=httpx.Response(201, json={"urls": {"get": POLL_URL}}))
        respx.get(POLL_URL).mock(side_effect=[httpx.Response(200, json=body) for body in poll_bodies])
        stream = litellm.completion(
            model="replicate/meta/meta-llama-3-8b-instruct",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            api_key="replicate-test-key",
        )
        return [chunk.choices[0].delta.content for chunk in stream]


def test_streaming_completion_emits_each_polled_delta_as_content():
    contents = _stream_completion(
        {"status": "processing", "output": ["Hel"]}, {"status": "succeeded", "output": ["Hel", "lo"]}
    )

    assert contents == ["Hel", "lo", None]


def test_streaming_completion_maps_a_non_text_output_to_unprocessable_entity():
    with pytest.raises(litellm.UnprocessableEntityError) as exc_info:
        _stream_completion({"status": "succeeded", "output": ["a", 1]})

    assert "Unable to parse response. Got=['a', 1]" in str(exc_info.value)


@pytest.mark.parametrize("body", [["x"], "Request timed out", ["Request was throttled", 429]])
def test_streaming_completion_maps_a_non_object_poll_body_to_an_api_error_whatever_text_it_carries(body: object):
    with pytest.raises(MidStreamFallbackError) as exc_info:
        _stream_completion(body)

    assert type(exc_info.value.original_exception) is litellm.APIError
    assert exc_info.value.original_exception.status_code == 500
