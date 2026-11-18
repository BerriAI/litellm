import json
import uuid
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, object_value
from integration._support.wire import Reply, Request, wire_server

_MODEL: Final = "alibaba/wan-3"
_PROMPT: Final = "a cat playing volleyball on a beach"


def _starting_prediction(request: Request) -> Reply:
    prediction_input: Final = object_value(object_value(json.loads(request.body))["input"])
    return Reply(
        status=201,
        body=json.dumps(
            {"id": "pred" + uuid.uuid4().hex, "model": _MODEL, "status": "starting", "input": prediction_input}
        ).encode(),
    )


@pytest.mark.parametrize(
    ("params", "status_code", "upstream_inputs"),
    [
        pytest.param({"seconds": "nan"}, 400, [], id="nan-seconds"),
        pytest.param({"seconds": "inf"}, 400, [], id="inf-seconds"),
        pytest.param({"seconds": "1e999"}, 400, [], id="overflowing-seconds"),
        pytest.param({"seconds": "abc"}, 400, [], id="non-numeric-seconds"),
        pytest.param({"size": "²x²"}, 200, [{"prompt": _PROMPT}], id="superscript-digit-size"),
        pytest.param(
            {"seconds": "5", "size": "1280x720"},
            200,
            [{"prompt": _PROMPT, "duration": 5, "resolution": "720p", "aspect_ratio": "16:9"}],
            id="valid-control",
        ),
    ],
)
def test_replicate_video_create_answers_malformed_seconds_and_size_without_a_server_error(
    gateway: Gateway, params: JSON_OBJECT, status_code: int, upstream_inputs: list[JSON_OBJECT]
) -> None:
    with wire_server(_starting_prediction) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"replicate/{_MODEL}", api_base=wire.url, api_key="synthetic-replicate-key")
        response: Final = gateway.request("POST", "/v1/videos", {"model": model, "prompt": _PROMPT, **params})
        assert response.status_code == status_code, response.text
        assert [
            (request.method, request.target, object_value(json.loads(request.body))["input"])
            for request in wire.drain()
        ] == [("POST", f"/v1/models/{_MODEL}/predictions", upstream_input) for upstream_input in upstream_inputs]
