import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_API_KEY: Final = "synthetic-decisions-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_INPUT: Final = "Listen to the attached clip."
_AUDIO: Final = ("data:audio/wav;base64,UklGRiQAAABXQVZF",)
_VIDEOS: Final = ("data:video/mp4;base64,AAAAIGZ0eXBpc29t",)
_QUESTION: Final[dict[str, JsonValue]] = {
    "type": "predicate",
    "name": "banana",
    "instructions": "Does the clip name a banana?",
}
_SPEND_QUERY: Final = 'SELECT status, model_group FROM "LiteLLM_SpendLogs" WHERE request_id = %s'

Route = Literal["/v1/decisions", "/v1/systemone"]
Media = Literal["audio", "videos"]


@dataclass(frozen=True, slots=True)
class _Upstream:
    model: str

    def reply(self, request: Request) -> Reply:
        questions: Final = object_value(_JSON_OBJECT.validate_json(request.body)["questions"])
        plain: Final[dict[str, JsonValue]] = {
            "model": self.model,
            "answers": {name: {"type": "noul", "noul": 0.9} for name in questions},
            "usage": {"input_tokens": 41, "output_tokens": 0},
        }
        return Reply(body=json.dumps(plain).encode())


def _unlisted_openrouter_model() -> _Upstream:
    return _Upstream(f"openrouter/example/omni-{uuid.uuid4().hex[:12]}")


def _decide(gateway: Gateway, model: str, route: Route, media: Mapping[str, JsonValue]) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = (
        {"model": model, "input": _INPUT, "questions": [_QUESTION], "num_retries": 0, **media}
        if route == "/v1/decisions"
        else {
            "model": model,
            "state": _INPUT,
            "questions": {"banana": {"type": "noul", "instructions": _QUESTION["instructions"]}},
            "num_retries": 0,
            **media,
        }
    )
    return gateway.request("POST", route, body)


def _sent_media(request: Request) -> dict[str, JsonValue]:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    return {kind: body[kind] for kind in ("audio", "videos") if kind in body}


def _assert_refused(response: httpx.Response, kinds: tuple[Media, ...]) -> None:
    assert response.status_code == 400, response.text
    assert f"does not support parameters: {list(kinds)}" in response.text, response.text


_BOTH: Final[dict[str, JsonValue]] = {"audio": list(_AUDIO), "videos": list(_VIDEOS)}


@pytest.mark.parametrize("route", ("/v1/decisions", "/v1/systemone"))
def test_a_deployment_flagged_for_audio_and_video_sends_both_clips_for_a_model_the_cost_map_does_not_list(
    gateway: Gateway, route: Route
) -> None:
    upstream: Final = _unlisted_openrouter_model()
    with wire_server(upstream.reply) as wire, gateway.scenario() as scenario:
        flags: Final[dict[str, JsonValue]] = {"supports_audio_input": True, "supports_video_input": True}
        flagged: Final = scenario.model(model=upstream.model, api_base=wire.url, api_key=_API_KEY, model_info=flags)
        flagged_dropping: Final = scenario.model(
            model=upstream.model, api_base=wire.url, api_key=_API_KEY, model_info=flags, drop_params=True
        )
        audio_only: Final = scenario.model(
            model=upstream.model, api_base=wire.url, api_key=_API_KEY, model_info={"supports_audio_input": True}
        )
        unflagged: Final = scenario.model(model=upstream.model, api_base=wire.url, api_key=_API_KEY)

        _assert_refused(_decide(gateway, unflagged, route, _BOTH), ("audio", "videos"))
        _assert_refused(_decide(gateway, audio_only, route, _BOTH), ("videos",))
        assert wire.drain() == ()

        answered: Final = _decide(gateway, flagged, route, _BOTH)
        assert answered.status_code == 200, answered.text
        (call,) = wire.drain()
        assert _sent_media(call) == _BOTH
        rows: Final = eventually(
            lambda: read_rows(_SPEND_QUERY, (answered.headers["x-litellm-call-id"],)),
            lambda found: len(found) == 1,
            seconds=70,
        )
        assert (rows[0]["status"], rows[0]["model_group"]) == ("success", flagged)

        kept: Final = _decide(gateway, flagged_dropping, route, _BOTH)
        assert kept.status_code == 200, kept.text
        (kept_call,) = wire.drain()
        assert _sent_media(kept_call) == _BOTH


@pytest.mark.parametrize(
    "value",
    ({"supports_audio_input": True, "supports_video_input": True}, 5, [{"supports_audio_input": True}], "x" * 5000),
    ids=("flags", "int", "list", "5kb"),
)
def test_a_caller_sent_model_info_never_turns_media_on(gateway: Gateway, value: JsonValue) -> None:
    upstream: Final = _unlisted_openrouter_model()
    with wire_server(upstream.reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=upstream.model, api_base=wire.url, api_key=_API_KEY)
        _assert_refused(_decide(gateway, model, "/v1/decisions", {**_BOTH, "model_info": value}), ("audio", "videos"))
        assert wire.drain() == ()
