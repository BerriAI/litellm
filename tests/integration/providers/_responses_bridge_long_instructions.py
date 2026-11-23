import itertools
import json
import uuid
from collections.abc import Callable, Mapping
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

from integration._support.client import Scenario
from integration._support.wire import Reply, Request
from pydantic import JsonValue, TypeAdapter

# OpenAI Responses API, observed 2026-11-19: one code point more is rejected with HTTP 400 code
# string_above_max_length, "Invalid 'instructions': string too long. Expected a string with maximum length 1048576"
OPENAI_INSTRUCTIONS_MAX_CHARS: Final = 1_048_576

BACKEND: Final = "gpt-6.2-sol"
PROVIDER_KEY: Final = "synthetic-openai-key"
USER_TEXT: Final = "Summarize the policy above."
ANSWER: Final = "Policy summarized."

DeploymentShape: TypeAlias = Literal["responses_prefix", "mode_responses"]
DEPLOYMENT_SHAPES: Final[tuple[DeploymentShape, ...]] = ("responses_prefix", "mode_responses")

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ITEMS: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
_SYSTEM_ROLES: Final = frozenset({"system", "developer"})


def bridged_model(scenario: Scenario, shape: DeploymentShape, wire_url: str) -> str:
    match shape:
        case "responses_prefix":
            return scenario.model(model=f"openai/responses/{BACKEND}", api_base=f"{wire_url}/v1", api_key=PROVIDER_KEY)
        case "mode_responses":
            return scenario.model(
                model=f"openai/{BACKEND}",
                api_base=f"{wire_url}/v1",
                api_key=PROVIDER_KEY,
                model_info={"mode": "responses"},
            )


def _too_long(length: int) -> bytes:
    return json.dumps(
        {
            "error": {
                "message": (
                    "Invalid 'instructions': string too long. Expected a string with maximum length "
                    f"{OPENAI_INSTRUCTIONS_MAX_CHARS}, but got a string with length {length} instead."
                ),
                "type": "invalid_request_error",
                "param": "instructions",
                "code": "string_above_max_length",
            }
        }
    ).encode()


def _completed() -> bytes:
    identity: Final = uuid.uuid4().hex
    return json.dumps(
        {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1789788253,
            "status": "completed",
            "model": BACKEND,
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{identity}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ANSWER, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 41, "output_tokens": 5, "total_tokens": 46},
        }
    ).encode()


def openai_responses_peer() -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        if request.method != "POST" or path != "/v1/responses":
            return Reply(status=404, body=b'{"error":{"message":"not scripted","type":"invalid_request_error"}}')
        instructions: Final = _JSON_OBJECT.validate_json(request.body).get("instructions")
        if isinstance(instructions, str) and len(instructions) > OPENAI_INSTRUCTIONS_MAX_CHARS:
            return Reply(status=400, body=_too_long(len(instructions)))
        return Reply(body=_completed())

    return respond


def posted_bodies(requests: tuple[Request, ...]) -> tuple[dict[str, JsonValue], ...]:
    posted: Final = tuple(request for request in requests if request.method == "POST")
    assert posted, [(request.method, request.target) for request in requests]
    assert all(urlsplit(request.target).path == "/v1/responses" for request in posted), [
        request.target for request in posted
    ]
    return tuple(_JSON_OBJECT.validate_json(request.body) for request in posted)


def _items(body: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    return _ITEMS.validate_python(body.get("input") or ())


def _text(item: Mapping[str, JsonValue]) -> str:
    content: Final = item.get("content")
    if isinstance(content, str):
        return content
    texts: Final = (part.get("text") for part in _ITEMS.validate_python(content or ()))
    return "".join(text for text in texts if isinstance(text, str))


def _is_system(item: Mapping[str, JsonValue]) -> bool:
    return item.get("role") in _SYSTEM_ROLES


def leading_system_text(body: Mapping[str, JsonValue]) -> str:
    return "".join(_text(item) for item in itertools.takewhile(_is_system, _items(body)))


def system_items(body: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(item for item in _items(body) if _is_system(item))


def user_texts(body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    return tuple(_text(item) for item in _items(body) if item.get("role") == "user")


def instructions_length(body: Mapping[str, JsonValue]) -> int | None:
    instructions: Final = body.get("instructions")
    return len(instructions) if isinstance(instructions, str) else None


def wire_shape(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "instructions_length": instructions_length(body),
        "input": [[item.get("role") or item.get("type"), len(_text(item))] for item in _items(body)],
        "truncation": body.get("truncation"),
        "previous_response_id": body.get("previous_response_id"),
    }


def assert_system_text_moved_into_input(body: Mapping[str, JsonValue], system: str) -> None:
    assert "instructions" not in body, wire_shape(body)
    carried_in_full: Final = leading_system_text(body) == system
    assert carried_in_full, wire_shape(body)
    assert user_texts(body) == (USER_TEXT,), wire_shape(body)


def assert_system_text_stays_instructions(body: Mapping[str, JsonValue], system: str) -> None:
    exact: Final = body.get("instructions") == system
    assert exact, wire_shape(body)
    assert_no_system_text_in_input(body)


def assert_no_system_text_in_input(body: Mapping[str, JsonValue]) -> None:
    no_system_items: Final = not system_items(body)
    assert no_system_items, wire_shape(body)
    assert user_texts(body) == (USER_TEXT,), wire_shape(body)
