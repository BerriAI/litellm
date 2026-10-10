import json
import uuid
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI
from pydantic import BaseModel, JsonValue, TypeAdapter

_TEXT_BACKEND: Final = "gpt-3.5-turbo-instruct"
_CHAT_BACKEND: Final = "gpt-5.4-mini"
_API_KEY: Final = "synthetic-openai-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SPEND_ROW: Final = 'SELECT request_tags, metadata FROM "LiteLLM_SpendLogs" WHERE request_id = %s'


class _TextChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class _TextCompletion(BaseModel):
    id: str
    object: str
    model: str
    choices: list[_TextChoice]


def _text_completion(identity: str, texts: tuple[str, ...]) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "text_completion",
            "created": 1,
            "model": _TEXT_BACKEND,
            "choices": [
                {"index": index, "text": text, "finish_reason": "stop", "logprobs": None}
                for index, text in enumerate(texts)
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
    ).encode()


def _chat_completion(identity: str, content: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": _CHAT_BACKEND,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
    ).encode()


def _openai(gateway: Gateway, key: str) -> OpenAI:
    return OpenAI(
        api_key=key,
        base_url=str(gateway.client.base_url).rstrip("/") + "/v1",
        max_retries=0,
        http_client=httpx.Client(timeout=15, trust_env=False),
    )


def _form_fields(fields: dict[str, str]) -> list[tuple[str, tuple[None, str]]]:
    return [(name, (None, value)) for name, value in fields.items()]


def _assert_text_completion_request(request: Request, prompt: str) -> None:
    assert (request.method, request.target) == ("POST", "/completions"), request
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    assert request.headers["content-type"] == "application/json"
    assert _JSON_OBJECT.validate_json(request.body) == {"model": _TEXT_BACKEND, "prompt": prompt}


def _assert_spend_row_carries_client_tag_and_key_identity(
    request_id: str, key: str, user_id: str, team_id: str
) -> None:
    rows: Final = eventually(lambda: read_rows(_SPEND_ROW, (request_id,)), lambda values: len(values) == 1, seconds=70)
    tags: Final = rows[0]["request_tags"]
    metadata: Final = rows[0]["metadata"]
    assert isinstance(tags, list) and isinstance(metadata, dict), rows
    assert "t1" in tags, rows
    assert (
        metadata["user_api_key"],
        metadata["user_api_key_user_id"],
        metadata["user_api_key_team_id"],
    ) == (sha256(key.encode()).hexdigest(), user_id, team_id), metadata


def test_text_completion_json_string_metadata_reaches_upstream_and_tags_spend(gateway: Gateway) -> None:
    pytest.skip("BUG: /v1/completions with metadata sent as a JSON string returns 500 before dispatch")
    identity: Final = f"cmpl-{uuid.uuid4().hex}"
    prompt: Final = f"tagged prompt {identity}"

    def respond(request: Request) -> Reply:
        _assert_text_completion_request(request, prompt)
        return Reply(body=_text_completion(identity, ("tagged",)))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_TEXT_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        team: Final = scenario.team()
        user_id: Final = f"integration-{uuid.uuid4().hex}"
        key: Final = scenario.key(models=[model], team_id=team, user_id=user_id)
        with _openai(gateway, key) as client:
            completion: Final = client.completions.create(
                model=model, prompt=prompt, extra_body={"metadata": json.dumps({"tags": ["t1"]})}
            )
        assert [(choice.index, choice.text) for choice in completion.choices] == [(0, "tagged")], completion
        assert completion.id == identity, completion
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/completions")]
        _assert_spend_row_carries_client_tag_and_key_identity(identity, key, user_id, team)


def test_text_completion_form_metadata_reaches_upstream_and_tags_spend(gateway: Gateway) -> None:
    identity: Final = f"cmpl-{uuid.uuid4().hex}"
    prompt: Final = f"tagged prompt {identity}"

    def respond(request: Request) -> Reply:
        _assert_text_completion_request(request, prompt)
        return Reply(body=_text_completion(identity, ("tagged",)))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_TEXT_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        team: Final = scenario.team()
        user_id: Final = f"integration-{uuid.uuid4().hex}"
        key: Final = scenario.key(models=[model], team_id=team, user_id=user_id)
        response: Final = gateway.client.post(
            "/v1/completions",
            files=_form_fields({"model": model, "prompt": prompt, "metadata": json.dumps({"tags": ["t1"]})}),
            headers={"Authorization": f"Bearer {key}"},
        )
        assert response.status_code == 200, response.text
        assert response.request.headers["content-type"].startswith("multipart/form-data"), response.request.headers
        payload: Final = _TextCompletion.model_validate_json(response.content)
        assert (payload.id, payload.object, payload.model) == (identity, "text_completion", model), response.text
        assert [(choice.index, choice.text) for choice in payload.choices] == [(0, "tagged")], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/completions")]
        _assert_spend_row_carries_client_tag_and_key_identity(identity, key, user_id, team)


def test_text_completion_typed_form_fields_reach_upstream_as_json_types(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: multipart /v1/completions forwards temperature, max_tokens, n and echo upstream as strings, "
        "so echo='false' is truthy"
    )
    identity: Final = f"cmpl-{uuid.uuid4().hex}"
    prompt: Final = f"typed form prompt {identity}"

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/completions"), request
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        assert request.headers["content-type"] == "application/json"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": _TEXT_BACKEND,
            "prompt": prompt,
            "temperature": 0.2,
            "max_tokens": 5,
            "n": 1,
            "echo": False,
        }, request.body
        return Reply(body=_text_completion(identity, ("tagged",)))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_TEXT_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        team: Final = scenario.team()
        user_id: Final = f"integration-{uuid.uuid4().hex}"
        key: Final = scenario.key(models=[model], team_id=team, user_id=user_id)
        response: Final = gateway.client.post(
            "/v1/completions",
            files=_form_fields(
                {
                    "model": model,
                    "prompt": prompt,
                    "temperature": "0.2",
                    "max_tokens": "5",
                    "n": "1",
                    "echo": "false",
                    "metadata": json.dumps({"tags": ["t1"]}),
                }
            ),
            headers={"Authorization": f"Bearer {key}"},
        )
        assert response.status_code == 200, response.text
        assert response.request.headers["content-type"].startswith("multipart/form-data"), response.request.headers
        payload: Final = _TextCompletion.model_validate_json(response.content)
        assert (payload.id, payload.object, payload.model) == (identity, "text_completion", model), response.text
        assert [(choice.index, choice.text) for choice in payload.choices] == [(0, "tagged")], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/completions")]
        _assert_spend_row_carries_client_tag_and_key_identity(identity, key, user_id, team)


def test_text_completion_sdk_parameter_set_reaches_upstream_whole(gateway: Gateway) -> None:
    identity: Final = f"cmpl-full-params-{uuid.uuid4().hex}"
    prompt: Final = f"full parameter prompt {identity}"
    expected: Final = {
        "model": _TEXT_BACKEND,
        "prompt": prompt,
        "suffix": " tail",
        "echo": True,
        "logprobs": 2,
        "best_of": 2,
        "n": 1,
        "logit_bias": {"50256": -100},
        "seed": 7,
        "user": "integration-user",
        "stop": ["\n", "END"],
        "presence_penalty": 0.5,
        "frequency_penalty": 0.25,
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 12,
    }

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/completions"), request
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        assert request.headers["content-type"] == "application/json"
        assert _JSON_OBJECT.validate_json(request.body) == expected, request.body
        return Reply(body=_text_completion(identity, ("one",)))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_TEXT_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        key: Final = scenario.key(models=[model])
        with _openai(gateway, key) as client:
            completion: Final = client.completions.create(
                model=model,
                prompt=prompt,
                suffix=" tail",
                echo=True,
                logprobs=2,
                best_of=2,
                n=1,
                logit_bias={"50256": -100},
                seed=7,
                user="integration-user",
                stop=["\n", "END"],
                presence_penalty=0.5,
                frequency_penalty=0.25,
                temperature=0.3,
                top_p=0.9,
                max_tokens=12,
            )
        assert (completion.id, completion.object) == (identity, "text_completion"), completion
        assert [(choice.index, choice.text) for choice in completion.choices] == [(0, "one")], completion
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/completions")]


@pytest.mark.parametrize(
    "prompt",
    (
        pytest.param(["first prompt", "second prompt"], id="string-list"),
        pytest.param([1, 2, 3], id="token-ids"),
        pytest.param([[1, 2], [3, 4]], id="token-id-batch"),
    ),
)
def test_text_completion_prompt_spellings_reach_upstream_unchanged(gateway: Gateway, prompt: JsonValue) -> None:
    identity: Final = f"cmpl-{uuid.uuid4().hex}"
    texts: Final = ("first", "second") if isinstance(prompt, list) and isinstance(prompt[0], (str, list)) else ("one",)

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/completions"), request
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        assert _JSON_OBJECT.validate_json(request.body) == {"model": _TEXT_BACKEND, "prompt": prompt}
        return Reply(body=_text_completion(identity, texts))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_TEXT_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        key: Final = scenario.key(models=[model])
        with _openai(gateway, key) as client:
            completion: Final = client.completions.create(
                model=model, prompt=prompt, extra_body={"cache": {"no-cache": True}}
            )
        assert completion.id == identity, completion
        assert completion.object == "text_completion", completion
        assert [(choice.index, choice.text) for choice in completion.choices] == list(enumerate(texts)), completion
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/completions")]


def test_text_completion_string_list_prompt_becomes_user_messages_for_chat_only_deployment(gateway: Gateway) -> None:
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    prompts: Final = [f"first {identity}", f"second {identity}"]

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/chat/completions"), request
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": _CHAT_BACKEND,
            "messages": [{"role": "user", "content": prompts[0]}, {"role": "user", "content": prompts[1]}],
        }
        return Reply(body=_chat_completion(identity, "answered"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_CHAT_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        key: Final = scenario.key(models=[model])
        with _openai(gateway, key) as client:
            completion: Final = client.completions.create(model=model, prompt=prompts)
        assert (completion.id, completion.object) == (identity, "text_completion"), completion
        assert [(choice.index, choice.text, choice.finish_reason) for choice in completion.choices] == [
            (0, "answered", "stop")
        ], completion
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]
