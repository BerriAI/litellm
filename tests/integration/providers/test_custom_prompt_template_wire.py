import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

import anthropic
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "llama3-custom-template"
_API_KEY: Final = "synthetic-ollama-key"
_SYSTEM: Final = "Answer in one word."
_QUESTION: Final = "What colour is the sky?"
_ANSWER: Final = "blue"
_INITIAL: Final = "<bos>"
_FINAL: Final = "<assistant>"
_ROLES: Final[dict[str, JsonValue]] = {
    "system": {"pre_message": "<sys>", "post_message": "</sys>"},
    "user": {"pre_message": "<user>", "post_message": "</user>"},
    "assistant": {"pre_message": "<asst>", "post_message": "</asst>"},
}
_ROLES_ONLY: Final[dict[str, JsonValue]] = {"roles": _ROLES}
_VALUES_ONLY: Final[dict[str, JsonValue]] = {"initial_prompt_value": _INITIAL, "final_prompt_value": _FINAL}
_FULL: Final[dict[str, JsonValue]] = {**_ROLES_ONLY, **_VALUES_ONLY}
_ROLES_PROMPT: Final = f"<sys>{_SYSTEM}</sys><user>{_QUESTION}</user>"
_VALUES_PROMPT: Final = f"{_INITIAL}{_SYSTEM}{_QUESTION}{_FINAL}"
_FULL_PROMPT: Final = f"{_INITIAL}{_ROLES_PROMPT}{_FINAL}"
_TEMPLATES: Final[Mapping[str, tuple[dict[str, JsonValue], str]]] = {
    "roles_only": (_ROLES_ONLY, _ROLES_PROMPT),
    "values_only": (_VALUES_ONLY, _VALUES_PROMPT),
}
_TEMPLATE_FIELDS: Final = frozenset({"roles", "initial_prompt_value", "final_prompt_value", "bos_token", "eos_token"})
_AWS: Final[dict[str, JsonValue]] = {
    "aws_access_key_id": "AKIASYNTHETICKEY0001",
    "aws_secret_access_key": "synthetic-secret-key-for-testing",
    "aws_region_name": "us-east-1",
}
_NO_CACHE: Final[dict[str, JsonValue]] = {"no-cache": True}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_LIST: Final = TypeAdapter(list[JsonValue])


def _generate_reply(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "model": _BACKEND,
                "created_at": "2026-10-15T00:00:00Z",
                "response": text,
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 30,
                "eval_count": 12,
            }
        ).encode()
    )


def _streamed_reply(text: str) -> Reply:
    frames: Final = (
        {"model": _BACKEND, "created_at": "2026-10-15T00:00:00Z", "response": text, "done": False},
        {
            "model": _BACKEND,
            "created_at": "2026-10-15T00:00:00Z",
            "response": "",
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 30,
            "eval_count": 12,
        },
    )
    return Reply(
        content_type="application/x-ndjson", chunks=tuple(json.dumps(frame).encode() + b"\n" for frame in frames)
    )


def _is_generate(request: Request) -> bool:
    return (request.method, request.target) == ("POST", "/api/generate")


def _echo_answer(request: Request) -> Reply:
    stream: Final = _JSON_OBJECT.validate_json(request.body).get("stream") is True
    return _streamed_reply(_ANSWER) if stream else _generate_reply(_ANSWER)


@contextmanager
def _ollama_server(respond: Callable[[Request], Reply] = _echo_answer) -> Iterator[Wire]:
    with wire_server(lambda request: respond(request) if _is_generate(request) else Reply(body=b"{}")) as wire:
        yield wire


def _generate_bodies(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return tuple(_JSON_OBJECT.validate_json(request.body) for request in wire.drain() if _is_generate(request))


def _assert_rendered(body: dict[str, JsonValue], prompt: str, *, stream: bool) -> None:
    assert body["model"] == _BACKEND
    assert body["prompt"] == prompt, body["prompt"]
    assert body["stream"] is stream
    options: Final = body.get("options", {})
    assert isinstance(options, dict)
    assert not _TEMPLATE_FIELDS & (set(body) | set(options)), sorted(body)


def _assert_only_rendered(wire: Wire, prompt: str, *, stream: bool = False) -> None:
    bodies: Final = _generate_bodies(wire)
    assert len(bodies) == 1, bodies
    _assert_rendered(bodies[0], prompt, stream=stream)


def _ollama_model(scenario: Scenario, wire: Wire, template: Mapping[str, JsonValue]) -> str:
    return scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY, **template)


def _chat_messages() -> list[dict[str, JsonValue]]:
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": _QUESTION}]


def _spend_row(identity: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT model_group, status, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (identity,),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def _model_spend_rows(model: str, expected: int) -> list[dict[str, JsonValue]]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
            (model,),
        ),
        lambda found: len(found) >= expected,
        seconds=70,
    )
    assert len({row["request_id"] for row in rows}) == len(rows) == expected, rows
    return rows


def _billed(model: str) -> dict[str, JsonValue]:
    return {"model_group": model, "status": "success", "prompt_tokens": 30, "completion_tokens": 12}


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


def _async_anthropic_client(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


def _post(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> tuple[int, str]:
    response: Final = gateway.request("POST", path, {**body, "cache": _NO_CACHE})
    return response.status_code, response.text


def _chat(gateway: Gateway, model: str, **extra: JsonValue) -> tuple[int, str]:
    return _post(gateway, "/v1/chat/completions", {"model": model, "messages": _chat_messages(), **extra})


def _chat_content(text: str) -> str:
    choices: Final = _JSON_LIST.validate_python(_JSON_OBJECT.validate_json(text)["choices"])
    message: Final = object_value(object_value(choices[0])["message"])
    return string_value(message["content"])


def test_openai_sdk_chat_renders_a_roles_only_template(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _ROLES_ONLY)
        completion: Final = _openai_client(gateway).chat.completions.create(
            model=model,
            messages=_chat_messages(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            extra_body={"cache": _NO_CACHE},
        )
        assert completion.choices[0].message.content == _ANSWER
        assert completion.choices[0].finish_reason == "stop"
        assert completion.usage is not None
        assert (completion.usage.prompt_tokens, completion.usage.completion_tokens) == (30, 12)
        _assert_only_rendered(wire, _ROLES_PROMPT)
        assert _spend_row(completion.id) == _billed(model)


async def test_async_openai_sdk_chat_stream_renders_a_roles_only_template(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _ROLES_ONLY)
        stream: Final = await _async_openai_client(gateway).chat.completions.create(
            model=model,
            messages=_chat_messages(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"cache": _NO_CACHE},
        )
        chunks: Final = [chunk async for chunk in stream]
        assert {chunk.id for chunk in chunks} == {chunks[0].id}
        assert "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices) == _ANSWER
        usages: Final = [chunk.usage for chunk in chunks if chunk.usage is not None]
        assert [(usage.prompt_tokens, usage.completion_tokens) for usage in usages] == [(30, 12)]
        _assert_only_rendered(wire, _ROLES_PROMPT, stream=True)
        assert _spend_row(chunks[0].id) == _billed(model)


def test_openai_sdk_completions_render_a_roles_only_template(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _ROLES_ONLY)
        completion: Final = _openai_client(gateway).completions.create(
            model=model, prompt=_QUESTION, extra_body={"cache": _NO_CACHE}
        )
        assert completion.choices[0].text == _ANSWER
        _assert_only_rendered(wire, f"<user>{_QUESTION}</user>")
        assert _spend_row(completion.id) == _billed(model)


def test_anthropic_sdk_messages_render_a_roles_only_template(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _ROLES_ONLY)
        message: Final = _anthropic_client(gateway).messages.create(
            model=model,
            max_tokens=64,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _QUESTION}],
            extra_body={"cache": _NO_CACHE},
        )
        assert message.stop_reason == "end_turn"
        assert [(block.type, getattr(block, "text", None)) for block in message.content] == [("text", _ANSWER)]
        assert (message.usage.input_tokens, message.usage.output_tokens) == (30, 12)
        _assert_only_rendered(wire, _ROLES_PROMPT)
        assert _spend_row(message.id) == _billed(model)


async def test_async_anthropic_sdk_messages_stream_render_a_roles_only_template(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _ROLES_ONLY)
        async with _async_anthropic_client(gateway).messages.stream(
            model=model,
            max_tokens=64,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _QUESTION}],
            extra_body={"cache": _NO_CACHE},
        ) as stream:
            text: Final = "".join([piece async for piece in stream.text_stream])
            final: Final = await stream.get_final_message()
        assert text == _ANSWER
        assert final.stop_reason == "end_turn"
        _assert_only_rendered(wire, _ROLES_PROMPT, stream=True)
        assert _spend_row(final.id) == _billed(model)


def test_openai_sdk_responses_render_a_roles_only_template(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _ROLES_ONLY)
        response: Final = _openai_client(gateway).responses.create(
            model=model, instructions=_SYSTEM, input=_QUESTION, store=False, extra_body={"cache": _NO_CACHE}
        )
        assert response.output_text == _ANSWER
        assert response.usage is not None and (response.usage.input_tokens, response.usage.output_tokens) == (30, 12)
        _assert_only_rendered(wire, _ROLES_PROMPT)
        rows: Final = _model_spend_rows(model, 1)
        assert (rows[0]["status"], rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == ("success", 30, 12)


async def test_async_openai_sdk_responses_stream_render_a_roles_only_template(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _ROLES_ONLY)
        stream: Final = await _async_openai_client(gateway).responses.create(
            model=model,
            instructions=_SYSTEM,
            input=_QUESTION,
            store=False,
            stream=True,
            extra_body={"cache": _NO_CACHE},
        )
        events: Final = [event async for event in stream]
        assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == _ANSWER
        completed: Final = [event for event in events if event.type == "response.completed"]
        assert len(completed) == 1 and completed[0].response.output_text == _ANSWER
        _assert_only_rendered(wire, _ROLES_PROMPT, stream=True)
        assert _model_spend_rows(model, 1)[0]["status"] == "success"


def test_prompt_values_only_template_wraps_the_bare_messages(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _VALUES_ONLY)
        code, text = _chat(gateway, model)
        assert code == 200, text
        assert _chat_content(text) == _ANSWER
        _assert_only_rendered(wire, _VALUES_PROMPT)


def test_full_template_renders_every_part(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _FULL)
        code, text = _chat(gateway, model)
        assert code == 200, text
        assert _chat_content(text) == _ANSWER
        _assert_only_rendered(wire, _FULL_PROMPT)


@dataclass(frozen=True, slots=True)
class _Member:
    deployment: Callable[[str], dict[str, JsonValue]]
    reply: Callable[[str], Reply]
    prompt_field: str
    target: Callable[[str], bool]


def _codestral_reply(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "cmpl-scripted",
                "object": "chat.completion",
                "model": "codestral-latest",
                "created": 1,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
            }
        ).encode()
    )


def _predibase_reply(text: str) -> Reply:
    return Reply(body=json.dumps({"generated_text": text, "details": {"finish_reason": "eos_token"}}).encode())


def _sagemaker_reply(text: str) -> Reply:
    return Reply(body=json.dumps([{"generated_text": text}]).encode())


def _bedrock_meta_reply(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {"generation": text, "prompt_token_count": 30, "generation_token_count": 12, "stop_reason": "stop"}
        ).encode()
    )


_FAMILY: Final[Mapping[str, _Member]] = {
    "codestral": _Member(
        deployment=lambda url: {
            "model": "text-completion-codestral/codestral-latest",
            "api_base": f"{url}/v1/fim/completions",
            "api_key": "scripted-key",
        },
        reply=_codestral_reply,
        prompt_field="prompt",
        target=lambda target: target == "/v1/fim/completions",
    ),
    "predibase": _Member(
        deployment=lambda url: {
            "model": "predibase/llama-3-8b-instruct",
            "api_base": url,
            "api_key": "scripted-key",
            "predibase_tenant_id": "scripted-tenant",
        },
        reply=_predibase_reply,
        prompt_field="inputs",
        target=lambda target: target.startswith("/scripted-tenant/deployments/v2/llms/llama-3-8b-instruct/generate"),
    ),
    "sagemaker": _Member(
        deployment=lambda url: {
            "model": "sagemaker/scripted-endpoint",
            "sagemaker_base_url": f"{url}/endpoints/scripted-endpoint/invocations",
            **_AWS,
        },
        reply=_sagemaker_reply,
        prompt_field="inputs",
        target=lambda target: target == "/endpoints/scripted-endpoint/invocations",
    ),
    "bedrock_invoke": _Member(
        deployment=lambda url: {"model": "bedrock/invoke/meta.llama3-8b-instruct-v1:0", "api_base": url, **_AWS},
        reply=_bedrock_meta_reply,
        prompt_field="prompt",
        target=lambda target: target.endswith("/invoke"),
    ),
}


@pytest.mark.parametrize("template", sorted(_TEMPLATES))
@pytest.mark.parametrize("member", sorted(_FAMILY))
def test_family_member_renders_a_partial_template(gateway: Gateway, member: str, template: str) -> None:
    spec: Final = _FAMILY[member]
    fields, prompt = _TEMPLATES[template]
    with wire_server(lambda _: spec.reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(**spec.deployment(wire.url), **fields)
        code, text = _chat(gateway, model)
        received: Final = tuple(request for request in wire.drain() if request.method == "POST")
        assert code == 200, text
        assert _chat_content(text) == _ANSWER
        assert len(received) == 1, [request.target for request in received]
        assert spec.target(received[0].target), received[0].target
        body: Final = _JSON_OBJECT.validate_json(received[0].body)
        assert body[spec.prompt_field] == prompt, body
        assert not _TEMPLATE_FIELDS & set(body), sorted(body)


@pytest.mark.parametrize("provider", ("vllm", "petals"))
def test_in_process_family_member_fails_alone_and_the_rig_keeps_serving(gateway: Gateway, provider: str) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        broken: Final = scenario.model(model=f"{provider}/scripted-model", **_ROLES_ONLY)
        healthy: Final = _ollama_model(scenario, wire, _ROLES_ONLY)
        failed, failure = _chat(gateway, broken)
        assert failed >= 400, failure
        code, text = _chat(gateway, healthy)
        assert code == 200, text
        _assert_only_rendered(wire, _ROLES_PROMPT)


def _watsonx_reply(text: str) -> Reply:
    result: Final[dict[str, JsonValue]] = {
        "generated_text": text,
        "generated_token_count": 12,
        "input_token_count": 30,
        "stop_reason": "eos_token",
    }
    return Reply(body=json.dumps({"model_id": "ibm/granite", "results": [result]}).encode())


def test_watsonx_text_deployment_with_a_template_still_answers(gateway: Gateway) -> None:
    with wire_server(lambda _: _watsonx_reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="watsonx_text/ibm/granite-chat-scripted",
            api_base=wire.url,
            token="scripted-token",
            project_id="p",
            **_ROLES_ONLY,
        )
        code, text = _chat(gateway, model)
        assert code == 200, text
        assert _chat_content(text) == _ANSWER
        posts: Final = tuple(request for request in wire.drain() if request.method == "POST")
        assert len(posts) == 1, [request.target for request in posts]
        body: Final = _JSON_OBJECT.validate_json(posts[0].body)
        assert _QUESTION in string_value(body["input"]), body
        assert not _TEMPLATE_FIELDS & set(body), sorted(body)


def _openai_chat_reply(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-scripted",
                "object": "chat.completion",
                "created": 1,
                "model": "scripted-compat",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
            }
        ).encode()
    )


def _converse_reply(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 30, "outputTokens": 12, "totalTokens": 42},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


_FORWARDING: Final[Mapping[str, tuple[Callable[[str], dict[str, JsonValue]], Callable[[str], Reply]]]] = {
    "openai_compatible": (
        lambda url: {"model": "openai/scripted-compat", "api_base": f"{url}/v1", "api_key": "scripted-key"},
        _openai_chat_reply,
    ),
    "bedrock_converse": (
        lambda url: {"model": "bedrock/converse/meta.llama3-8b-instruct-v1:0", "api_base": url, **_AWS},
        _converse_reply,
    ),
}


def _keys_anywhere(value: JsonValue) -> frozenset[str]:
    if isinstance(value, dict):
        nested: Final = frozenset(key for child in value.values() for key in _keys_anywhere(child))
        return frozenset(value) | nested
    if isinstance(value, list):
        return frozenset(key for child in value for key in _keys_anywhere(child))
    return frozenset()


@pytest.mark.parametrize("source", ("deployment", "request"))
@pytest.mark.parametrize("member", sorted(_FORWARDING))
def test_template_fields_never_reach_a_provider_that_does_not_render_them(
    gateway: Gateway, member: str, source: str
) -> None:
    deployment, reply = _FORWARDING[member]
    with wire_server(lambda _: reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(**deployment(wire.url), **(_FULL if source == "deployment" else {}))
        code, text = _chat(gateway, model, **(_FULL if source == "request" else {}))
        assert code == 200, text
        assert _chat_content(text) == _ANSWER
        posts: Final = tuple(request for request in wire.drain() if request.method == "POST")
        assert len(posts) == 1, [request.target for request in posts]
        body: Final = _JSON_OBJECT.validate_json(posts[0].body)
        assert not _TEMPLATE_FIELDS & _keys_anywhere(body), body
        assert _INITIAL not in posts[0].body.decode() and _FINAL not in posts[0].body.decode(), body


@pytest.mark.parametrize("value", (5, [_INITIAL]), ids=("int", "list"))
@pytest.mark.parametrize("source", ("deployment", "request"))
def test_non_string_initial_prompt_value_is_rejected_by_name(gateway: Gateway, source: str, value: JsonValue) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        template: Final[dict[str, JsonValue]] = {**_FULL, "initial_prompt_value": value}
        broken: Final = _ollama_model(scenario, wire, template if source == "deployment" else _FULL)
        healthy: Final = _ollama_model(scenario, wire, _FULL)
        failed, failure = _chat(gateway, broken, **({"initial_prompt_value": value} if source == "request" else {}))
        assert failed == 500, failure
        assert "initial_prompt_value: Input should be a valid string" in failure, failure
        assert _generate_bodies(wire) == ()
        code, text = _chat(gateway, healthy)
        assert code == 200, text
        _assert_only_rendered(wire, _FULL_PROMPT)


@pytest.mark.parametrize("value", ("", None), ids=("empty", "null"))
def test_blank_initial_prompt_value_in_the_request_drops_the_deployment_value(
    gateway: Gateway, value: JsonValue
) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _FULL)
        code, text = _chat(gateway, model, initial_prompt_value=value)
        assert code == 200, text
        _assert_only_rendered(wire, f"{_ROLES_PROMPT}{_FINAL}")


def test_request_initial_prompt_value_overrides_the_deployment_value(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _FULL)
        code, text = _chat(gateway, model, initial_prompt_value="<override>")
        assert code == 200, text
        _assert_only_rendered(wire, f"<override>{_ROLES_PROMPT}{_FINAL}")


def test_five_kilobyte_initial_prompt_value_leads_the_prompt(gateway: Gateway) -> None:
    large: Final = "x" * 5120
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _ROLES_ONLY)
        code, text = _chat(gateway, model, initial_prompt_value=large)
        assert code == 200, text
        _assert_only_rendered(wire, f"{large}{_ROLES_PROMPT}")


def test_identical_requests_each_render_and_bill_once(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _ROLES_ONLY)
        first, first_text = _chat(gateway, model)
        second, second_text = _chat(gateway, model)
        assert (first, second) == (200, 200), (first_text, second_text)
        bodies: Final = _generate_bodies(wire)
        assert len(bodies) == 2, bodies
        for body in bodies:
            _assert_rendered(body, _ROLES_PROMPT, stream=False)
        identities: Final = {string_value(_JSON_OBJECT.validate_json(text)["id"]) for text in (first_text, second_text)}
        assert len(identities) == 2
        for identity in identities:
            assert _spend_row(identity) == _billed(model)


def test_template_with_only_a_bos_token_names_the_fields_it_lacks(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, {"bos_token": "<s>"})
        code, text = _chat(gateway, model)
        assert code == 500, text
        assert "it sets none of roles, initial_prompt_value, or final_prompt_value" in text, text
        assert _generate_bodies(wire) == ()


def test_roles_that_are_not_a_mapping_render_without_role_markers(gateway: Gateway) -> None:
    with _ollama_server() as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, {**_VALUES_ONLY, "roles": json.dumps(_ROLES)})
        code, text = _chat(gateway, model)
        assert code == 200, text
        _assert_only_rendered(wire, _VALUES_PROMPT)


def test_provider_error_reaches_the_caller_after_one_attempt(gateway: Gateway) -> None:
    missing: Final = Reply(status=404, body=json.dumps({"error": f"model '{_BACKEND}' not found"}).encode())
    with _ollama_server(lambda _: missing) as wire, gateway.scenario() as scenario:
        model: Final = _ollama_model(scenario, wire, _ROLES_ONLY)
        code, text = _chat(gateway, model)
        assert code == 404, text
        assert f"model '{_BACKEND}' not found" in text, text
        bodies: Final = _generate_bodies(wire)
        assert len(bodies) == 1, bodies
        _assert_rendered(bodies[0], _ROLES_PROMPT, stream=False)


def _template_keys(fields: Mapping[str, JsonValue]) -> tuple[bool, bool]:
    return "initial_prompt_value" in fields, "final_prompt_value" in fields


def test_pass_through_body_treats_initial_and_final_prompt_values_alike(gateway: Gateway) -> None:
    with wire_server(lambda _: _generate_reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        path: Final = f"/integration-template-passthrough-{uuid.uuid4().hex}"
        created: Final = gateway.post(
            "/config/pass_through_endpoint", {"path": path, "target": f"{wire.url}/api/generate"}
        )
        endpoint: Final = object_value(_JSON_LIST.validate_python(created["endpoints"])[0])
        scenario.cleanups.callback(
            lambda: gateway.request(
                "DELETE", "/config/pass_through_endpoint", params={"endpoint_id": str(endpoint["id"])}
            )
        )
        response: Final = eventually(
            lambda: gateway.request("POST", path, {"model": _BACKEND, "prompt": _QUESTION, "stream": False, **_FULL}),
            lambda answered: answered.status_code != 404,
            seconds=30,
        )
        assert response.status_code == 200, response.text
        posts: Final = tuple(request for request in wire.drain() if request.method == "POST")
        forwarded: Final = _JSON_OBJECT.validate_json(posts[-1].body)
        assert (forwarded["model"], forwarded["prompt"]) == (_BACKEND, _QUESTION), forwarded
        initial, final = _template_keys(forwarded)
        assert initial == final, sorted(forwarded)


def _a2a_reply(request: Request) -> Reply:
    result: Final[dict[str, JsonValue]] = {
        "kind": "message",
        "role": "agent",
        "messageId": f"{uuid.uuid4().hex}-out",
        "parts": [{"kind": "text", "text": _ANSWER}],
    }
    outbound: Final = _JSON_OBJECT.validate_json(request.body)
    return Reply(body=json.dumps({"jsonrpc": "2.0", "id": outbound["id"], "result": result}).encode())


def test_a2a_params_treat_initial_and_final_prompt_values_alike(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    agent_name: Final = f"template-agent-{marker}"

    def peer(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(_agent_card(wire.url, agent_name)).encode())
        return _a2a_reply(request)

    with wire_server(peer) as wire, gateway.scenario() as scenario:
        created: Final = gateway.post(
            "/v1/agents", {"agent_name": agent_name, "agent_card_params": _agent_card(wire.url, agent_name)}
        )
        agent_id: Final = string_value(created["agent_id"])
        scenario.cleanups.callback(lambda: gateway.request("DELETE", f"/v1/agents/{agent_id}"))
        message: Final[dict[str, JsonValue]] = {
            "kind": "message",
            "role": "user",
            "messageId": f"{marker}-in",
            "parts": [{"kind": "text", "text": _QUESTION}],
        }
        invoked: Final = gateway.request(
            "POST",
            f"/a2a/{agent_id}",
            {
                "jsonrpc": "2.0",
                "id": marker,
                "method": "message/send",
                "params": {"message": message, "initial_prompt_value": _INITIAL, "final_prompt_value": _FINAL},
            },
        )
        assert invoked.status_code == 200, invoked.text
        assert object_value(invoked.json()["result"])["parts"] == [{"kind": "text", "text": _ANSWER}], invoked.text
        posts: Final = tuple(request for request in wire.drain() if request.method == "POST")
        assert len(posts) == 1, [request.target for request in posts]
        params: Final = object_value(_JSON_OBJECT.validate_json(posts[0].body)["params"])
        assert _template_keys(params) == (False, False), sorted(params)


def _agent_card(url: str, name: str) -> dict[str, JsonValue]:
    return {
        "protocolVersion": "0.3",
        "name": name,
        "description": "Scripted A2A peer",
        "version": "1.0.0",
        "url": url + "/",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }
