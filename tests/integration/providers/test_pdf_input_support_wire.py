import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.openai_wire import chat_reply
from integration._support.pdf_document import (
    LETTER,
    chat_body,
    chat_file,
    messages_body,
    pdf_data_url,
    pdf_document,
    responses_body,
    responses_input_file,
)
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_API_KEY: Final = "synthetic-pdf-gate-key"
_ANSWER: Final = "The report covers revenue."
_FILENAME: Final = "report.pdf"
_PDF: Final = pdf_data_url((LETTER,))
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final = TypeAdapter(list[JsonValue])
_COST_MAP_PATH: Final = Path(__file__).parents[3] / "litellm" / "model_prices_and_context_window_backup.json"


@dataclass(frozen=True, slots=True)
class _Entrypoint:
    path: str
    body: Callable[[str, str], Mapping[str, JsonValue]]
    stream: bool = False
    provider: str = "hosted_vllm"


_ENTRYPOINTS: Final = (
    pytest.param(
        _Entrypoint("/v1/chat/completions", lambda model, text: chat_body(model, [chat_file(_PDF, _FILENAME)], text)),
        id="chat-completions",
    ),
    pytest.param(
        _Entrypoint(
            "/v1/chat/completions",
            lambda model, text: chat_body(model, [chat_file(_PDF, _FILENAME)], text, stream=True),
            stream=True,
        ),
        id="chat-completions-stream",
    ),
    pytest.param(
        _Entrypoint("/v1/messages", lambda model, text: messages_body(model, [pdf_document((LETTER,))], text)),
        id="messages",
    ),
    pytest.param(
        _Entrypoint(
            "/v1/responses",
            lambda model, text: responses_body(model, [responses_input_file(_PDF, _FILENAME)], text),
            provider="lm_studio",
        ),
        id="responses-bridge",
    ),
)


def _azure_ai_chat_model_pdf_input(entry: JsonValue) -> bool | None:
    match entry:
        case {"litellm_provider": "azure_ai", "mode": "chat", "supports_pdf_input": bool() as supports}:
            return supports
        case _:
            return None


def _azure_ai_chat_model(reads_pdfs: bool) -> str:
    entries: Final = _JSON_OBJECT.validate_json(_COST_MAP_PATH.read_bytes())
    matching: Final = sorted(
        name for name, entry in entries.items() if _azure_ai_chat_model_pdf_input(entry) is reads_pdfs
    )
    assert matching, f"the cost map marks no azure_ai chat model with supports_pdf_input {reads_pdfs}"
    return matching[0]


def _ollama_reply(backend: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "model": backend,
                "created_at": "2026-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": _ANSWER},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 5,
                "eval_count": 3,
            }
        ).encode()
    )


def _responder(backend: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target.endswith("/api/chat"):
            return _ollama_reply(backend)
        body: Final = _JSON_OBJECT.validate_json(request.body)
        return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", backend, _ANSWER, stream=body.get("stream") is True)

    return respond


def _post(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", path, body, headers={"anthropic-version": "2023-06-01"})


def _post_streaming(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    with gateway.client.stream(
        "POST", path, json=dict(body), headers={"Authorization": f"Bearer {gateway.key}"}
    ) as response:
        response.read()
        return response


def _send(gateway: Gateway, entrypoint: _Entrypoint, model: str, text: str) -> httpx.Response:
    body: Final = entrypoint.body(model, text)
    if entrypoint.stream:
        return _post_streaming(gateway, entrypoint.path, body)
    return _post(gateway, entrypoint.path, body)


def _error_message(response: httpx.Response) -> str:
    match _JSON_OBJECT.validate_json(response.content):
        case {"error": {"message": str() as message}}:
            return message
        case _:
            raise AssertionError(response.text)


def _assert_refused_before_the_provider(response: httpx.Response, wire: Wire, backend: str) -> None:
    assert response.status_code == 400, response.text
    message: Final = _error_message(response)
    assert "PDF" in message and backend in message, message
    assert wire.drain() == ()


def _bodies(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return tuple(_JSON_OBJECT.validate_json(request.body) for request in wire.drain() if request.method == "POST")


def _only_body(wire: Wire) -> dict[str, JsonValue]:
    bodies: Final = _bodies(wire)
    assert len(bodies) == 1, bodies
    return bodies[0]


def _assert_reaches_the_provider(gateway: Gateway, wire: Wire, model: str) -> None:
    response: Final = _post(gateway, "/v1/chat/completions", chat_body(model, [], f"text only {uuid.uuid4().hex}"))
    assert response.status_code == 200, response.text
    assert len(_bodies(wire)) == 1


def _openai_compatible(
    scenario: Scenario,
    wire: Wire,
    backend: str,
    model_info: Mapping[str, JsonValue],
    provider: str = "hosted_vllm",
    **parameters: JsonValue,
) -> str:
    return scenario.model(
        model=f"{provider}/{backend}",
        api_base=f"{wire.url}/v1",
        api_key=_API_KEY,
        model_info=model_info,
        **parameters,
    )


def _backend() -> str:
    return f"pdf-gate-{uuid.uuid4().hex[:12]}"


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_pdf_to_a_deployment_marked_unable_to_read_it_is_refused_before_the_provider(
    gateway: Gateway, entrypoint: _Entrypoint
) -> None:
    backend: Final = _backend()
    with wire_server(_responder(backend)) as wire, gateway.scenario() as scenario:
        model: Final = _openai_compatible(scenario, wire, backend, {"supports_pdf_input": False}, entrypoint.provider)
        response: Final = _send(gateway, entrypoint, model, f"Summarize the report {uuid.uuid4().hex}")
        _assert_refused_before_the_provider(response, wire, backend)
        _assert_reaches_the_provider(gateway, wire, model)


def _base_model_deployment(scenario: Scenario, wire: Wire) -> tuple[str, str]:
    backend: Final = f"pdf-gate-{uuid.uuid4().hex[:12]}"
    model: Final = scenario.model(
        model=f"azure_ai/{backend}",
        api_base=wire.url,
        api_key=_API_KEY,
        model_info={"base_model": _azure_ai_chat_model(reads_pdfs=False)},
    )
    return model, backend


def _litellm_params_base_model_deployment(scenario: Scenario, wire: Wire) -> tuple[str, str]:
    backend: Final = _backend()
    model: Final = scenario.model(
        model=f"azure_ai/{backend}",
        api_base=wire.url,
        api_key=_API_KEY,
        base_model=_azure_ai_chat_model(reads_pdfs=False),
    )
    return model, backend


def _cost_map_deployment(scenario: Scenario, wire: Wire) -> tuple[str, str]:
    marked: Final = _azure_ai_chat_model(reads_pdfs=False)
    return scenario.model(model=marked, api_base=wire.url, api_key=_API_KEY), marked.removeprefix("azure_ai/")


def _ollama_deployment(scenario: Scenario, wire: Wire) -> tuple[str, str]:
    backend: Final = _backend()
    return scenario.model(model=f"ollama_chat/{backend}", api_base=wire.url, api_key=None), backend


@pytest.mark.parametrize(
    "deploy",
    [
        pytest.param(_base_model_deployment, id="model-info-base-model"),
        pytest.param(_litellm_params_base_model_deployment, id="litellm-params-base-model"),
        pytest.param(_cost_map_deployment, id="cost-map"),
        pytest.param(_ollama_deployment, id="ollama"),
    ],
)
def test_pdf_to_a_model_its_declaration_marks_unable_is_refused_before_the_provider(
    gateway: Gateway, deploy: Callable[[Scenario, Wire], tuple[str, str]]
) -> None:
    with wire_server(_responder("scripted")) as wire, gateway.scenario() as scenario:
        model, backend = deploy(scenario, wire)
        response: Final = _post(
            gateway,
            "/v1/chat/completions",
            chat_body(model, [chat_file(_PDF, _FILENAME)], f"Summarize the report {uuid.uuid4().hex}"),
        )
        _assert_refused_before_the_provider(response, wire, backend)
        _assert_reaches_the_provider(gateway, wire, model)


def _pdf_turns(question: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": [chat_file(_PDF, _FILENAME), {"type": "text", "text": question}]},
        {"role": "assistant", "content": "It is a quarterly report."},
        {"role": "user", "content": [chat_file(_PDF, "appendix.pdf")]},
        {"role": "user", "content": "Who wrote it?"},
    ]


def _answer(response: httpx.Response) -> JsonValue:
    match _JSON_OBJECT.validate_json(response.content):
        case {"choices": [{"message": {"content": content}}]}:
            return content
        case _:
            raise AssertionError(response.text)


def _turns_without_pdfs(question: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": [{"type": "text", "text": question}]},
        {"role": "assistant", "content": "It is a quarterly report."},
        {"role": "user", "content": "Who wrote it?"},
    ]


@pytest.mark.parametrize(
    ("deployment_params", "request_params"),
    [
        pytest.param({"drop_params": True}, {}, id="deployment-drop-params"),
        pytest.param({}, {"drop_params": True}, id="request-drop-params"),
    ],
)
def test_drop_params_sends_the_conversation_without_the_pdfs_a_model_cannot_read(
    gateway: Gateway, deployment_params: Mapping[str, JsonValue], request_params: Mapping[str, JsonValue]
) -> None:
    backend: Final = _backend()
    question: Final = f"Summarize the report {uuid.uuid4().hex}"
    with wire_server(_responder(backend)) as wire, gateway.scenario() as scenario:
        model: Final = _openai_compatible(scenario, wire, backend, {"supports_pdf_input": False}, **deployment_params)
        response: Final = _post(
            gateway,
            "/v1/chat/completions",
            {"model": model, "max_tokens": 64, "messages": _pdf_turns(question), **request_params},
        )
        assert response.status_code == 200, response.text
        assert _answer(response) == _ANSWER
        assert _only_body(wire)["messages"] == _turns_without_pdfs(question)


def test_drop_params_refuses_a_request_left_with_no_user_turn_once_the_pdf_is_dropped(gateway: Gateway) -> None:
    backend: Final = _backend()
    with wire_server(_responder(backend)) as wire, gateway.scenario() as scenario:
        model: Final = _openai_compatible(scenario, wire, backend, {"supports_pdf_input": False}, drop_params=True)
        response: Final = _post(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [
                    {"role": "system", "content": f"Answer from the attachment {uuid.uuid4().hex}"},
                    {"role": "user", "content": [chat_file(_PDF, _FILENAME)]},
                ],
            },
        )
        _assert_refused_before_the_provider(response, wire, backend)
        _assert_reaches_the_provider(gateway, wire, model)


def _file_and_question(question: str) -> list[JsonValue]:
    return [{"role": "user", "content": [chat_file(_PDF, _FILENAME), {"type": "text", "text": question}]}]


@pytest.mark.parametrize(
    ("model_info", "deployment_params", "keeps_pdf"),
    [
        pytest.param({}, {}, True, id="undeclared"),
        pytest.param({}, {"drop_params": True}, False, id="undeclared-drop-params"),
        pytest.param({"supports_pdf_input": True}, {"drop_params": True}, True, id="declared-drop-params"),
        pytest.param(
            {},
            {"drop_params": True, "base_model": _azure_ai_chat_model(reads_pdfs=True)},
            True,
            id="base-model-reads-drop-params",
        ),
    ],
)
def test_azure_ai_model_outside_the_cost_map_gets_the_pdf_unless_drop_params_finds_no_declaration(
    gateway: Gateway, model_info: Mapping[str, JsonValue], deployment_params: Mapping[str, JsonValue], keeps_pdf: bool
) -> None:
    backend: Final = _backend()
    question: Final = f"Summarize the report {uuid.uuid4().hex}"
    with wire_server(_responder(backend)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure_ai/{backend}",
            api_base=wire.url,
            api_key=_API_KEY,
            model_info=model_info,
            **deployment_params,
        )
        response: Final = _post(
            gateway, "/v1/chat/completions", chat_body(model, [chat_file(_PDF, _FILENAME)], question)
        )
        assert response.status_code == 200, response.text
        expected: Final[list[JsonValue]] = (
            _file_and_question(question) if keeps_pdf else [{"role": "user", "content": question}]
        )
        assert _only_body(wire)["messages"] == expected


def _sibling_pdf_request(gateway: Gateway, model: str) -> httpx.Response:
    return _post(
        gateway,
        "/v1/chat/completions",
        chat_body(model, [chat_file(_PDF, _FILENAME)], f"Summarize the report {uuid.uuid4().hex}"),
    )


def _forwarded_file_parts(bodies: Sequence[dict[str, JsonValue]]) -> tuple[JsonValue, ...]:
    return tuple(_MESSAGES.validate_python(body["messages"])[0] for body in bodies)


def test_pdf_opt_out_on_one_deployment_leaves_its_sibling_on_the_same_model_reading_pdfs(gateway: Gateway) -> None:
    backend: Final = _backend()
    burst: Final = 6
    with wire_server(_responder(backend)) as wire, gateway.scenario() as scenario:
        sibling: Final = _openai_compatible(scenario, wire, backend, {})
        marked: Final = _openai_compatible(scenario, wire, backend, {"supports_pdf_input": False})
        with ThreadPoolExecutor(max_workers=burst) as pool:
            responses: Final = tuple(pool.map(lambda _: _sibling_pdf_request(gateway, sibling), range(burst)))
        assert [response.status_code for response in responses] == [200] * burst, [r.text for r in responses]
        forwarded: Final = _forwarded_file_parts(_bodies(wire))
        assert len(forwarded) == burst, forwarded
        assert all(
            _MESSAGES.validate_python(_JSON_OBJECT.validate_python(message)["content"])[0] == chat_file(_PDF, _FILENAME)
            for message in forwarded
        ), forwarded
        refused: Final = _sibling_pdf_request(gateway, marked)
        _assert_refused_before_the_provider(refused, wire, backend)
