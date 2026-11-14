import base64
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final
from unittest.mock import patch

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
import litellm.main
from litellm.llms.custom_httpx.http_handler import HTTPHandler


@pytest.fixture
def local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


def test_reasoning_effort_stays_unsupported_on_vertex_partner_models(local_model_cost_map):
    assert "reasoning_effort" in litellm.get_supported_openai_params(
        model="mistral-medium-3", custom_llm_provider="mistral"
    )
    assert "reasoning_effort" not in litellm.get_supported_openai_params(
        model="mistral-medium-3", custom_llm_provider="vertex_ai"
    )
    dropped = litellm.get_optional_params(
        model="mistral-medium-3",
        custom_llm_provider="vertex_ai",
        reasoning_effort="high",
        drop_params=True,
    )
    assert "reasoning_effort" not in dropped


LABEL_FIELDS: Final = ("user_api_key_alias", "user_api_key_team_alias", "spend_logs_metadata")
IDENTITY: Final = MappingProxyType({"user_api_key_alias": "chargeback-key", "user_api_key_team_alias": "ai-platform"})
PROXY_METADATA: Final = MappingProxyType({**IDENTITY, "spend_logs_metadata": {"cost-center": "cc-1"}})
FORGED: Final = MappingProxyType({"user_api_key_alias": "forged", "app": "bot"})
CHAT_REPLY: Final = MappingProxyType(
    {
        "id": "chat-1",
        "object": "chat.completion",
        "created": 1,
        "model": "mistral-large-2411",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
)
SENT_BODY: Final = TypeAdapter(Mapping[str, object])
SENT_LABELS: Final = TypeAdapter(Mapping[str, str])


def _transport(sent: list[httpx.Request]) -> httpx.MockTransport:
    def reply(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=dict(CHAT_REPLY))

    return httpx.MockTransport(reply)


def _send_mistral_chat(transport: httpx.MockTransport, monkeypatch: pytest.MonkeyPatch) -> None:
    litellm.completion(
        model="vertex_ai/mistral-large-2411",
        messages=[{"role": "user", "content": "hi"}],
        vertex_location="us-central1",
        metadata=dict(PROXY_METADATA),
        labels=dict(FORGED),
        client=HTTPHandler(client=httpx.Client(transport=transport)),
    )


def _send_codestral_fim(transport: httpx.MockTransport, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "module_level_client", HTTPHandler(client=httpx.Client(transport=transport)))
    monkeypatch.setitem(
        litellm.known_tokenizer_config,
        "codestral-2",
        {"status": "success", "tokenizer": {"chat_template": "{{ messages[0]['content'] }}"}},
    )
    litellm.text_completion(
        model="vertex_ai/codestral-2",
        prompt="def add(",
        vertex_location="us-central1",
        metadata=dict(PROXY_METADATA),
        labels=dict(FORGED),
    )


MISTRAL_CALLS: Final = pytest.mark.parametrize(
    "send, todays_body",
    [
        (
            _send_mistral_chat,
            {"model": "mistral-large-2411", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        ),
        (_send_codestral_fim, {"model": "codestral-2", "prompt": "def add("}),
    ],
    ids=["chat", "codestral-fim"],
)


def _sent_mistral_request(
    send: Callable[[httpx.MockTransport, pytest.MonkeyPatch], None], monkeypatch: pytest.MonkeyPatch
) -> httpx.Request:
    sent: Final[list[httpx.Request]] = []
    with patch.object(
        litellm.main.vertex_partner_models_chat_completion, "_ensure_access_token", return_value=("token", "proj")
    ):
        send(_transport(sent), monkeypatch)
    assert [request.url.path.rsplit(":", 1)[-1] for request in sent] == ["rawPredict"]
    return sent[0]


@MISTRAL_CALLS
def test_mistral_labels_header_carries_identity_only_while_the_setting_is_on(
    send: Callable[[httpx.MockTransport, pytest.MonkeyPatch], None],
    todays_body: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", None)
    setting_off: Final = _sent_mistral_request(send, monkeypatch)
    monkeypatch.setattr(litellm, "vertex_request_label_fields", LABEL_FIELDS)
    setting_on: Final = _sent_mistral_request(send, monkeypatch)

    assert "x-vertex-ai-labels" not in setting_off.headers
    assert SENT_BODY.validate_json(setting_off.content) == {**todays_body, "labels": dict(FORGED)}
    assert list(SENT_LABELS.validate_json(base64.b64decode(setting_on.headers["x-vertex-ai-labels"])).items()) == [
        *IDENTITY.items(),
        ("app", "bot"),
        ("cost-center", "cc-1"),
    ]
    assert SENT_BODY.validate_json(setting_on.content) == todays_body
    assert (setting_on.headers["content-type"], setting_on.headers["authorization"]) == (
        "application/json",
        "Bearer token",
    )


def test_mistral_extra_body_labels_join_the_labels_header_instead_of_the_body(monkeypatch: pytest.MonkeyPatch):
    def send(transport: httpx.MockTransport, _: pytest.MonkeyPatch) -> None:
        litellm.completion(
            model="vertex_ai/mistral-large-2411",
            messages=[{"role": "user", "content": "hi"}],
            vertex_location="us-central1",
            metadata=dict(PROXY_METADATA),
            labels={"app": "bot"},
            extra_body={"labels": {"app": "agent", "env": "prod"}, "safe_prompt": True},
            client=HTTPHandler(client=httpx.Client(transport=transport)),
        )

    monkeypatch.setattr(litellm, "vertex_request_label_fields", LABEL_FIELDS)
    sent: Final = _sent_mistral_request(send, monkeypatch)

    assert list(SENT_LABELS.validate_json(base64.b64decode(sent.headers["x-vertex-ai-labels"])).items()) == [
        *IDENTITY.items(),
        ("app", "agent"),
        ("env", "prod"),
        ("cost-center", "cc-1"),
    ]
    assert SENT_BODY.validate_json(sent.content) == {
        "model": "mistral-large-2411",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "safe_prompt": True,
    }
