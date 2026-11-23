import base64
import copy
import re
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
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


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


MISTRAL_TOOL_CALL_ID: Final = re.compile(r"[a-zA-Z0-9]{9}")
SENT_MAPPINGS: Final = TypeAdapter(list[Mapping[str, object]])


def _tool_call(tool_call_id: object, name: str) -> dict[str, object]:
    return {"id": tool_call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


@pytest.mark.parametrize(
    "model", ["vertex_ai/mistral-large-2411", "vertex_ai/mistral-large@2407", "vertex_ai/mistral-large-3"]
)
def test_mistral_chat_sends_replayed_tool_call_ids_as_nine_alphanumeric_characters(
    model: str, monkeypatch: pytest.MonkeyPatch
):
    replayed_ids: Final = ("call_u0U86kcRTK484QKTyPKASCeg", "call_abc123", "toolu_01ABCdef", "D681PevKs")
    messages: Final = [
        {"role": "user", "content": "Weather and time in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call(tool_call_id, f"tool_{n}") for n, tool_call_id in enumerate(replayed_ids)],
        },
        *({"role": "tool", "tool_call_id": tool_call_id, "content": "ok"} for tool_call_id in replayed_ids),
        {"role": "user", "content": "Sum it up."},
    ]
    callers_messages: Final = copy.deepcopy(messages)

    def send(transport: httpx.MockTransport, _: pytest.MonkeyPatch) -> None:
        litellm.completion(
            model=model,
            messages=messages,
            vertex_location="us-central1",
            client=HTTPHandler(client=httpx.Client(transport=transport)),
        )

    sent_messages: Final = SENT_MAPPINGS.validate_python(
        SENT_BODY.validate_json(_sent_mistral_request(send, monkeypatch).content)["messages"]
    )
    sent_call_ids: Final = [
        tool_call["id"] for tool_call in SENT_MAPPINGS.validate_python(sent_messages[1]["tool_calls"])
    ]
    sent_reply_ids: Final = [message["tool_call_id"] for message in sent_messages[2:6]]

    assert sent_reply_ids == sent_call_ids
    assert all(
        isinstance(tool_call_id, str) and MISTRAL_TOOL_CALL_ID.fullmatch(tool_call_id) for tool_call_id in sent_call_ids
    )
    assert len(set(sent_call_ids)) == len(replayed_ids)
    assert sent_call_ids[3] == "D681PevKs"
    assert messages == callers_messages


def _sent_body_without_model(model: str, monkeypatch: pytest.MonkeyPatch) -> Mapping[str, object]:
    messages: Final = [
        {
            "role": "user",
            "name": "alice",
            "content": [{"type": "text", "text": "Weather "}, {"type": "text", "text": "in Paris?"}],
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Look it up first.",
            "tool_calls": [_tool_call("call_abc123", "get_weather")],
        },
        {"role": "tool", "tool_call_id": "call_abc123", "content": [{"type": "text", "text": "18C"}]},
    ]

    def send(transport: httpx.MockTransport, _: pytest.MonkeyPatch) -> None:
        litellm.completion(
            model=model,
            messages=messages,
            tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
            tool_choice="required",
            seed=7,
            vertex_location="us-central1",
            client=HTTPHandler(client=httpx.Client(transport=transport)),
        )

    sent: Final = SENT_BODY.validate_json(_sent_mistral_request(send, monkeypatch).content)
    return {key: value for key, value in sent.items() if key != "model"}


@pytest.mark.parametrize("model", ["vertex_ai/mistral-large@2407", "vertex_ai/mistral-large-3"])
def test_mistral_chat_sends_every_vertex_mistral_name_the_body_a_listed_mistral_model_gets(
    model: str, local_model_cost_map, monkeypatch: pytest.MonkeyPatch
):
    assert _sent_body_without_model(model, monkeypatch) == _sent_body_without_model(
        "vertex_ai/mistral-medium-3", monkeypatch
    )


async def test_mistral_chat_maps_a_tool_call_id_the_same_way_on_sync_and_async_requests(local_model_cost_map):
    config: Final = ProviderConfigManager.get_provider_chat_config(
        model="mistral-medium-3", provider=LlmProviders.VERTEX_AI
    )
    history: Final = [
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("call_abc123", "get_weather")]},
        {"role": "tool", "tool_call_id": "call_abc123", "content": "18C"},
    ]

    sync_messages: Final = config.transform_messages(messages=copy.deepcopy(history), model="mistral-medium-3")
    async_messages: Final = await config.transform_messages(
        messages=copy.deepcopy(history), model="mistral-medium-3", is_async=True
    )

    assert async_messages == sync_messages
    assert sync_messages[1]["tool_call_id"] != "call_abc123"


@pytest.mark.parametrize("tool_call_id", [None, 12345], ids=["no-id", "integer-id"])
def test_mistral_chat_sends_a_tool_call_whose_id_is_not_a_string_as_is(tool_call_id: object, local_model_cost_map):
    config: Final = ProviderConfigManager.get_provider_chat_config(
        model="mistral-medium-3", provider=LlmProviders.VERTEX_AI
    )
    tool_call: Final = _tool_call(tool_call_id, "get_weather")

    sent: Final = config.transform_messages(
        messages=[
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": tool_call_id, "content": "18C"},
        ],
        model="mistral-medium-3",
    )

    assert sent[0]["tool_calls"] == [tool_call]
    assert sent[1].get("tool_call_id") == tool_call_id


@pytest.mark.parametrize("model", ["vertex_ai/mistral-medium-3", "vertex_ai/mistral-ocr-2505"])
def test_mistral_chat_sends_penalties_and_n_to_every_vertex_mistral_name(
    model: str, local_model_cost_map, monkeypatch: pytest.MonkeyPatch
):
    def send(transport: httpx.MockTransport, _: pytest.MonkeyPatch) -> None:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            presence_penalty=0.1,
            frequency_penalty=0.2,
            n=2,
            vertex_location="us-central1",
            client=HTTPHandler(client=httpx.Client(transport=transport)),
        )

    sent: Final = SENT_BODY.validate_json(_sent_mistral_request(send, monkeypatch).content)

    assert {key: sent.get(key) for key in ("presence_penalty", "frequency_penalty", "n")} == {
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "n": 2,
    }


@pytest.mark.parametrize("model", ["vertex_ai/mistral-medium-3", "vertex_ai/mistral-ocr-2505"])
def test_mistral_chat_sends_several_text_parts_as_separate_parts(
    model: str, local_model_cost_map, monkeypatch: pytest.MonkeyPatch
):
    two_parts: Final = [
        {"type": "text", "text": "apple", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "banana"},
    ]
    messages: Final = [
        {"role": "user", "content": two_parts},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("call_abc123", "get_weather")]},
        {"role": "tool", "tool_call_id": "call_abc123", "content": two_parts},
        {"role": "assistant", "content": two_parts},
        {"role": "user", "content": [{"type": "text", "text": "Repeat it."}]},
    ]

    def send(transport: httpx.MockTransport, _: pytest.MonkeyPatch) -> None:
        litellm.completion(
            model=model,
            messages=messages,
            vertex_location="us-central1",
            client=HTTPHandler(client=httpx.Client(transport=transport)),
        )

    sent_messages: Final = SENT_MAPPINGS.validate_python(
        SENT_BODY.validate_json(_sent_mistral_request(send, monkeypatch).content)["messages"]
    )

    sent_parts: Final = [{"type": "text", "text": "apple"}, {"type": "text", "text": "banana"}]
    assert [message.get("content") for message in sent_messages] == [
        sent_parts,
        None,
        sent_parts,
        sent_parts,
        "Repeat it.",
    ]


def test_mistral_chat_sends_text_parts_without_extra_keys_next_to_an_image(
    local_model_cost_map, monkeypatch: pytest.MonkeyPatch
):
    cached_text: Final = {"type": "text", "text": "Describe it.", "cache_control": {"type": "ephemeral"}}
    image_part: Final = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    messages: Final = [
        {"role": "user", "content": [cached_text, image_part]},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("call_abc123", "look")]},
        {"role": "tool", "tool_call_id": "call_abc123", "content": [cached_text]},
        {"role": "assistant", "content": [cached_text]},
        {"role": "user", "content": "Again."},
    ]

    def send(transport: httpx.MockTransport, _: pytest.MonkeyPatch) -> None:
        litellm.completion(
            model="vertex_ai/mistral-medium-3",
            messages=messages,
            vertex_location="us-central1",
            client=HTTPHandler(client=httpx.Client(transport=transport)),
        )

    sent_messages: Final = SENT_MAPPINGS.validate_python(
        SENT_BODY.validate_json(_sent_mistral_request(send, monkeypatch).content)["messages"]
    )

    bare_text: Final = {"type": "text", "text": "Describe it."}
    assert [message.get("content") for message in sent_messages] == [
        [bare_text, image_part],
        None,
        [bare_text],
        [bare_text],
        "Again.",
    ]


def test_mistral_chat_sends_a_list_with_a_non_text_part_the_way_the_mistral_api_gets_it(local_model_cost_map):
    messages: Final = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe "},
                {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
            ],
        }
    ]
    vertex_config: Final = ProviderConfigManager.get_provider_chat_config(
        model="mistral-medium-3", provider=LlmProviders.VERTEX_AI
    )
    mistral_config: Final = ProviderConfigManager.get_provider_chat_config(
        model="mistral-medium-3", provider=LlmProviders.MISTRAL
    )

    assert vertex_config.transform_messages(
        messages=copy.deepcopy(messages), model="mistral-medium-3"
    ) == mistral_config.transform_messages(messages=copy.deepcopy(messages), model="mistral-medium-3")
