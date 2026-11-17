"""Contract: chat completions on IBM watsonx.ai (`watsonx/...`) through the proxy, against a loopback peer.

watsonx's `/text/chat` endpoint is OpenAI-like, but the URL, auth and body carry IBM specifics a port must keep:

- URL: `<api_base>/ml/v1/text/chat?version=2024-03-13` (`/ml/v1/text/chat_stream` when streaming); a
  `deployment/<id>` model goes to `/ml/v1/deployments/<id>/text/chat`; `api_version` replaces the version query;
- auth: `Authorization: ZenApiKey <key>` when `zen_api_key` is set, `Bearer <token>` when a ready token is set,
  otherwise an IAM token exchange (not exercised here, it needs an IAM peer);
- body: `model_id` and `model` are the model name without the `watsonx/` prefix, and `project_id` (or
  `space_id`) rides in the body; a deployment model carries neither;
- params: `tool_choice` `auto`/`none`/`required` is sent as `tool_choice_option`, a named tool stays
  `tool_choice`; tools lose `strict` and `additionalProperties`;
- a deployment without project and space id is refused 401 before any call.

Stream usage and request id headers for watsonx are covered in `streaming/test_openai_like_stream_usage_wire.py`
and `providers/test_openai_like_response_headers_wire.py`.
"""

from __future__ import annotations

from typing import Final

from integration._support.client import Gateway, Scenario, list_value, object_value
from integration.providers.dec1_otherprov_support import (
    TOOL,
    Peer,
    chat,
    deployment,
    peer,
    served,
    stream_chunks,
)
from pydantic import JsonValue

_MODEL: Final = "watsonx/ibm/granite-3-8b-instruct"
_PROJECT: Final = "dec1-project-0001"
_ZEN_KEY: Final = "dec1-zen-key"


def _watsonx(scenario: Scenario, provider: Peer, model: str = _MODEL, **extra: JsonValue) -> str:
    """A watsonx deployment authenticated with a ZenApiKey and a project id, pointed at the peer."""
    return deployment(
        scenario,
        provider,
        model,
        with_key=False,
        zen_api_key=_ZEN_KEY,
        project_id=_PROJECT,
        **extra,
    )


def test_chat_posts_to_text_chat_with_the_default_version_query(gateway: Gateway) -> None:
    """A watsonx deployment with `api_base` the bare peer URL.

    The peer receives `POST /ml/v1/text/chat` with query `version=2024-03-13`
    (`WatsonXAIEndpoint.CHAT`, `types/llms/watsonx.py:54`; `_add_api_version_to_url`, `watsonx/common_utils.py:291-295`;
    default `WATSONX_DEFAULT_API_VERSION`, `litellm/__init__.py:340`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = _watsonx(scenario, provider)
        served(chat(gateway, model))
        call: Final = provider.only()
        assert (call.method, call.path) == ("POST", "/ml/v1/text/chat")
        assert call.query["version"] == ["2024-03-13"], call.query


def test_zen_api_key_is_sent_as_a_zenapikey_authorization(gateway: Gateway) -> None:
    """With `zen_api_key` set and no token, the peer sees `authorization: ZenApiKey dec1-zen-key`
    (`IBMWatsonXMixin.validate_environment`, `watsonx/common_utils.py:257-268`), and the key is not in the body
    (it is popped from the params)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = _watsonx(scenario, provider)
        served(chat(gateway, model))
        call: Final = provider.only()
        assert call.headers["authorization"] == f"ZenApiKey {_ZEN_KEY}"
        assert "zen_api_key" not in call.body, call.body


def test_a_ready_token_is_sent_as_bearer_without_an_iam_exchange(gateway: Gateway) -> None:
    """A deployment with `token: "dec1-ready-token"` (and no zen key) sends `authorization: Bearer dec1-ready-token`
    and makes no IAM call, since a token wins over every other credential
    (`watsonx/common_utils.py:257-266`). UNVERIFIED: that a deployment-level `token` reaches `optional_params`;
    the peer would otherwise see an IAM-derived bearer.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, _MODEL, with_key=False, token="dec1-ready-token", project_id=_PROJECT
        )
        served(chat(gateway, model))
        assert provider.only().headers["authorization"] == "Bearer dec1-ready-token"


def test_body_carries_model_id_and_project_id(gateway: Gateway) -> None:
    """The peer body has `model_id: "ibm/granite-3-8b-instruct"`, `model` equal to the same name, and
    `project_id: "dec1-project-0001"` (`_prepare_payload`, `watsonx/common_utils.py:351-360`; the handler sends
    `model_id` as `model`, `watsonx/chat/handler.py:76-77`), with no `space_id`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = _watsonx(scenario, provider)
        served(chat(gateway, model))
        sent: Final = provider.only().body
        assert (sent["model_id"], sent["model"], sent["project_id"]) == (
            "ibm/granite-3-8b-instruct",
            "ibm/granite-3-8b-instruct",
            _PROJECT,
        ), sent
        assert "space_id" not in sent, sent


def test_space_id_is_sent_when_there_is_no_project(gateway: Gateway) -> None:
    """A deployment with `space_id: "dec1-space"` and no project id sends `space_id` in the body and no
    `project_id` (`watsonx/common_utils.py:355-359`). UNVERIFIED: assumes the proxy has no `WATSONX_PROJECT_ID`,
    `WX_PROJECT_ID` or `PROJECT_ID` in its environment."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(
            scenario, provider, _MODEL, with_key=False, zen_api_key=_ZEN_KEY, space_id="dec1-space"
        )
        served(chat(gateway, model))
        sent: Final = provider.only().body
        assert sent["space_id"] == "dec1-space" and "project_id" not in sent, sent


def test_deployment_model_goes_to_the_deployment_chat_path_without_ids(gateway: Gateway) -> None:
    """`watsonx/deployment/dec1-dep-42` posts to `/ml/v1/deployments/dec1-dep-42/text/chat?version=2024-03-13`
    (`get_complete_url`, `watsonx/chat/transformation.py:93-98`) and the body has neither `model_id` nor
    `project_id` (`_prepare_payload` returns `{}` for deployments, `watsonx/common_utils.py:353-354`)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = _watsonx(scenario, provider, "watsonx/deployment/dec1-dep-42")
        served(chat(gateway, model))
        call: Final = provider.only()
        assert call.path == "/ml/v1/deployments/dec1-dep-42/text/chat", call.path
        assert call.query["version"] == ["2024-03-13"]
        assert "model_id" not in call.body and "project_id" not in call.body, call.body


def test_api_version_replaces_the_default_version_query(gateway: Gateway) -> None:
    """A deployment with `api_version: "2025-02-11"` sends `?version=2025-02-11` and no `api_version` body key
    (`get_complete_url` pops it, `watsonx/chat/transformation.py:104`). UNVERIFIED: that `api_version` reaches
    `optional_params` for watsonx rather than staying a litellm param (then the default would be sent)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = _watsonx(scenario, provider, api_version="2025-02-11")
        served(chat(gateway, model))
        call: Final = provider.only()
        assert call.query["version"] == ["2025-02-11"], call.query
        assert "api_version" not in call.body


def test_stream_posts_to_chat_stream(gateway: Gateway) -> None:
    """A streamed call goes to `/ml/v1/text/chat_stream?version=2024-03-13` (`WatsonXAIEndpoint.CHAT_STREAM`,
    `types/llms/watsonx.py:55`) and the caller receives the peer's content."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = _watsonx(scenario, provider)
        chunks: Final = stream_chunks(gateway, model)
        assert chunks, chunks
        assert provider.only().path == "/ml/v1/text/chat_stream"


def test_required_tool_choice_is_sent_as_tool_choice_option(gateway: Gateway) -> None:
    """`tool_choice: "required"` is a tool choice option for watsonx, so the body has
    `tool_choice_option: "required"` and no `tool_choice` (`IBMWatsonXChatConfig.map_openai_params`,
    `watsonx/chat/transformation.py:43-73`)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = _watsonx(scenario, provider)
        served(chat(gateway, model, tools=[TOOL], tool_choice="required"))
        sent: Final = provider.only().body
        assert sent["tool_choice_option"] == "required" and "tool_choice" not in sent, sent


def test_named_tool_choice_stays_tool_choice(gateway: Gateway) -> None:
    """A named choice `{"type": "function", "function": {"name": "get_weather"}}` is not an option string, so it is
    sent unchanged as `tool_choice` and there is no `tool_choice_option` (`watsonx/chat/transformation.py:69-73`)."""
    named: Final = {"type": "function", "function": {"name": "get_weather"}}
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = _watsonx(scenario, provider)
        served(chat(gateway, model, tools=[TOOL], tool_choice=named))
        sent: Final = provider.only().body
        assert sent["tool_choice"] == named and "tool_choice_option" not in sent, sent


def test_tools_lose_strict_and_additional_properties(gateway: Gateway) -> None:
    """The caller's tool has `strict: true` and `additionalProperties: false`; the body's tool has neither, while
    the name and properties stay (`remove_additional_properties`, `remove_strict_from_schema`,
    `watsonx/chat/transformation.py:58-65`)."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = _watsonx(scenario, provider)
        served(chat(gateway, model, tools=[TOOL]))
        function: Final = object_value(object_value(list_value(provider.only().body["tools"])[0])["function"])
        assert "strict" not in function and function["name"] == "get_weather", function
        assert "additionalProperties" not in object_value(function["parameters"]), function


def test_missing_project_and_space_is_refused_before_any_call(gateway: Gateway) -> None:
    """A watsonx deployment with a zen key but no project id or space id is refused with 401 and the message
    `Watsonx project_id and space_id not set` (`get_api_params`, `watsonx/common_utils.py:122-126`); the peer is
    never called. UNVERIFIED: assumes no project or space id in the proxy environment."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL, with_key=False, zen_api_key=_ZEN_KEY)
        response: Final = chat(gateway, model)
        assert response.status_code == 401, response.text
        assert "Watsonx project_id and space_id not set" in response.text
        assert provider.posts() == ()
