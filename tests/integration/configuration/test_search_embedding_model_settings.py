from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final
from zipfile import ZipFile

import pytest
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120))

_MODEL_NAME: Final = "search-embedding"
_AGENT_NOT_CONFIGURED: Final = (
    "agent search needs litellm_settings.agent_search_embedding_model set to an embedding model from model_list"
)
_SKILL_NOT_CONFIGURED: Final = (
    "skill search needs litellm_settings.skill_search_embedding_model set to an embedding model from model_list"
)
_SKILL_MARKDOWN: Final = """---
name: echo-sample
description: Echo text for search validation
---

# Echo sample

Return the supplied text unchanged
"""
_JSON_ARRAY: Final = TypeAdapter(list[JsonValue])


@dataclass(frozen=True, slots=True)
class _SearchProxy:
    owned: OwnedProxy
    wire: Wire
    skill_id: str
    agent_id: str


def _embedding_reply(request: Request) -> Reply:
    if request.method != "POST" or request.target != "/v1/embeddings":
        return Reply(body=b'{"object":"list","data":[]}')
    body: Final = JSON_OBJECT.validate_json(request.body)
    input_value: Final = body.get("input")
    input_count: Final = len(input_value) if isinstance(input_value, list) else 1
    embeddings: Final = [
        {"object": "embedding", "index": index, "embedding": [0.25, 0.75]} for index in range(input_count)
    ]
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": embeddings,
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": input_count, "total_tokens": input_count},
            }
        ).encode()
    )


def _write_config(directory: Path, upstream_url: str, agent_model: str, skill_model: str) -> Path:
    config: Final = directory / f"search_embedding_{uuid.uuid4().hex}.yaml"
    contents: Final[dict[str, JsonValue]] = {
        "model_list": [
            {
                "model_name": _MODEL_NAME,
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_base": f"{upstream_url}/v1",
                    "api_key": "synthetic-openai-key",
                },
            }
        ],
        "litellm_settings": {
            "agent_search_embedding_model": agent_model,
            "skill_search_embedding_model": skill_model,
        },
        "general_settings": {
            "master_key": "os.environ/LITELLM_MASTER_KEY",
            "database_url": "os.environ/DATABASE_URL",
            "store_model_in_db": True,
        },
        "agents": [
            {
                "agent_name": "plain",
                "agent_card_params": {
                    "protocolVersion": "1.0",
                    "name": "plain",
                    "description": "echo agent",
                    "url": "http://agent:9000/plain/",
                    "supportedInterfaces": [
                        {
                            "url": "http://agent:9000/plain/",
                            "protocolBinding": "JSONRPC",
                            "protocolVersion": "1.0",
                        }
                    ],
                    "version": "1.0.0",
                    "capabilities": {"streaming": True},
                    "defaultInputModes": ["text/plain"],
                    "defaultOutputModes": ["text/plain"],
                    "skills": [],
                },
            }
        ],
    }
    config.write_text(json.dumps(contents))
    return config


def _create_skill(candidate: OwnedProxy) -> str:
    archive_data: Final = BytesIO()
    with ZipFile(archive_data, "w") as archive:
        archive.writestr("echo-sample/SKILL.md", _SKILL_MARKDOWN)
    response: Final = candidate.gateway.request_multipart(
        "/v1/skills?beta=true&custom_llm_provider=litellm_proxy",
        {
            "display_title": "Echo search fixture",
            "description": "Echo sample skill for search validation",
            "instructions": "Return the supplied text unchanged",
        },
        {"files[]": ("echo-skill.zip", archive_data.getvalue(), "application/zip")},
    )
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])


def _agent_id(candidate: OwnedProxy) -> str:
    response: Final = candidate.gateway.request("GET", "/v1/agents")
    assert response.status_code == 200, response.text
    agents: Final = _JSON_ARRAY.validate_json(response.content)
    assert len(agents) == 1
    return string_value(object_value(agents[0])["agent_id"])


@pytest.fixture(scope="module")
def empty_search_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_SearchProxy]:
    directory: Final = tmp_path_factory.mktemp("search-empty-model")
    with wire_server(_embedding_reply) as upstream, gateway_from_environment() as gateway:
        config: Final = _write_config(
            directory,
            upstream.url,
            "os.environ/AGENT_SEARCH_EMBEDDING_MODEL",
            "os.environ/SKILL_SEARCH_EMBEDDING_MODEL",
        )
        with owned_proxy_process(
            gateway,
            directory,
            {"AGENT_SEARCH_EMBEDDING_MODEL": "", "SKILL_SEARCH_EMBEDDING_MODEL": ""},
            config=config,
        ) as owned:
            yield _SearchProxy(owned, upstream, _create_skill(owned), _agent_id(owned))


@pytest.fixture(scope="module")
def configured_search_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_SearchProxy]:
    directory: Final = tmp_path_factory.mktemp("search-configured-model")
    with wire_server(_embedding_reply) as upstream, gateway_from_environment() as gateway:
        config: Final = _write_config(directory, upstream.url, _MODEL_NAME, _MODEL_NAME)
        with owned_proxy_process(
            gateway,
            directory,
            {"AGENT_SEARCH_EMBEDDING_MODEL": _MODEL_NAME, "SKILL_SEARCH_EMBEDDING_MODEL": _MODEL_NAME},
            config=config,
        ) as owned:
            yield _SearchProxy(owned, upstream, _create_skill(owned), _agent_id(owned))


def _assert_not_configured(response_body: JsonValue, expected_error: str, reason: str) -> None:
    detail: Final = object_value(object_value(response_body)["detail"])
    assert detail == {"error": expected_error, "message": reason}


def _assert_no_embedding_requests(candidate: _SearchProxy) -> None:
    embedding_requests: Final = tuple(
        request for request in candidate.wire.drain() if request.method == "POST" and request.target == "/v1/embeddings"
    )
    assert embedding_requests == ()


def _assert_mcp_search_not_configured(gateway: Gateway, key: str, tool_name: str, reason: str) -> None:
    response: Final = gateway.request(
        "POST",
        "/mcp-rest/tools/call",
        {"name": tool_name, "arguments": {"query": "echo"}},
        key=key,
    )
    assert response.status_code == 200, response.text
    result: Final = JSON_OBJECT.validate_json(response.content)
    assert result.get("isError") is True, response.text
    content: Final = object_value(list_value(result["content"])[0])
    text: Final = string_value(content["text"])
    assert reason in text
    assert "You passed in model=" not in text


def test_empty_agent_embedding_setting_returns_not_configured(empty_search_proxy: _SearchProxy) -> None:
    response: Final = empty_search_proxy.owned.gateway.request("GET", "/v1/agents?query=echo")
    assert response.status_code == 400, response.text
    _assert_not_configured(
        JSON_OBJECT.validate_json(response.content),
        "agent_search_not_configured",
        _AGENT_NOT_CONFIGURED,
    )
    _assert_no_embedding_requests(empty_search_proxy)


def test_empty_skill_embedding_setting_returns_not_configured(empty_search_proxy: _SearchProxy) -> None:
    response: Final = empty_search_proxy.owned.gateway.request(
        "GET", "/v1/skills?custom_llm_provider=litellm_proxy&query=echo"
    )
    assert response.status_code == 400, response.text
    _assert_not_configured(
        JSON_OBJECT.validate_json(response.content),
        "skill_search_not_configured",
        _SKILL_NOT_CONFIGURED,
    )
    _assert_no_embedding_requests(empty_search_proxy)


def test_empty_agent_embedding_setting_returns_not_configured_through_mcp(empty_search_proxy: _SearchProxy) -> None:
    with empty_search_proxy.owned.gateway.scenario() as scenario:
        key: Final = scenario.key(
            object_permission={
                "agents": [empty_search_proxy.agent_id],
                "mcp_tool_search_enabled": True,
            },
        )
        _assert_mcp_search_not_configured(
            empty_search_proxy.owned.gateway,
            key,
            "agent_search",
            _AGENT_NOT_CONFIGURED,
        )
    _assert_no_embedding_requests(empty_search_proxy)


def test_empty_skill_embedding_setting_returns_not_configured_through_mcp(empty_search_proxy: _SearchProxy) -> None:
    with empty_search_proxy.owned.gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="proxy_admin")
        key: Final = scenario.key(user_id=user_id, object_permission={"mcp_tool_search_enabled": True})
        _assert_mcp_search_not_configured(
            empty_search_proxy.owned.gateway,
            key,
            "skill_search",
            _SKILL_NOT_CONFIGURED,
        )
    _assert_no_embedding_requests(empty_search_proxy)


def test_configured_agent_embedding_model_returns_search_score(configured_search_proxy: _SearchProxy) -> None:
    configured_search_proxy.wire.drain()
    response: Final = configured_search_proxy.owned.gateway.request("GET", "/v1/agents?query=echo")
    assert response.status_code == 200, response.text
    agents: Final = _JSON_ARRAY.validate_json(response.content)
    assert len(agents) == 1
    agent: Final = object_value(agents[0])
    assert string_value(agent["agent_name"]) == "plain"
    assert isinstance(agent["search_score"], (int, float))
    embedding_requests: Final = tuple(
        request
        for request in configured_search_proxy.wire.drain()
        if request.method == "POST" and request.target == "/v1/embeddings"
    )
    assert embedding_requests


def test_configured_skill_embedding_model_returns_search_score(configured_search_proxy: _SearchProxy) -> None:
    configured_search_proxy.wire.drain()
    response: Final = configured_search_proxy.owned.gateway.request(
        "GET", "/v1/skills?custom_llm_provider=litellm_proxy&query=echo"
    )
    assert response.status_code == 200, response.text
    skills: Final[Sequence[JsonValue]] = list_value(JSON_OBJECT.validate_json(response.content)["data"])
    matching_skills: Final = tuple(
        object_value(skill) for skill in skills if object_value(skill)["id"] == configured_search_proxy.skill_id
    )
    assert len(matching_skills) == 1
    assert isinstance(matching_skills[0]["search_score"], (int, float))
    embedding_requests: Final = tuple(
        request
        for request in configured_search_proxy.wire.drain()
        if request.method == "POST" and request.target == "/v1/embeddings"
    )
    assert embedding_requests
