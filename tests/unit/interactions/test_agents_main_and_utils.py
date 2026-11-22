"""
Unit tests for litellm/interactions/agents/utils.py and main.py
focused on the managed agents SDK surface added in the
"Gemini managed agents support" PR.

The tests mock the underlying HTTP handler so they cover the public
sync + async create/list/get/delete/list_versions entry points and the
small helper utilities without touching the network.
"""

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx


import litellm
from litellm.interactions.agents import (
    acreate,
    adelete,
    aget,
    alist,
    alist_versions,
    create,
    delete,
    get,
    list as list_agents,
    list_versions,
)
from litellm.interactions.agents.main import (
    _get_agents_api_config,
    _make_logging_obj,
)
from litellm.interactions.agents.utils import get_provider_agents_api_config
from litellm.llms.base_llm.agents.transformation import BaseAgentsAPIConfig
from litellm.llms.gemini.agents.transformation import GeminiAgentsConfig


_HANDLER_PATH = "litellm.interactions.agents.main.agents_http_handler"


# ---------------------------------------------------------------------------
# utils.get_provider_agents_api_config
# ---------------------------------------------------------------------------


class TestGetProviderAgentsApiConfig:
    def test_returns_gemini_config_for_gemini(self):
        cfg = get_provider_agents_api_config("gemini")
        assert isinstance(cfg, GeminiAgentsConfig)
        assert isinstance(cfg, BaseAgentsAPIConfig)

    @pytest.mark.parametrize(
        "provider", ["openai", "anthropic", "bedrock", "vertex_ai", "unknown"]
    )
    def test_returns_none_for_non_gemini(self, provider):
        assert get_provider_agents_api_config(provider) is None

    def test_returns_none_for_none(self):
        assert get_provider_agents_api_config(None) is None


# ---------------------------------------------------------------------------
# main._get_agents_api_config
# ---------------------------------------------------------------------------


class TestGetAgentsApiConfig:
    def test_returns_config_for_gemini(self):
        cfg = _get_agents_api_config("gemini")
        assert isinstance(cfg, GeminiAgentsConfig)

    def test_raises_bad_request_for_unsupported_provider(self):
        with pytest.raises(litellm.BadRequestError) as excinfo:
            _get_agents_api_config("openai")
        assert "does not have a native" in str(excinfo.value)


# ---------------------------------------------------------------------------
# main._make_logging_obj
# ---------------------------------------------------------------------------


class TestMakeLoggingObj:
    def test_calls_update_from_kwargs_and_returns_same_obj(self):
        logging_obj = MagicMock()
        kwargs = {"litellm_logging_obj": logging_obj, "litellm_call_id": "abc-123"}

        returned = _make_logging_obj(
            kwargs=kwargs,
            model="my-agent",
            custom_llm_provider="gemini",
            call_type="create_agent",
            optional_params={"foo": "bar"},
        )

        assert returned is logging_obj
        logging_obj.update_from_kwargs.assert_called_once()
        kwargs_call = logging_obj.update_from_kwargs.call_args.kwargs
        assert kwargs_call["model"] == "my-agent"
        assert kwargs_call["optional_params"] == {"foo": "bar"}
        assert kwargs_call["custom_llm_provider"] == "gemini"
        assert kwargs_call["litellm_params"]["litellm_call_id"] == "abc-123"


# ---------------------------------------------------------------------------
# Sync entry points: create / list / get / delete / list_versions
# ---------------------------------------------------------------------------


def _stub_handler(return_value):
    """Build a stub AgentsHTTPHandler whose CRUD methods return *return_value*."""
    handler = MagicMock()
    handler.create_agent.return_value = return_value
    handler.list_agents.return_value = return_value
    handler.get_agent.return_value = return_value
    handler.delete_agent.return_value = return_value
    handler.list_agent_versions.return_value = return_value
    return handler


class TestSyncEntryPoints:
    def test_create_passes_args_to_handler(self):
        sentinel = MagicMock(name="create_response")
        with patch(_HANDLER_PATH, _stub_handler(sentinel)) as handler:
            response = create(
                name="waverunner",
                base_agent="gemini-2.5-flash",
                instructions="be helpful",
                base_environment={"type": "remote"},
                custom_llm_provider="gemini",
                api_key="AIza-test",
                extra_headers={"X-Test": "1"},
                extra_body={"foo": "bar"},
            )

        assert response is sentinel
        handler.create_agent.assert_called_once()
        kw = handler.create_agent.call_args.kwargs
        assert kw["create_request"].id == "waverunner"
        assert kw["_is_async"] is False
        assert kw["extra_headers"] == {"X-Test": "1"}
        assert kw["extra_body"] == {"foo": "bar"}
        assert isinstance(kw["agents_api_config"], GeminiAgentsConfig)

    def test_create_defaults_custom_llm_provider_to_gemini(self):
        sentinel = MagicMock(name="create_response")
        with patch(_HANDLER_PATH, _stub_handler(sentinel)) as handler:
            create(name="agent-x", api_key="AIza")
        assert handler.create_agent.call_args.kwargs["_is_async"] is False
        cfg = handler.create_agent.call_args.kwargs["agents_api_config"]
        assert isinstance(cfg, GeminiAgentsConfig)

    def test_create_raises_for_unsupported_provider(self):
        with pytest.raises(litellm.exceptions.BadRequestError):
            create(name="agent-x", custom_llm_provider="openai", api_key="sk-x")

    def test_list_passes_args_to_handler(self):
        sentinel = MagicMock(name="list_response")
        with patch(_HANDLER_PATH, _stub_handler(sentinel)) as handler:
            response = list_agents(custom_llm_provider="gemini", api_key="AIza")
        assert response is sentinel
        handler.list_agents.assert_called_once()
        assert handler.list_agents.call_args.kwargs["_is_async"] is False

    def test_get_passes_args_to_handler(self):
        sentinel = MagicMock(name="get_response")
        with patch(_HANDLER_PATH, _stub_handler(sentinel)) as handler:
            response = get(name="waverunner", api_key="AIza")
        assert response is sentinel
        kw = handler.get_agent.call_args.kwargs
        assert kw["name"] == "waverunner"
        assert kw["_is_async"] is False

    def test_delete_passes_args_to_handler(self):
        sentinel = MagicMock(name="delete_response")
        with patch(_HANDLER_PATH, _stub_handler(sentinel)) as handler:
            response = delete(name="waverunner", api_key="AIza")
        assert response is sentinel
        kw = handler.delete_agent.call_args.kwargs
        assert kw["name"] == "waverunner"
        assert kw["_is_async"] is False

    def test_list_versions_passes_args_to_handler(self):
        sentinel = MagicMock(name="versions_response")
        with patch(_HANDLER_PATH, _stub_handler(sentinel)) as handler:
            response = list_versions(name="waverunner", api_key="AIza")
        assert response is sentinel
        kw = handler.list_agent_versions.call_args.kwargs
        assert kw["name"] == "waverunner"
        assert kw["_is_async"] is False


# ---------------------------------------------------------------------------
# Async entry points
# ---------------------------------------------------------------------------


class TestAsyncEntryPoints:
    """Async entry points delegate to their sync counterparts via run_in_executor."""

    @pytest.mark.asyncio
    async def test_acreate_dispatches_with_async_flag(self):
        sentinel = MagicMock(name="acreate_response")

        def fake_create_agent(**kwargs):
            assert kwargs["_is_async"] is True
            assert kwargs["create_request"].id == "waverunner"
            return sentinel

        handler = MagicMock()
        handler.create_agent.side_effect = fake_create_agent

        with patch(_HANDLER_PATH, handler):
            response = await acreate(
                name="waverunner",
                base_agent="gemini-2.5-flash",
                api_key="AIza",
            )
        assert response is sentinel

    @pytest.mark.asyncio
    async def test_acreate_awaits_coroutine_result(self):
        async def _coro():
            return "async-value"

        handler = MagicMock()
        handler.create_agent.return_value = _coro()

        with patch(_HANDLER_PATH, handler):
            response = await acreate(name="waverunner", api_key="AIza")

        assert response == "async-value"

    @pytest.mark.asyncio
    async def test_alist_dispatches_with_async_flag(self):
        sentinel = MagicMock(name="alist_response")

        def fake_list_agents(**kwargs):
            assert kwargs["_is_async"] is True
            return sentinel

        handler = MagicMock()
        handler.list_agents.side_effect = fake_list_agents

        with patch(_HANDLER_PATH, handler):
            response = await alist(api_key="AIza")
        assert response is sentinel

    @pytest.mark.asyncio
    async def test_aget_dispatches_with_async_flag(self):
        sentinel = MagicMock(name="aget_response")

        def fake_get_agent(**kwargs):
            assert kwargs["_is_async"] is True
            assert kwargs["name"] == "waverunner"
            return sentinel

        handler = MagicMock()
        handler.get_agent.side_effect = fake_get_agent

        with patch(_HANDLER_PATH, handler):
            response = await aget(name="waverunner", api_key="AIza")
        assert response is sentinel

    @pytest.mark.asyncio
    async def test_adelete_dispatches_with_async_flag(self):
        sentinel = MagicMock(name="adelete_response")

        def fake_delete_agent(**kwargs):
            assert kwargs["_is_async"] is True
            assert kwargs["name"] == "waverunner"
            return sentinel

        handler = MagicMock()
        handler.delete_agent.side_effect = fake_delete_agent

        with patch(_HANDLER_PATH, handler):
            response = await adelete(name="waverunner", api_key="AIza")
        assert response is sentinel

    @pytest.mark.asyncio
    async def test_alist_versions_dispatches_with_async_flag(self):
        sentinel = MagicMock(name="alist_versions_response")

        def fake_versions(**kwargs):
            assert kwargs["_is_async"] is True
            assert kwargs["name"] == "waverunner"
            return sentinel

        handler = MagicMock()
        handler.list_agent_versions.side_effect = fake_versions

        with patch(_HANDLER_PATH, handler):
            response = await alist_versions(name="waverunner", api_key="AIza")
        assert response is sentinel


# ---------------------------------------------------------------------------
# Async error wrapping: exception_type must be invoked
# ---------------------------------------------------------------------------


class TestAsyncErrorWrapping:
    """If the underlying handler raises, async entry points re-raise via
    litellm.exception_type so users get a normalised provider error."""

    @pytest.mark.asyncio
    async def test_acreate_wraps_exception(self):
        handler = MagicMock()
        handler.create_agent.side_effect = RuntimeError("kaboom")

        with patch(_HANDLER_PATH, handler):
            with pytest.raises(litellm.APIConnectionError):
                await acreate(name="waverunner", api_key="AIza")

    @pytest.mark.asyncio
    async def test_aget_wraps_exception(self):
        handler = MagicMock()
        handler.get_agent.side_effect = RuntimeError("kaboom")

        with patch(_HANDLER_PATH, handler):
            with pytest.raises(litellm.APIConnectionError):
                await aget(name="waverunner", api_key="AIza")

    @pytest.mark.asyncio
    async def test_alist_wraps_exception(self):
        handler = MagicMock()
        handler.list_agents.side_effect = RuntimeError("kaboom")

        with patch(_HANDLER_PATH, handler):
            with pytest.raises(litellm.APIConnectionError):
                await alist(api_key="AIza")

    @pytest.mark.asyncio
    async def test_adelete_wraps_exception(self):
        handler = MagicMock()
        handler.delete_agent.side_effect = RuntimeError("kaboom")

        with patch(_HANDLER_PATH, handler):
            with pytest.raises(litellm.APIConnectionError):
                await adelete(name="waverunner", api_key="AIza")

    @pytest.mark.asyncio
    async def test_alist_versions_wraps_exception(self):
        handler = MagicMock()
        handler.list_agent_versions.side_effect = RuntimeError("kaboom")

        with patch(_HANDLER_PATH, handler):
            with pytest.raises(litellm.APIConnectionError):
                await alist_versions(name="waverunner", api_key="AIza")


_GEMINI_AGENTS_URL: Final = "https://generativelanguage.googleapis.com/v1beta/agents"


def _sent_json(route: respx.Route) -> object:
    return json.loads(route.calls.last.request.content)


class TestCreateRequestBody:
    def test_create_sends_every_current_field_verbatim(self, respx_mock: respx.MockRouter) -> None:
        route: Final = respx_mock.post(_GEMINI_AGENTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "sentinel", "base_agent": "antigravity-preview-09-2026"})
        )

        response: Final = create(
            id="sentinel",
            base_agent="antigravity-preview-09-2026",
            agent_config={"type": "antigravity", "model": "gemini-3.8-flash", "max_total_tokens": None},
            system_instruction="Always answer in exactly three words.",
            base_environment={"type": "remote", "sources": [{"type": "gcs", "source": "gs://bucket/skill"}]},
            description="Watches the queue.",
            tools=[{"type": "code_execution"}],
            api_key="AIza-test",
        )

        assert _sent_json(route) == {
            "id": "sentinel",
            "base_agent": "antigravity-preview-09-2026",
            "agent_config": {"type": "antigravity", "model": "gemini-3.8-flash", "max_total_tokens": None},
            "system_instruction": "Always answer in exactly three words.",
            "base_environment": {"type": "remote", "sources": [{"type": "gcs", "source": "gs://bucket/skill"}]},
            "description": "Watches the queue.",
            "tools": [{"type": "code_execution"}],
        }
        assert (response.id, response.name) == ("sentinel", "sentinel")

    @pytest.mark.parametrize(
        ("identifier", "instruction", "expected"),
        [
            pytest.param(
                {"name": "slides"},
                {"instructions": "Make slides."},
                {"id": "slides", "system_instruction": "Make slides."},
                id="preview-aliases",
            ),
            pytest.param(
                {"id": "slides", "name": "slides"},
                {"system_instruction": "Make slides.", "instructions": "Make slides."},
                {"id": "slides", "system_instruction": "Make slides."},
                id="same-value-under-both-names",
            ),
        ],
    )
    def test_create_sends_preview_aliases_under_current_names(
        self,
        respx_mock: respx.MockRouter,
        identifier: Mapping[str, str],
        instruction: Mapping[str, str],
        expected: Mapping[str, str],
    ) -> None:
        route: Final = respx_mock.post(_GEMINI_AGENTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "slides"})
        )

        create(**identifier, **instruction, base_agent="antigravity-preview-09-2026", api_key="AIza-test")

        assert _sent_json(route) == {**expected, "base_agent": "antigravity-preview-09-2026"}

    @pytest.mark.parametrize(
        "fields",
        [
            pytest.param({"base_agent": "antigravity-preview-09-2026"}, id="no-identifier"),
            pytest.param({"id": "slides", "name": "decks"}, id="different-id-and-name"),
            pytest.param(
                {"id": "slides", "system_instruction": "Make slides.", "instructions": "Make decks."},
                id="different-system-instruction-and-instructions",
            ),
            pytest.param({"id": "slides", "agent_config": "antigravity"}, id="agent-config-not-an-object"),
        ],
    )
    def test_create_rejects_a_malformed_identifier_or_instruction(
        self, respx_mock: respx.MockRouter, fields: Mapping[str, str]
    ) -> None:
        with pytest.raises(litellm.BadRequestError) as exc_info:
            create(**fields, api_key="AIza-test")

        assert exc_info.value.status_code == 400
        assert not respx_mock.calls

    def test_create_with_drop_params_drops_only_the_invalid_optional_fields(
        self, respx_mock: respx.MockRouter
    ) -> None:
        route: Final = respx_mock.post(_GEMINI_AGENTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "slides"})
        )

        create(
            id="slides",
            agent_config="antigravity",
            tools="code_execution",
            instructions="Make slides.",
            drop_params=True,
            api_key="AIza-test",
        )

        assert _sent_json(route) == {"id": "slides", "system_instruction": "Make slides."}

    def test_create_with_drop_params_drops_a_clashing_instructions_alias(self, respx_mock: respx.MockRouter) -> None:
        route: Final = respx_mock.post(_GEMINI_AGENTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "slides"})
        )

        create(
            id="slides",
            system_instruction="Make slides.",
            instructions="Make decks.",
            drop_params=True,
            api_key="AIza-test",
        )

        assert _sent_json(route) == {"id": "slides", "system_instruction": "Make slides."}

    def test_create_with_drop_params_still_rejects_a_clashing_identifier(self, respx_mock: respx.MockRouter) -> None:
        with pytest.raises(litellm.BadRequestError) as exc_info:
            create(id="slides", name="decks", drop_params=True, api_key="AIza-test")

        assert exc_info.value.status_code == 400
        assert not respx_mock.calls

    @pytest.mark.parametrize(
        ("fields", "expected_message"),
        [
            pytest.param(
                {"id": "slides", "instructions": 5, "tools": "code_execution"},
                "Invalid agent create request: 'instructions' must be a string; 'tools' must be a list of objects."
                " Set drop_params to drop invalid optional fields.",
                id="names-the-alias-the-caller-sent",
            ),
            pytest.param(
                {"name": 7, "base_agent": "antigravity-preview-09-2026"},
                "Invalid agent create request: 'name' must be a string.",
                id="names-the-id-alias-the-caller-sent",
            ),
            pytest.param(
                {"id": 7, "agent_config": "antigravity", "drop_params": True},
                "Invalid agent create request: 'agent_config' must be an object; 'id' must be a string.",
                id="invalid-id-is-never-dropped",
            ),
        ],
    )
    def test_create_rejects_invalid_fields_by_the_names_the_caller_sent(
        self, respx_mock: respx.MockRouter, fields: Mapping[str, object], expected_message: str
    ) -> None:
        with pytest.raises(litellm.BadRequestError) as exc_info:
            create(**fields, api_key="AIza-test")

        assert exc_info.value.status_code == 400
        assert exc_info.value.message.endswith(expected_message)
        assert not respx_mock.calls

    @pytest.mark.asyncio
    async def test_acreate_sends_every_current_field_verbatim(
        self, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
        route: Final = respx_mock.post(_GEMINI_AGENTS_URL).mock(
            return_value=httpx.Response(200, json={"id": "sentinel"})
        )

        response: Final = await acreate(
            id="sentinel",
            agent_config={"type": "antigravity", "model": "gemini-3.8-flash"},
            system_instruction="Always answer in exactly three words.",
            api_key="AIza-test",
        )

        assert _sent_json(route) == {
            "id": "sentinel",
            "agent_config": {"type": "antigravity", "model": "gemini-3.8-flash"},
            "system_instruction": "Always answer in exactly three words.",
        }
        assert response.id == "sentinel"


class TestAgentIdentifier:
    def test_get_by_id_fetches_that_agent(self, respx_mock: respx.MockRouter) -> None:
        route: Final = respx_mock.get(f"{_GEMINI_AGENTS_URL}/slides").mock(
            return_value=httpx.Response(200, json={"id": "slides", "base_agent": "antigravity-preview-09-2026"})
        )

        response: Final = get(id="slides", api_key="AIza-test")

        assert route.call_count == 1
        assert (response.id, response.base_agent) == ("slides", "antigravity-preview-09-2026")

    def test_delete_by_id_deletes_that_agent(self, respx_mock: respx.MockRouter) -> None:
        route: Final = respx_mock.delete(f"{_GEMINI_AGENTS_URL}/slides").mock(
            return_value=httpx.Response(200, json={})
        )

        result: Final = delete(id="slides", api_key="AIza-test")

        assert route.call_count == 1
        assert (result.name, result.deleted) == ("slides", True)

    def test_list_versions_by_id_lists_that_agents_versions(self, respx_mock: respx.MockRouter) -> None:
        version: Final = {"agent": "slides", "name": "agents/slides/versions/v1"}
        route: Final = respx_mock.get(f"{_GEMINI_AGENTS_URL}/slides/versions").mock(
            return_value=httpx.Response(200, json={"agentVersions": [version]})
        )

        response: Final = list_versions(id="slides", api_key="AIza-test")

        assert route.call_count == 1
        assert response.agent_versions == [version]

    @pytest.mark.parametrize(
        "operation",
        [pytest.param(get, id="get"), pytest.param(delete, id="delete"), pytest.param(list_versions, id="list_versions")],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            pytest.param({}, id="no-identifier"),
            pytest.param({"id": "slides", "name": "decks"}, id="different-id-and-name"),
        ],
    )
    def test_rejects_a_missing_or_conflicting_identifier(
        self,
        respx_mock: respx.MockRouter,
        operation: Callable[..., object],
        identifier: Mapping[str, str],
    ) -> None:
        with pytest.raises(litellm.BadRequestError) as exc_info:
            operation(**identifier, api_key="AIza-test")

        assert exc_info.value.status_code == 400
        assert not respx_mock.calls

    @pytest.mark.asyncio
    async def test_aget_rejects_a_missing_identifier(self, respx_mock: respx.MockRouter) -> None:
        with pytest.raises(litellm.BadRequestError) as exc_info:
            await aget(api_key="AIza-test")

        assert exc_info.value.status_code == 400
        assert not respx_mock.calls
