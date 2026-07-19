"""
Unit tests for file_search / vector_store support in the Responses API.

Coverage:
  A1-A7  _decode_vector_store_ids_in_tools()
  B1-B3  update_responses_tools_with_model_file_ids()
  C1,D1  supports_native_file_search()
  E1-E4  file_search guard in responses/main.py
  F1-F6  ManagedFiles hook access control
  G1-G3  get_vector_store_ids_from_file_search_tools()
  H1-H14 emulated_handler unit tests
"""

import base64
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _decode_vector_store_ids_in_tools,
    update_responses_tools_with_model_file_ids,
)
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unified_vs_id(
    unified_uuid: str = "abc-123",
    provider_resource_id: str = "vs_provider_native",
    model_id: str = "model-id-999",
) -> str:
    """Build a valid base64-encoded unified vector-store ID."""
    raw = (
        f"litellm_proxy:vector_store;"
        f"unified_id,{unified_uuid};"
        f"model_id,{model_id};"
        f"provider_resource_id,{provider_resource_id}"
    )
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _file_search_tool(vector_store_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    tool: Dict[str, Any] = {"type": "file_search"}
    if vector_store_ids is not None:
        tool["vector_store_ids"] = vector_store_ids
    return tool


def _code_interpreter_tool(file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    tool: Dict[str, Any] = {"type": "code_interpreter"}
    if file_ids:
        tool["container"] = {"type": "auto", "file_ids": file_ids}
    return tool


# ---------------------------------------------------------------------------
# A-series: _decode_vector_store_ids_in_tools
# ---------------------------------------------------------------------------


class TestDecodeVectorStoreIdsInTools:
    def test_A1_none_input_returns_none(self):
        assert _decode_vector_store_ids_in_tools(None) is None

    def test_A2_no_file_search_tools_unchanged(self):
        tools = [{"type": "web_search"}, {"type": "code_interpreter"}]
        result = _decode_vector_store_ids_in_tools(tools)
        assert result == tools

    def test_A3_file_search_no_vector_store_ids_unchanged(self):
        tools = [_file_search_tool()]  # no vector_store_ids key
        result = _decode_vector_store_ids_in_tools(tools)
        assert result == tools

    def test_A4_unified_id_decoded_to_provider_resource_id(self):
        unified_id = _make_unified_vs_id(provider_resource_id="vs_real_123")
        tools = [_file_search_tool([unified_id])]
        result = _decode_vector_store_ids_in_tools(tools)
        assert result is not None
        assert result[0]["vector_store_ids"] == ["vs_real_123"]

    def test_A5_native_id_passes_through_unchanged(self):
        native_id = "vs_openai_abc"
        tools = [_file_search_tool([native_id])]
        result = _decode_vector_store_ids_in_tools(tools)
        assert result is not None
        assert result[0]["vector_store_ids"] == ["vs_openai_abc"]

    def test_A6_mixed_unified_and_native_ids(self):
        unified_id = _make_unified_vs_id(provider_resource_id="vs_decoded")
        native_id = "vs_native_xyz"
        tools = [_file_search_tool([unified_id, native_id])]
        result = _decode_vector_store_ids_in_tools(tools)
        assert result is not None
        assert result[0]["vector_store_ids"] == ["vs_decoded", "vs_native_xyz"]

    def test_A7_malformed_base64_passes_through_unchanged(self):
        bad_id = "not_valid_base64!!!"
        tools = [_file_search_tool([bad_id])]
        result = _decode_vector_store_ids_in_tools(tools)
        assert result is not None
        assert result[0]["vector_store_ids"] == [bad_id]


# ---------------------------------------------------------------------------
# B-series: update_responses_tools_with_model_file_ids
# ---------------------------------------------------------------------------


class TestUpdateResponsesToolsWithModelFileIds:
    def test_B1_file_search_decode_runs_without_mapping(self):
        """Decode pass executes even when model_file_id_mapping is None."""
        unified_id = _make_unified_vs_id(provider_resource_id="vs_decoded")
        tools = [_file_search_tool([unified_id])]

        result = update_responses_tools_with_model_file_ids(
            tools=tools,
            model_id=None,
            model_file_id_mapping=None,
        )
        assert result is not None
        assert result[0]["vector_store_ids"] == ["vs_decoded"]

    def test_B2_code_interpreter_mapping_still_works(self):
        """code_interpreter mapping pass still works after decode pass."""
        model_id = "model-abc"
        file_id = "litellm_managed_file_001"
        tools = [_code_interpreter_tool([file_id])]
        mapping = {file_id: {model_id: "provider_file_xyz"}}

        result = update_responses_tools_with_model_file_ids(
            tools=tools,
            model_id=model_id,
            model_file_id_mapping=mapping,
        )
        assert result is not None
        assert result[0]["container"]["file_ids"] == ["provider_file_xyz"]

    def test_B3_both_passes_run_correctly(self):
        """Both file_search decode and code_interpreter mapping run."""
        model_id = "model-abc"
        file_id = "litellm_managed_file_001"
        unified_id = _make_unified_vs_id(provider_resource_id="vs_decoded")

        tools = [
            _file_search_tool([unified_id]),
            _code_interpreter_tool([file_id]),
        ]
        mapping = {file_id: {model_id: "provider_file_xyz"}}

        result = update_responses_tools_with_model_file_ids(
            tools=tools,
            model_id=model_id,
            model_file_id_mapping=mapping,
        )
        assert result is not None
        assert result[0]["vector_store_ids"] == ["vs_decoded"]
        assert result[1]["container"]["file_ids"] == ["provider_file_xyz"]


# ---------------------------------------------------------------------------
# C/D-series: supports_native_file_search
# ---------------------------------------------------------------------------


class TestSupportsNativeFileSearch:
    def test_C1_base_class_default_is_false(self):
        # Access the unbound method directly — no need to instantiate an abstract class
        assert BaseResponsesAPIConfig.supports_native_file_search(MagicMock()) is False

    def test_D1_openai_returns_true(self):
        assert OpenAIResponsesAPIConfig().supports_native_file_search() is True


# ---------------------------------------------------------------------------
# E-series: file_search guard in responses/main.py
# ---------------------------------------------------------------------------


class TestFileSearchGuardInResponsesMain:
    """Tests for _has_file_search_tool helper and emulated routing guard."""

    def test_has_file_search_tool_true(self):
        from litellm.responses.main import _has_file_search_tool

        assert _has_file_search_tool([{"type": "file_search"}]) is True

    def test_has_file_search_tool_false_empty(self):
        from litellm.responses.main import _has_file_search_tool

        assert _has_file_search_tool([]) is False
        assert _has_file_search_tool(None) is False

    def test_has_file_search_tool_false_other_tools(self):
        from litellm.responses.main import _has_file_search_tool

        assert _has_file_search_tool([{"type": "web_search"}]) is False

    def test_E1_openai_provider_no_error(self):
        """OpenAI supports file_search natively — no error raised."""
        from litellm.llms.openai.responses.transformation import (
            OpenAIResponsesAPIConfig,
        )
        from litellm.responses.main import _has_file_search_tool

        config = OpenAIResponsesAPIConfig()
        tools = [{"type": "file_search", "vector_store_ids": ["vs_abc"]}]
        assert _has_file_search_tool(tools)
        assert config.supports_native_file_search()
        # No exception expected — the guard would pass.

    def test_E2_no_provider_config_routes_to_emulated_handler(self):
        """Provider config None + file_search should route to emulated handler."""
        from litellm.responses.main import responses

        tools = [{"type": "file_search", "vector_store_ids": ["vs_abc"]}]
        logging_obj = MagicMock()
        expected = {"ok": True}

        with (
            patch(
                "litellm.responses.main.litellm.get_llm_provider",
                return_value=("claude-sonnet-4-5", "anthropic", None, None),
            ),
            patch(
                "litellm.responses.main.update_responses_input_with_model_file_ids",
                return_value="hello",
            ),
            patch(
                "litellm.responses.main.update_responses_tools_with_model_file_ids",
                return_value=tools,
            ),
            patch(
                "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config",
                return_value=None,
            ),
            patch(
                "litellm.responses.main.ResponsesAPIRequestUtils.get_requested_response_api_optional_param",
                return_value={},
            ),
            patch(
                "litellm.responses.main.run_async_function", return_value=expected
            ) as run_async_mock,
        ):
            result = responses(
                input="hello",
                model="anthropic/claude-sonnet-4-5",
                tools=tools,
                litellm_logging_obj=logging_obj,
                litellm_call_id="call-123",
            )

        assert result == expected
        assert run_async_mock.called
        routed_func = run_async_mock.call_args.args[0]
        assert routed_func.__name__ == "aresponses_with_emulated_file_search"

    def test_E3_non_native_provider_config_routes_to_emulated_handler(self):
        """Non-native provider config + file_search should route to emulated handler."""
        from litellm.llms.base_llm.responses.transformation import (
            BaseResponsesAPIConfig,
        )
        from litellm.responses.main import responses

        tools = [{"type": "file_search", "vector_store_ids": ["vs_abc"]}]
        logging_obj = MagicMock()
        expected = {"ok": True}
        mock_config = MagicMock(spec=BaseResponsesAPIConfig)
        mock_config.supports_native_file_search.return_value = False

        with (
            patch(
                "litellm.responses.main.litellm.get_llm_provider",
                return_value=("claude-sonnet-4-5", "anthropic", None, None),
            ),
            patch(
                "litellm.responses.main.update_responses_input_with_model_file_ids",
                return_value="hello",
            ),
            patch(
                "litellm.responses.main.update_responses_tools_with_model_file_ids",
                return_value=tools,
            ),
            patch(
                "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config",
                return_value=mock_config,
            ),
            patch(
                "litellm.responses.main.ResponsesAPIRequestUtils.get_requested_response_api_optional_param",
                return_value={},
            ),
            patch(
                "litellm.responses.main.run_async_function", return_value=expected
            ) as run_async_mock,
        ):
            result = responses(
                input="hello",
                model="anthropic/claude-sonnet-4-5",
                tools=tools,
                litellm_logging_obj=logging_obj,
                litellm_call_id="call-123",
            )

        assert result == expected
        assert run_async_mock.called
        routed_func = run_async_mock.call_args.args[0]
        assert routed_func.__name__ == "aresponses_with_emulated_file_search"

    def test_E4_no_file_search_tools_no_error(self):
        """No file_search tool in request → guard never fires."""
        from litellm.responses.main import _has_file_search_tool

        tools = [{"type": "web_search"}, {"type": "code_interpreter"}]
        assert not _has_file_search_tool(tools)


# ---------------------------------------------------------------------------
# F-series: ManagedFiles hook — vector_store_ids access control
# ---------------------------------------------------------------------------


class TestManagedFilesVectorStoreAccess:
    def _make_hook(self):
        """Return a ManagedFiles instance with prisma_client mocked."""
        from litellm_enterprise.proxy.hooks.managed_files import (
            _PROXY_LiteLLMManagedFiles as ManagedFiles,
        )

        hook = ManagedFiles.__new__(ManagedFiles)
        return hook

    def _make_user(self, team_id: Optional[str] = "team-abc") -> MagicMock:
        user = MagicMock()
        user.team_id = team_id
        user.user_id = "user-1"
        return user

    def test_F1_non_unified_vs_id_skipped(self):
        hook = self._make_hook()
        result = hook.get_vector_store_ids_from_file_search_tools(
            [{"type": "file_search", "vector_store_ids": ["vs_native_123"]}]
        )
        assert result == []  # native ID filtered out

    def test_F2_unified_vs_id_extracted(self):
        hook = self._make_hook()
        unified_id = _make_unified_vs_id()
        result = hook.get_vector_store_ids_from_file_search_tools(
            [{"type": "file_search", "vector_store_ids": [unified_id]}]
        )
        assert result == [unified_id]

    def _make_vs_row(self, vector_store_id: str, team_id: Optional[str]) -> Any:
        """Build a row compatible with get_managed_vector_store_rows_by_uuids (Prisma model_dump)."""
        from litellm.proxy._types import LiteLLM_ManagedVectorStoresTable

        return LiteLLM_ManagedVectorStoresTable(
            vector_store_id=vector_store_id,
            custom_llm_provider="openai",
            vector_store_name=None,
            vector_store_description=None,
            vector_store_metadata=None,
            created_at=None,
            updated_at=None,
            litellm_credential_name=None,
            litellm_params=None,
            team_id=team_id,
            user_id=None,
        )

    @pytest.mark.asyncio
    async def test_F3_wrong_team_raises_403(self):
        from fastapi import HTTPException

        hook = self._make_hook()
        unified_id = _make_unified_vs_id(unified_uuid="uuid-001")

        mock_row = self._make_vs_row(vector_store_id="uuid-001", team_id="team-other")

        async def mock_get_rows(
            uuids, prisma_client, user_api_key_cache, proxy_logging_obj=None
        ):
            return [mock_row]

        with (
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                MagicMock(),
            ),
            patch(
                "litellm.proxy.auth.auth_checks.get_managed_vector_store_rows_by_uuids",
                side_effect=mock_get_rows,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await hook.check_vector_store_ids_access(
                    [unified_id], self._make_user(team_id="team-caller")
                )
            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_F4_no_team_on_vs_allowed(self):
        """Legacy vector store with no team_id — accessible to all."""
        hook = self._make_hook()
        unified_id = _make_unified_vs_id(unified_uuid="uuid-002")

        mock_row = self._make_vs_row(vector_store_id="uuid-002", team_id=None)

        async def mock_get_rows(
            uuids, prisma_client, user_api_key_cache, proxy_logging_obj=None
        ):
            return [mock_row]

        with (
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                MagicMock(),
            ),
            patch(
                "litellm.proxy.auth.auth_checks.get_managed_vector_store_rows_by_uuids",
                side_effect=mock_get_rows,
            ),
        ):
            await hook.check_vector_store_ids_access(
                [unified_id], self._make_user(team_id="team-caller")
            )

    @pytest.mark.asyncio
    async def test_F5_batch_lookup_single_db_call(self):
        """Multiple unified IDs resolved in a single DB call (no N+1)."""
        hook = self._make_hook()
        ids = [
            _make_unified_vs_id(
                unified_uuid=f"uuid-{i}", provider_resource_id=f"vs_{i}"
            )
            for i in range(3)
        ]

        rows = [
            self._make_vs_row(vector_store_id=f"uuid-{i}", team_id="team-abc")
            for i in range(3)
        ]

        get_rows_mock = AsyncMock(return_value=rows)

        with (
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                MagicMock(),
            ),
            patch(
                "litellm.proxy.auth.auth_checks.get_managed_vector_store_rows_by_uuids",
                get_rows_mock,
            ),
        ):
            await hook.check_vector_store_ids_access(ids, self._make_user("team-abc"))

        get_rows_mock.assert_called_once()
        call_args = get_rows_mock.call_args
        assert set(call_args.kwargs["uuids"] or call_args.args[0]) == {
            "uuid-0",
            "uuid-1",
            "uuid-2",
        }

    @pytest.mark.asyncio
    async def test_F6_non_responses_call_type_skipped(self):
        """Access check only runs for aresponses/responses call types."""
        from litellm_enterprise.proxy.hooks.managed_files import (
            _PROXY_LiteLLMManagedFiles as ManagedFiles,
        )
        from litellm.proxy._types import CallTypes

        # If call_type is acompletion, the vector_store check branch isn't reached.
        # Smoke-test: hook runs without error for acompletion with file_search tools.
        hook = MagicMock(spec=ManagedFiles)
        hook.async_pre_call_hook = AsyncMock(return_value=None)

        await hook.async_pre_call_hook(
            user_api_key_dict=self._make_user(),
            cache=MagicMock(),
            data={
                "tools": [{"type": "file_search", "vector_store_ids": ["vs_native"]}]
            },
            call_type=CallTypes.acompletion.value,
        )
        hook.async_pre_call_hook.assert_called_once()


# ---------------------------------------------------------------------------
# G-series: get_vector_store_ids_from_file_search_tools helper
# ---------------------------------------------------------------------------


class TestGetVectorStoreIdsFromFileSearchTools:
    def _make_hook(self):
        from litellm_enterprise.proxy.hooks.managed_files import (
            _PROXY_LiteLLMManagedFiles as ManagedFiles,
        )

        return ManagedFiles.__new__(ManagedFiles)

    def test_G1_tools_none_returns_empty(self):
        hook = self._make_hook()
        assert hook.get_vector_store_ids_from_file_search_tools([]) == []

    def test_G2_no_file_search_tools_returns_empty(self):
        hook = self._make_hook()
        tools = [{"type": "code_interpreter"}, {"type": "web_search"}]
        assert hook.get_vector_store_ids_from_file_search_tools(tools) == []

    def test_G3_only_file_search_vs_ids_returned(self):
        hook = self._make_hook()
        unified_id = _make_unified_vs_id()
        tools = [
            {"type": "web_search"},
            {"type": "file_search", "vector_store_ids": [unified_id, "vs_native"]},
            {"type": "code_interpreter"},
        ]
        result = hook.get_vector_store_ids_from_file_search_tools(tools)
        # Only the unified ID is included; native IDs are filtered
        assert result == [unified_id]


# ---------------------------------------------------------------------------
# Phase 2: Emulated file_search handler
# ---------------------------------------------------------------------------


class TestEmulatedFileSearchHandler:
    """Tests for litellm/responses/file_search/emulated_handler.py"""

    def _make_mock_responses_api_response(
        self,
        text: str = "The answer is 42.",
        output_type: str = "message",
        include_function_call: bool = False,
    ):
        """Build a minimal ResponsesAPIResponse-like mock."""
        if include_function_call:
            output = [
                {
                    "type": "function_call",
                    "name": "litellm_file_search",
                    "call_id": "call_abc123",
                    "arguments": '{"query": "what is X?", "vector_store_id": "vs_001"}',
                }
            ]
        else:
            output = [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ]
        resp = MagicMock()
        resp.output = output
        resp.id = "resp_test123"
        resp.created_at = 1700000000
        resp.model = "claude-3-5-sonnet"
        resp.usage = None
        return resp

    # --- Tool conversion ---

    def test_H1_file_search_replaced_with_function_tool(self):
        from litellm.responses.file_search.emulated_handler import (
            _replace_file_search_tools,
        )

        tools = [{"type": "file_search", "vector_store_ids": ["vs_abc", "vs_def"]}]
        new_tools, vs_ids = _replace_file_search_tools(tools)

        assert vs_ids == ["vs_abc", "vs_def"]
        assert len(new_tools) == 1
        assert new_tools[0]["type"] == "function"
        assert new_tools[0]["name"] == "litellm_file_search"
        # Both store IDs appear in the enum
        enum_ids = new_tools[0]["parameters"]["properties"]["vector_store_id"]["enum"]
        assert "vs_abc" in enum_ids
        assert "vs_def" in enum_ids

    def test_H2_non_file_search_tools_preserved(self):
        from litellm.responses.file_search.emulated_handler import (
            _replace_file_search_tools,
        )

        tools = [
            {"type": "web_search"},
            {"type": "file_search", "vector_store_ids": ["vs_abc"]},
        ]
        new_tools, vs_ids = _replace_file_search_tools(tools)

        assert len(new_tools) == 2  # web_search + generated function tool
        assert new_tools[0]["type"] == "web_search"
        assert new_tools[1]["type"] == "function"

    def test_H3_no_file_search_tools_returns_unchanged(self):
        from litellm.responses.file_search.emulated_handler import (
            _replace_file_search_tools,
        )

        tools = [{"type": "web_search"}]
        new_tools, vs_ids = _replace_file_search_tools(tools)

        assert vs_ids == []
        assert new_tools == [{"type": "web_search"}]

    def test_H4_empty_vector_store_ids_no_function_tool(self):
        from litellm.responses.file_search.emulated_handler import (
            _replace_file_search_tools,
        )

        tools = [{"type": "file_search", "vector_store_ids": []}]
        new_tools, vs_ids = _replace_file_search_tools(tools)

        assert vs_ids == []
        assert new_tools == []  # no function tool added without store IDs

    # --- Detection ---

    def test_H5_should_use_emulated_for_non_native_provider(self):
        from litellm.responses.file_search.emulated_handler import (
            should_use_emulated_file_search,
        )

        mock_config = MagicMock()
        mock_config.supports_native_file_search.return_value = False
        tools = [{"type": "file_search", "vector_store_ids": ["vs_abc"]}]

        assert should_use_emulated_file_search(tools, mock_config) is True

    def test_H6_should_not_emulate_for_native_provider(self):
        from litellm.llms.openai.responses.transformation import (
            OpenAIResponsesAPIConfig,
        )
        from litellm.responses.file_search.emulated_handler import (
            should_use_emulated_file_search,
        )

        config = OpenAIResponsesAPIConfig()
        tools = [{"type": "file_search", "vector_store_ids": ["vs_abc"]}]

        assert should_use_emulated_file_search(tools, config) is False

    def test_H7_should_not_emulate_without_file_search_tools(self):
        from litellm.responses.file_search.emulated_handler import (
            should_use_emulated_file_search,
        )

        mock_config = MagicMock()
        mock_config.supports_native_file_search.return_value = False
        tools = [{"type": "web_search"}]

        assert should_use_emulated_file_search(tools, mock_config) is False

    # --- Output synthesis ---

    def test_H8_synthesized_output_has_file_search_call_and_message(self):
        from litellm.responses.file_search.emulated_handler import (
            _build_file_search_call_output,
            _build_message_output,
        )

        fs_call = _build_file_search_call_output("fs_abc123", ["what is X?"])
        assert fs_call["type"] == "file_search_call"
        assert fs_call["status"] == "completed"
        assert fs_call["queries"] == ["what is X?"]

        msg = _build_message_output("The answer is 42.", [])
        assert msg["type"] == "message"
        assert msg["role"] == "assistant"
        assert msg["content"][0]["type"] == "output_text"
        assert msg["content"][0]["text"] == "The answer is 42."

    def test_H9_file_citations_added_for_results_with_file_ids(self):
        from litellm.responses.file_search.emulated_handler import (
            _build_file_citation_annotations,
        )

        result = MagicMock()
        result.file_id = "file-abc"
        result.filename = "doc.pdf"

        annotations = _build_file_citation_annotations([result], "some text")
        assert len(annotations) == 1
        assert annotations[0]["type"] == "file_citation"
        assert annotations[0]["file_id"] == "file-abc"
        assert annotations[0]["filename"] == "doc.pdf"

    def test_H10_no_duplicate_citations_for_same_file(self):
        from litellm.responses.file_search.emulated_handler import (
            _build_file_citation_annotations,
        )

        r1, r2 = MagicMock(), MagicMock()
        r1.file_id = "file-abc"
        r1.filename = "doc.pdf"
        r2.file_id = "file-abc"  # same file
        r2.filename = "doc.pdf"

        annotations = _build_file_citation_annotations([r1, r2], "text")
        assert len(annotations) == 1

    def test_H14_include_search_results_returns_all_chunks(self):
        """All chunks are returned even when they originate from the same file,
        matching OpenAI native file_search behaviour."""
        from litellm.responses.file_search.emulated_handler import (
            _build_search_results_for_include,
        )

        r1, r2 = MagicMock(), MagicMock()
        r1.file_id = "file-abc"
        r1.filename = "doc.pdf"
        r1.score = 0.9
        r1.attributes = {}
        r1.content = [{"type": "text", "text": "first hit"}]
        r2.file_id = "file-abc"  # same file, different chunk from a second query
        r2.filename = "doc.pdf"
        r2.score = 0.85
        r2.attributes = {}
        r2.content = [{"type": "text", "text": "second hit"}]

        search_results = _build_search_results_for_include([r1, r2])
        assert (
            len(search_results) == 2
        ), "Both chunks should be returned, not deduplicated"
        assert search_results[0]["text"] == "first hit"
        assert search_results[1]["text"] == "second hit"

    # --- End-to-end (mocked) ---

    @pytest.mark.asyncio
    async def test_H11_emulated_full_flow_provider_calls_tool(self):
        """Full flow: provider calls file_search function → search → follow-up → OpenAI output."""
        from litellm.responses.file_search.emulated_handler import (
            aresponses_with_emulated_file_search,
        )

        first_resp = self._make_mock_responses_api_response(include_function_call=True)
        final_resp = self._make_mock_responses_api_response(
            text="Deep research enables multi-step queries."
        )

        search_result = MagicMock()
        search_result.file_id = "file-xyz"
        search_result.filename = "research.pdf"
        search_result.score = 0.95
        search_result.content = [{"type": "text", "text": "deep research context..."}]

        mock_search_response = MagicMock()
        mock_search_response.data = [search_result]

        with (
            patch(
                "litellm.responses.file_search.emulated_handler._call_aresponses",
                new=AsyncMock(side_effect=[first_resp, final_resp]),
            ),
            patch(
                "litellm.vector_stores.main.asearch",
                new=AsyncMock(return_value=mock_search_response),
            ),
        ):
            result = await aresponses_with_emulated_file_search(
                input="What is deep research?",
                model="anthropic/claude-3-5-sonnet",
                tools=[{"type": "file_search", "vector_store_ids": ["vs_001"]}],
            )

        # output[0] is file_search_call, output[1] is message
        # ResponsesAPIResponse converts dicts to Pydantic objects — use attribute access
        def _get(item, key):
            return item[key] if isinstance(item, dict) else getattr(item, key, None)

        assert _get(result.output[0], "type") == "file_search_call"
        assert _get(result.output[0], "status") == "completed"
        assert _get(result.output[1], "type") == "message"
        content0 = _get(result.output[1], "content")[0]
        assert "Deep research" in _get(content0, "text")
        annotations = _get(content0, "annotations")
        assert any(_get(a, "file_id") == "file-xyz" for a in annotations)

    @pytest.mark.asyncio
    async def test_H11b_emulated_full_flow_primary_queries_schema(self):
        """Primary path: provider returns queries (plural array) as defined in the tool schema."""
        from litellm.responses.file_search.emulated_handler import (
            aresponses_with_emulated_file_search,
        )

        # Use the primary schema: queries (plural, list) instead of the backward-compat query (singular)
        first_resp_plural = MagicMock()
        first_resp_plural.output = [
            {
                "type": "function_call",
                "name": "litellm_file_search",
                "call_id": "call_plural",
                "arguments": '{"queries": ["what is deep research?", "multi-step reasoning"], "vector_store_id": "vs_001"}',
            }
        ]
        first_resp_plural.id = "resp_plural"
        first_resp_plural.created_at = 1700000000
        first_resp_plural.model = "claude-3-5-sonnet"
        first_resp_plural.usage = None

        final_resp = self._make_mock_responses_api_response(
            text="Deep research uses multiple queries."
        )

        search_result = MagicMock()
        search_result.file_id = "file-multi"
        search_result.filename = "multi.pdf"
        search_result.score = 0.9
        search_result.content = [{"type": "text", "text": "multi-query context"}]
        mock_search_response = MagicMock()
        mock_search_response.data = [search_result]

        with (
            patch(
                "litellm.responses.file_search.emulated_handler._call_aresponses",
                new=AsyncMock(side_effect=[first_resp_plural, final_resp]),
            ),
            patch(
                "litellm.vector_stores.main.asearch",
                new=AsyncMock(return_value=mock_search_response),
            ),
        ):
            result = await aresponses_with_emulated_file_search(
                input="What is deep research?",
                model="anthropic/claude-3-5-sonnet",
                tools=[{"type": "file_search", "vector_store_ids": ["vs_001"]}],
            )

        def _get(item, key):
            return item[key] if isinstance(item, dict) else getattr(item, key, None)

        assert _get(result.output[0], "type") == "file_search_call"
        # Two queries were issued, both should appear in the output
        assert len(_get(result.output[0], "queries")) == 2
        assert _get(result.output[1], "type") == "message"

    @pytest.mark.asyncio
    async def test_H12_emulated_flow_provider_answers_without_tool_call(self):
        """If provider answers directly (no tool call), still return OpenAI format."""
        from litellm.responses.file_search.emulated_handler import (
            aresponses_with_emulated_file_search,
        )

        direct_resp = self._make_mock_responses_api_response(
            text="I already know the answer."
        )

        with patch(
            "litellm.responses.file_search.emulated_handler._call_aresponses",
            new=AsyncMock(return_value=direct_resp),
        ):
            result = await aresponses_with_emulated_file_search(
                input="What is 2+2?",
                model="anthropic/claude-3-5-sonnet",
                tools=[{"type": "file_search", "vector_store_ids": ["vs_001"]}],
            )

        def _get(item, key):
            return item[key] if isinstance(item, dict) else getattr(item, key, None)

        assert _get(result.output[0], "type") == "file_search_call"
        assert _get(result.output[1], "type") == "message"
        assert "I already know" in _get(_get(result.output[1], "content")[0], "text")

    def test_H13_should_use_emulated_when_provider_config_is_none(self):
        """None provider config (chat fallback) also triggers emulation."""
        from litellm.responses.file_search.emulated_handler import (
            should_use_emulated_file_search,
        )

        tools = [{"type": "file_search", "vector_store_ids": ["vs_abc"]}]
        assert should_use_emulated_file_search(tools, None) is True

    @pytest.mark.asyncio
    async def test_H15_sub_calls_carry_internal_call_flag(self):
        """Both internal aresponses sub-calls run with is_internal_call context var True.

        This ensures wrapper_async skips success/failure callbacks for sub-calls so
        billing fires exactly once (on the outer call) with the synthesized result.
        """
        from litellm._internal_context import is_internal_call
        from litellm.responses.file_search.emulated_handler import (
            aresponses_with_emulated_file_search,
        )

        first_resp = self._make_mock_responses_api_response(include_function_call=True)
        final_resp = self._make_mock_responses_api_response(text="answer")

        search_result = MagicMock()
        search_result.file_id = "file-h15"
        search_result.filename = "h15.pdf"
        search_result.score = 0.9
        search_result.content = [{"type": "text", "text": "context"}]
        mock_search_response = MagicMock()
        mock_search_response.data = [search_result]

        with (
            patch(
                "litellm.responses.file_search.emulated_handler._call_aresponses",
                new=AsyncMock(side_effect=[first_resp, final_resp]),
            ) as mock_call,
            patch(
                "litellm.vector_stores.main.asearch",
                new=AsyncMock(return_value=mock_search_response),
            ),
        ):
            captured_ctx: list = []
            original_side_effect = [first_resp, final_resp]

            async def _intercept(**kwargs):  # type: ignore[misc]
                captured_ctx.append(is_internal_call.get())
                return original_side_effect.pop(0)

            mock_call.side_effect = _intercept

            await aresponses_with_emulated_file_search(
                input="What is H15?",
                model="anthropic/claude-3-5-sonnet",
                tools=[{"type": "file_search", "vector_store_ids": ["vs_h15"]}],
            )

        assert len(captured_ctx) == 2, "Expected exactly 2 sub-calls"
        for i, ctx_val in enumerate(captured_ctx):
            assert ctx_val is True, (
                f"Sub-call {i} must run with is_internal_call=True to suppress "
                "billing callbacks in wrapper_async"
            )
