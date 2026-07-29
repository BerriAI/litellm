"""
Tests for Anthropic-native ``web_search_tool_result`` block emission.

Covers the path that lets Claude Desktop / Anthropic SDK clients render
citations when their request used a native ``web_search_*`` tool against a
provider (e.g. Bedrock) that can't run web search natively.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.websearch_interception.handler import (
    WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY,
    WEBSEARCH_NATIVE_BLOCKS_METADATA_KEY,
    WebSearchInterceptionLogger,
)
from litellm.integrations.websearch_interception.tools import (
    is_anthropic_native_web_search_tool,
    is_web_search_tool,
)
from litellm.integrations.websearch_interception.transformation import (
    WebSearchTransformation,
)
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.types.integrations.custom_logger import (
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
)


def _make_search_response() -> SearchResponse:
    return SearchResponse(
        results=[
            SearchResult(
                title="LiteLLM Docs",
                url="https://docs.litellm.ai/",
                snippet="Unified interface for LLMs.",
                date="2025-01-15",
            ),
            SearchResult(
                title="Bedrock Pricing",
                url="https://aws.amazon.com/bedrock/pricing/",
                snippet="Pay-per-use pricing model.",
                date=None,
            ),
        ]
    )


class TestIsAnthropicNativeWebSearchTool:
    """The detector must match native tools without catching look-alikes."""

    def test_matches_web_search_20250305(self):
        assert is_anthropic_native_web_search_tool({"type": "web_search_20250305", "name": "web_search", "max_uses": 5})

    def test_matches_future_dated_variant(self):
        assert is_anthropic_native_web_search_tool({"type": "web_search_20260101", "name": "web_search"})

    def test_rejects_litellm_standard(self):
        assert not is_anthropic_native_web_search_tool({"name": "litellm_web_search", "input_schema": {}})

    def test_rejects_openai_function_shape(self):
        assert not is_anthropic_native_web_search_tool({"type": "function", "function": {"name": "litellm_web_search"}})

    def test_rejects_claude_desktop_builtin(self):
        # Claude Desktop's builtin client-side ``WebSearch`` tool must not be
        # misidentified — that's the collision PR #25242 introduced.
        assert not is_anthropic_native_web_search_tool({"name": "WebSearch"})

    def test_rejects_unrelated_tool(self):
        assert not is_anthropic_native_web_search_tool({"type": "function", "function": {"name": "calculator"}})

    def test_handles_missing_type(self):
        assert not is_anthropic_native_web_search_tool({"name": "web_search"})


class TestLegacyWebSearchNameGate:
    """The bare ``WebSearch`` name is a legacy interception marker. Real
    client-side ``WebSearch`` tools (Cowork, Claude Desktop) carry an
    ``input_schema`` and must pass through untouched — otherwise the proxy
    hijacks them server-side and the client's own tool handler never fires,
    which means the separate ``web_search_20250305`` sub-request (where
    citations actually flow) is never made."""

    def test_bare_legacy_name_still_matched(self):
        # Caller deliberately uses the bare-name interception marker —
        # back-compat for anyone relying on the old shape.
        assert is_web_search_tool({"name": "WebSearch"})

    def test_real_client_tool_passes_through(self):
        # Cowork's client-side WebSearch tool ships with input_schema.
        cowork_tool = {
            "name": "WebSearch",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
        assert not is_web_search_tool(cowork_tool)

    def test_real_client_tool_with_description_passes_through(self):
        # description-only client tools (no schema) are not valid Anthropic
        # tools; only the schema-bearing shape is the disambiguator. This
        # case stays matched on the assumption it's a legacy marker.
        assert is_web_search_tool({"name": "WebSearch", "description": "search"})


class TestBuildWebSearchToolResultBlock:
    """The block-builder must produce the Anthropic-native shape exactly."""

    def test_shape_with_results(self):
        block = WebSearchTransformation.build_web_search_tool_result_block(
            tool_use_id="toolu_abc",
            search_response=_make_search_response(),
        )
        assert block["type"] == "web_search_tool_result"
        assert block["tool_use_id"] == "toolu_abc"
        assert len(block["content"]) == 2
        first = block["content"][0]
        assert first["type"] == "web_search_result"
        assert first["url"] == "https://docs.litellm.ai/"
        assert first["title"] == "LiteLLM Docs"
        assert first["page_age"] == "2025-01-15"
        assert first["encrypted_content"] == ""

    def test_handles_none_search_response(self):
        block = WebSearchTransformation.build_web_search_tool_result_block(
            tool_use_id="toolu_abc",
            search_response=None,
        )
        assert block["type"] == "web_search_tool_result"
        assert block["tool_use_id"] == "toolu_abc"
        assert block["content"] == []

    def test_handles_empty_results(self):
        block = WebSearchTransformation.build_web_search_tool_result_block(
            tool_use_id="toolu_xyz",
            search_response=SearchResponse(results=[]),
        )
        assert block["content"] == []


class TestPreRequestHookFlagsNativeTools:
    """The pre-request hook must mark the request when a native tool is used."""

    @pytest.mark.asyncio
    async def test_native_tool_sets_flag(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        kwargs = {
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            "litellm_params": {"custom_llm_provider": "bedrock"},
        }
        out = await logger.async_pre_request_hook(model="bedrock/claude", messages=[], kwargs=kwargs)
        assert out is not None
        assert out.get(WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY) is True

    @pytest.mark.asyncio
    async def test_litellm_standard_tool_does_not_set_flag(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        kwargs = {
            "tools": [{"name": "litellm_web_search", "input_schema": {}}],
            "litellm_params": {"custom_llm_provider": "bedrock"},
        }
        out = await logger.async_pre_request_hook(model="bedrock/claude", messages=[], kwargs=kwargs)
        assert out is not None
        assert WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY not in out


class TestBuildPlanAttachesBlocks:
    """async_build_agentic_loop_plan must put pre-built blocks on metadata."""

    @pytest.mark.asyncio
    async def test_metadata_carries_blocks_when_flag_set(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        tool_calls = [
            {
                "id": "toolu_one",
                "type": "tool_use",
                "name": "litellm_web_search",
                "input": {"query": "what is litellm"},
            }
        ]
        patch_obj = AgenticLoopRequestPatch(
            model="bedrock/claude",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
        )
        structured = [_make_search_response()]

        with patch.object(
            logger,
            "_build_anthropic_request_patch",
            new=AsyncMock(return_value=(patch_obj, structured)),
        ):
            plan = await logger.async_build_agentic_loop_plan(
                tools={"tool_calls": tool_calls, "thinking_blocks": []},
                model="bedrock/claude",
                messages=[],
                response=MagicMock(),
                anthropic_messages_provider_config=None,
                anthropic_messages_optional_request_params={},
                logging_obj=MagicMock(model_call_details={}),
                stream=False,
                kwargs={WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY: True},
            )

        blocks = plan.metadata.get(WEBSEARCH_NATIVE_BLOCKS_METADATA_KEY)
        assert isinstance(blocks, list)
        assert len(blocks) == 2
        server_use, tool_result = blocks
        assert server_use["type"] == "server_tool_use"
        assert server_use["id"].startswith("srvtoolu_")
        assert server_use["input"] == {"query": "what is litellm"}
        assert tool_result["type"] == "web_search_tool_result"
        assert tool_result["tool_use_id"] == server_use["id"]
        assert tool_result["tool_use_id"] != "toolu_one"
        assert tool_result["content"][0]["url"] == "https://docs.litellm.ai/"

    @pytest.mark.asyncio
    async def test_metadata_does_not_carry_blocks_when_flag_absent(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        tool_calls = [
            {
                "id": "toolu_one",
                "type": "tool_use",
                "name": "litellm_web_search",
                "input": {"query": "what is litellm"},
            }
        ]
        patch_obj = AgenticLoopRequestPatch(
            model="bedrock/claude",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
        )

        with patch.object(
            logger,
            "_build_anthropic_request_patch",
            new=AsyncMock(return_value=(patch_obj, [_make_search_response()])),
        ):
            plan = await logger.async_build_agentic_loop_plan(
                tools={"tool_calls": tool_calls, "thinking_blocks": []},
                model="bedrock/claude",
                messages=[],
                response=MagicMock(),
                anthropic_messages_provider_config=None,
                anthropic_messages_optional_request_params={},
                logging_obj=MagicMock(model_call_details={}),
                stream=False,
                kwargs={},
            )

        assert WEBSEARCH_NATIVE_BLOCKS_METADATA_KEY not in plan.metadata


class TestPostHookInjectsBlocks:
    """The post-hook must prepend blocks; absent metadata is a no-op."""

    @pytest.mark.asyncio
    async def test_injects_when_metadata_present(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        block = WebSearchTransformation.build_web_search_tool_result_block(
            tool_use_id="toolu_abc",
            search_response=_make_search_response(),
        )
        plan = AgenticLoopPlan(
            run_agentic_loop=True,
            metadata={WEBSEARCH_NATIVE_BLOCKS_METADATA_KEY: [block]},
        )
        response = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Based on the search..."}],
            "stop_reason": "end_turn",
        }

        out = await logger.async_post_agentic_loop_response_hook(response=response, plan=plan, kwargs={})

        # Native block must be first so the client can pair it with the
        # tool_use before reading the assistant text.
        assert out["content"][0]["type"] == "web_search_tool_result"
        assert out["content"][0]["tool_use_id"] == "toolu_abc"
        assert out["content"][1]["type"] == "text"

    @pytest.mark.asyncio
    async def test_noop_when_metadata_absent(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        plan = AgenticLoopPlan(run_agentic_loop=True, metadata={})
        response = {
            "id": "msg_1",
            "content": [{"type": "text", "text": "answer"}],
        }
        out = await logger.async_post_agentic_loop_response_hook(response=response, plan=plan, kwargs={})
        assert out == response

    @pytest.mark.asyncio
    async def test_handles_object_style_response(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        block = WebSearchTransformation.build_web_search_tool_result_block(
            tool_use_id="toolu_obj",
            search_response=_make_search_response(),
        )
        plan = AgenticLoopPlan(
            run_agentic_loop=True,
            metadata={WEBSEARCH_NATIVE_BLOCKS_METADATA_KEY: [block]},
        )

        class _Resp:
            def __init__(self):
                self.content = [{"type": "text", "text": "ok"}]

        resp = _Resp()
        out = await logger.async_post_agentic_loop_response_hook(response=resp, plan=plan, kwargs={})
        assert out.content[0]["type"] == "web_search_tool_result"
        assert out.content[1]["type"] == "text"


class TestShortCircuitEmitsNativeBlocks:
    """Standalone /v1/messages sub-requests (Cowork's separate search call)
    hit ``try_short_circuit_search``, which builds a synthetic response and
    never enters the agentic loop. The native-block emission must happen
    here too, otherwise the citations panel stays empty."""

    @pytest.mark.asyncio
    async def test_native_tool_short_circuit_emits_blocks(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger,
            "_execute_search",
            new=AsyncMock(return_value=("Title: x\nURL: y", _make_search_response())),
        ):
            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[{"role": "user", "content": "search query"}],
                tools=[
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ],
                custom_llm_provider="github_copilot",
            )

        assert result is not None
        block_types = [b["type"] for b in result["content"]]
        # Order matters: native clients expect tool_use before tool_result.
        assert block_types == ["server_tool_use", "web_search_tool_result", "text"]
        server_use, tool_result, _ = result["content"]
        assert server_use["name"] == "web_search"
        assert server_use["input"] == {"query": "search query"}
        # tool_use_id must match between the server_tool_use and the
        # web_search_tool_result block so the client can pair them.
        assert server_use["id"].startswith("srvtoolu_")
        assert tool_result["tool_use_id"] == server_use["id"]
        # The actual search results carry through (urls + titles).
        assert len(tool_result["content"]) == 2
        assert tool_result["content"][0]["url"] == "https://docs.litellm.ai/"

    @pytest.mark.asyncio
    async def test_litellm_standard_tool_short_circuit_stays_text_only(self):
        """Non-native tool → existing text-only short-circuit, no regression."""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(
            logger,
            "_execute_search",
            new=AsyncMock(return_value=("Title: x\nURL: y", _make_search_response())),
        ):
            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[{"role": "user", "content": "search query"}],
                tools=[
                    {
                        "name": "litellm_web_search",
                        "input_schema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }
                ],
                custom_llm_provider="github_copilot",
            )

        assert result is not None
        block_types = [b["type"] for b in result["content"]]
        assert block_types == ["text"]

    @pytest.mark.asyncio
    async def test_native_short_circuit_failure_still_emits_blocks(self):
        """Search failure on native path: emit blocks with empty results +
        the legacy text-error block, so the client gets a well-formed
        response instead of a malformed half-shape."""
        logger = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

        with patch.object(logger, "_execute_search", side_effect=RuntimeError("boom")):
            result = await logger.try_short_circuit_search(
                model="github_copilot/claude-sonnet-4",
                messages=[{"role": "user", "content": "search query"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                custom_llm_provider="github_copilot",
            )

        assert result is not None
        block_types = [b["type"] for b in result["content"]]
        assert block_types == ["server_tool_use", "web_search_tool_result", "text"]
        tool_result = result["content"][1]
        assert tool_result["content"] == []
        text_block = result["content"][2]
        assert "Search failed" in text_block["text"]


class TestLegacyPathMatchesNewPath:
    """The legacy ``_execute_agentic_loop`` must inject blocks too."""

    @pytest.mark.asyncio
    async def test_legacy_path_injects_when_flag_set(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        tool_calls = [
            {
                "id": "toolu_legacy",
                "type": "tool_use",
                "name": "litellm_web_search",
                "input": {"query": "q"},
            }
        ]
        patch_obj = AgenticLoopRequestPatch(
            model="bedrock/claude",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
            optional_params={},
        )
        followup_response = {
            "id": "msg_followup",
            "content": [{"type": "text", "text": "final answer"}],
        }

        with (
            patch.object(
                logger,
                "_build_anthropic_request_patch",
                new=AsyncMock(return_value=(patch_obj, [_make_search_response()])),
            ),
            patch(
                "litellm.integrations.websearch_interception.handler.anthropic_messages.acreate",
                new=AsyncMock(return_value=followup_response),
            ),
        ):
            out = await logger._execute_agentic_loop(
                model="bedrock/claude",
                messages=[],
                tool_calls=tool_calls,
                thinking_blocks=[],
                anthropic_messages_optional_request_params={},
                logging_obj=MagicMock(model_call_details={}),
                stream=False,
                kwargs={WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY: True},
            )

        assert out["content"][0]["type"] == "server_tool_use"
        assert out["content"][0]["id"].startswith("srvtoolu_")
        assert out["content"][1]["type"] == "web_search_tool_result"
        assert out["content"][1]["tool_use_id"] == out["content"][0]["id"]
        assert out["content"][1]["tool_use_id"] != "toolu_legacy"
        assert out["content"][2]["type"] == "text"


class TestNativeResultBlockUsesServerToolId:
    """Regression for the multi-turn 400 on Bedrock/Anthropic Claude.

    The agentic-loop path must mint ``srvtoolu_`` ids and pair each
    ``web_search_tool_result`` with a ``server_tool_use`` block, rather than
    reusing the client's original ``toolu_`` tool_call id. The reused id has no
    matching ``server_tool_use`` block and fails ``^srvtoolu_`` validation when
    the assistant turn is replayed on the next request."""

    def test_blocks_never_reuse_client_toolu_id(self):
        tool_calls = [
            {
                "id": "toolu_client_original",
                "type": "tool_use",
                "name": "web_search",
                "input": {"query": "weather in tokyo"},
            }
        ]
        blocks = WebSearchInterceptionLogger._build_native_result_blocks(
            tool_calls=tool_calls,
            structured_results=[_make_search_response()],
        )

        assert [b["type"] for b in blocks] == [
            "server_tool_use",
            "web_search_tool_result",
        ]
        server_use, tool_result = blocks
        assert server_use["id"].startswith("srvtoolu_")
        assert server_use["name"] == "web_search"
        assert server_use["input"] == {"query": "weather in tokyo"}
        assert tool_result["tool_use_id"] == server_use["id"]
        assert tool_result["tool_use_id"].startswith("srvtoolu_")
        assert "toolu_client_original" not in (
            server_use["id"],
            tool_result["tool_use_id"],
        )
        assert tool_result["content"][0]["url"] == "https://docs.litellm.ai/"

    def test_multiple_tool_calls_get_distinct_server_ids(self):
        tool_calls = [
            {"id": "toolu_a", "name": "web_search", "input": {"query": "a"}},
            {"id": "toolu_b", "name": "web_search", "input": {"query": "b"}},
        ]
        blocks = WebSearchInterceptionLogger._build_native_result_blocks(
            tool_calls=tool_calls,
            structured_results=[_make_search_response(), None],
        )

        assert [b["type"] for b in blocks] == [
            "server_tool_use",
            "web_search_tool_result",
            "server_tool_use",
            "web_search_tool_result",
        ]
        first_use, first_result, second_use, second_result = blocks
        assert first_use["id"] != second_use["id"]
        assert first_result["tool_use_id"] == first_use["id"]
        assert second_result["tool_use_id"] == second_use["id"]
        assert all(b["id"].startswith("srvtoolu_") for b in (first_use, second_use))

    def test_missing_query_emits_empty_input(self):
        tool_calls = [{"id": "toolu_x", "name": "web_search", "input": {}}]
        blocks = WebSearchInterceptionLogger._build_native_result_blocks(
            tool_calls=tool_calls,
            structured_results=[None],
        )

        server_use, tool_result = blocks
        assert server_use["input"] == {}
        assert server_use["id"].startswith("srvtoolu_")
        assert tool_result["tool_use_id"] == server_use["id"]

    def test_server_tool_use_name_is_native_not_intercepted(self):
        # After pre-request conversion the model's tool_call carries the
        # litellm_web_search name; the native block must report web_search.
        tool_calls = [{"id": "toolu_x", "name": "litellm_web_search", "input": {"query": "q"}}]
        blocks = WebSearchInterceptionLogger._build_native_result_blocks(
            tool_calls=tool_calls,
            structured_results=[None],
        )

        server_use = blocks[0]
        assert server_use["name"] == "web_search"
