"""
Tests for tag merging logic in litellm_pre_call_utils.

Verifies fix for https://github.com/BerriAI/litellm/issues/14052:
x-litellm-tags must be merged with team/project tags, not overwrite them.
"""

import pytest

from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup


class TestAddRequestTagToMetadata:
    """Tests for ProxyRequestSetup.add_request_tag_to_metadata"""

    def test_header_tags_only(self):
        headers = {"x-litellm-tags": "tag1, tag2"}
        data: dict = {}
        result = LiteLLMProxyRequestSetup.add_request_tag_to_metadata(
            llm_router=None, headers=headers, data=data
        )
        assert result == ["tag1", "tag2"]

    def test_body_tags_only(self):
        headers: dict = {}
        data = {"tags": ["body1", "body2"]}
        result = LiteLLMProxyRequestSetup.add_request_tag_to_metadata(
            llm_router=None, headers=headers, data=data
        )
        assert result == ["body1", "body2"]

    def test_header_and_body_tags_merged(self):
        headers = {"x-litellm-tags": "header1, header2"}
        data = {"tags": ["body1", "body2"]}
        result = LiteLLMProxyRequestSetup.add_request_tag_to_metadata(
            llm_router=None, headers=headers, data=data
        )
        assert result == ["header1", "header2", "body1", "body2"]

    def test_header_and_body_tags_deduplicated(self):
        headers = {"x-litellm-tags": "shared, header_only"}
        data = {"tags": ["shared", "body_only"]}
        result = LiteLLMProxyRequestSetup.add_request_tag_to_metadata(
            llm_router=None, headers=headers, data=data
        )
        assert result == ["shared", "header_only", "body_only"]

    def test_no_tags_returns_none(self):
        result = LiteLLMProxyRequestSetup.add_request_tag_to_metadata(
            llm_router=None, headers={}, data={}
        )
        assert result is None

    def test_header_tags_list_format(self):
        headers = {"x-litellm-tags": ["a", "b"]}
        data: dict = {}
        result = LiteLLMProxyRequestSetup.add_request_tag_to_metadata(
            llm_router=None, headers=headers, data=data
        )
        assert result == ["a", "b"]


class TestTagMergeWithExistingMetadata:
    """
    Tests the _merge_tags helper directly to verify that request tags
    are merged with pre-existing team/project tags without duplicates.
    """

    @staticmethod
    def _merge(existing_tags, new_tags):
        """Delegate to the real merge implementation."""
        return LiteLLMProxyRequestSetup._merge_tags(
            request_tags=existing_tags, tags_to_add=new_tags
        )

    def test_team_tags_preserved_when_request_tags_added(self):
        existing = ["team:finance", "env:prod"]
        request = ["route:gpt4"]
        merged = self._merge(existing, request)
        assert "team:finance" in merged
        assert "env:prod" in merged
        assert "route:gpt4" in merged

    def test_no_existing_tags_none(self):
        merged = self._merge(None, ["new_tag"])
        assert merged == ["new_tag"]

    def test_no_existing_tags_empty_list(self):
        merged = self._merge([], ["new_tag"])
        assert merged == ["new_tag"]

    def test_no_request_tags(self):
        merged = self._merge(["existing"], None)
        assert merged == ["existing"]

    def test_duplicate_removal(self):
        existing = ["shared", "team_only"]
        request = ["shared", "request_only"]
        merged = self._merge(existing, request)
        assert merged == ["shared", "team_only", "request_only"]

    def test_all_sources_coexist(self):
        existing = ["team:a", "project:b"]
        request = ["header:c", "body:d"]
        merged = self._merge(existing, request)
        assert merged == ["team:a", "project:b", "header:c", "body:d"]

    def test_order_preserved(self):
        existing = ["first", "second"]
        request = ["third", "fourth"]
        merged = self._merge(existing, request)
        assert merged == ["first", "second", "third", "fourth"]

    def test_empty_existing_tags_list(self):
        existing = []
        request = ["new_tag"]
        merged = self._merge(existing, request)
        assert merged == ["new_tag"]

    def test_both_none_returns_empty(self):
        merged = self._merge(None, None)
        assert merged == []

    def test_both_empty_returns_empty(self):
        merged = self._merge([], [])
        assert merged == []



class TestAddLitellmDataToRequestTagMerge:
    """
    Integration test: calls add_litellm_data_to_request end-to-end with both
    team metadata tags AND request x-litellm-tags to verify merge works and
    team tags are always preserved (request tags append, never override).
    """

    @staticmethod
    async def _run_merge(team_tags, header_tags_csv):
        """Helper that drives the full add_litellm_data_to_request flow."""
        from unittest.mock import Mock

        from starlette.datastructures import State
        from starlette.requests import Request

        from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
        from litellm.types.utils import UserAPIKeyAuth

        mock_request = Mock(spec=Request)
        mock_request.url.path = "/chat/completions"
        mock_request.query_params = {}
        mock_request.headers = {"x-litellm-tags": header_tags_csv}
        mock_request.state = State()

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test_key",
            user_id="test_user",
            org_id="test_org",
            team_metadata={"tags": team_tags},
        )

        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
        }

        proxy_config = Mock()
        result = await add_litellm_data_to_request(
            data=data,
            request=mock_request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=proxy_config,
        )
        return result["metadata"]["tags"]

    @pytest.mark.asyncio
    async def test_team_tags_merged_with_request_header_tags(self):
        """Team tags are preserved and request header tags are appended."""
        tags = await self._run_merge(
            team_tags=["team:finance", "env:prod"],
            header_tags_csv="route:gpt4, priority:high",
        )
        # Team tags must always be present (never overridden)
        assert "team:finance" in tags
        assert "env:prod" in tags
        # Request tags are appended
        assert "route:gpt4" in tags
        assert "priority:high" in tags

    @pytest.mark.asyncio
    async def test_duplicate_tags_between_team_and_request(self):
        """Shared tags between team and request are deduplicated."""
        tags = await self._run_merge(
            team_tags=["shared", "team_only"],
            header_tags_csv="shared, request_only",
        )
        assert tags.count("shared") == 1
        assert "team_only" in tags
        assert "request_only" in tags

    @pytest.mark.asyncio
    async def test_team_tags_with_no_request_tags(self):
        """Team tags alone still appear in final metadata."""
        from unittest.mock import Mock

        from starlette.datastructures import State
        from starlette.requests import Request

        from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
        from litellm.types.utils import UserAPIKeyAuth

        mock_request = Mock(spec=Request)
        mock_request.url.path = "/chat/completions"
        mock_request.query_params = {}
        mock_request.headers = {}
        mock_request.state = State()

        user_api_key_dict = UserAPIKeyAuth(
            api_key="test_key",
            user_id="test_user",
            org_id="test_org",
            team_metadata={"tags": ["team:ops"]},
        )
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
        }
        proxy_config = Mock()
        result = await add_litellm_data_to_request(
            data=data,
            request=mock_request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=proxy_config,
        )
        assert "team:ops" in result["metadata"]["tags"]
