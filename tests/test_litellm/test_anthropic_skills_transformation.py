"""
Unit tests for Anthropic Skills API request/response transformation.

These tests validate URL construction, header generation, request payload
building, and response parsing without requiring a live Anthropic API key
or beta access to the Skills API.
"""
from unittest.mock import MagicMock, patch
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.constants import ANTHROPIC_SKILLS_API_BETA_VERSION
from litellm.llms.anthropic.skills.transformation import AnthropicSkillsConfig
from litellm.types.llms.anthropic_skills import (
    CreateSkillRequest,
    DeleteSkillResponse,
    ListSkillsParams,
    ListSkillsResponse,
    Skill,
)
from litellm.types.router import GenericLiteLLMParams


FAKE_API_KEY = "sk-ant-test-key-1234"
FAKE_API_BASE = "https://api.anthropic.com"


def _make_mock_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "https://api.anthropic.com/v1/skills"),
    )


def _make_skill_payload(**kwargs) -> dict:
    defaults = {
        "id": "skill_abc123",
        "created_at": "2025-10-15T12:00:00Z",
        "updated_at": "2025-10-15T12:00:00Z",
        "source": "custom",
        "type": "skill",
        "display_title": "Test Skill",
        "latest_version": "v1",
    }
    defaults.update(kwargs)
    return defaults


class TestAnthropicSkillsConfigURLConstruction:
    def setup_method(self):
        self.config = AnthropicSkillsConfig()

    def test_url_without_skill_id(self):
        url = self.config.get_complete_url(
            api_base=FAKE_API_BASE,
            endpoint="skills",
        )
        assert url == f"{FAKE_API_BASE}/v1/skills"

    def test_url_with_skill_id(self):
        url = self.config.get_complete_url(
            api_base=FAKE_API_BASE,
            endpoint="skills",
            skill_id="skill_abc123",
        )
        assert url == f"{FAKE_API_BASE}/v1/skills/skill_abc123"

    def test_url_falls_back_to_anthropic_default(self):
        with patch(
            "litellm.llms.anthropic.common_utils.AnthropicModelInfo.get_api_base",
            return_value="https://api.anthropic.com",
        ):
            url = self.config.get_complete_url(
                api_base=None,
                endpoint="skills",
            )
        assert url == "https://api.anthropic.com/v1/skills"

    def test_url_with_custom_api_base(self):
        custom_base = "https://my-proxy.example.com"
        url = self.config.get_complete_url(
            api_base=custom_base,
            endpoint="skills",
        )
        assert url == f"{custom_base}/v1/skills"


class TestAnthropicSkillsConfigHeaderValidation:
    def setup_method(self):
        self.config = AnthropicSkillsConfig()

    def _make_litellm_params(self, api_key=FAKE_API_KEY):
        return GenericLiteLLMParams(api_key=api_key)

    def test_sets_api_key_header(self):
        with patch(
            "litellm.llms.anthropic.common_utils.AnthropicModelInfo.get_api_key",
            return_value=FAKE_API_KEY,
        ):
            headers = self.config.validate_environment(
                headers={}, litellm_params=self._make_litellm_params()
            )
        assert headers["x-api-key"] == FAKE_API_KEY

    def test_sets_anthropic_version_header(self):
        with patch(
            "litellm.llms.anthropic.common_utils.AnthropicModelInfo.get_api_key",
            return_value=FAKE_API_KEY,
        ):
            headers = self.config.validate_environment(
                headers={}, litellm_params=self._make_litellm_params()
            )
        assert headers["anthropic-version"] == "2023-06-01"

    def test_sets_skills_beta_header(self):
        with patch(
            "litellm.llms.anthropic.common_utils.AnthropicModelInfo.get_api_key",
            return_value=FAKE_API_KEY,
        ):
            headers = self.config.validate_environment(
                headers={}, litellm_params=self._make_litellm_params()
            )
        assert headers["anthropic-beta"] == ANTHROPIC_SKILLS_API_BETA_VERSION

    def test_merges_existing_beta_header_string(self):
        with patch(
            "litellm.llms.anthropic.common_utils.AnthropicModelInfo.get_api_key",
            return_value=FAKE_API_KEY,
        ):
            headers = self.config.validate_environment(
                headers={"anthropic-beta": "other-beta-2024-01-01"},
                litellm_params=self._make_litellm_params(),
            )
        assert isinstance(headers["anthropic-beta"], list)
        assert "other-beta-2024-01-01" in headers["anthropic-beta"]
        assert ANTHROPIC_SKILLS_API_BETA_VERSION in headers["anthropic-beta"]

    def test_merges_existing_beta_header_list(self):
        with patch(
            "litellm.llms.anthropic.common_utils.AnthropicModelInfo.get_api_key",
            return_value=FAKE_API_KEY,
        ):
            headers = self.config.validate_environment(
                headers={"anthropic-beta": ["other-beta-2024-01-01"]},
                litellm_params=self._make_litellm_params(),
            )
        assert ANTHROPIC_SKILLS_API_BETA_VERSION in headers["anthropic-beta"]
        assert "other-beta-2024-01-01" in headers["anthropic-beta"]

    def test_does_not_duplicate_beta_header(self):
        with patch(
            "litellm.llms.anthropic.common_utils.AnthropicModelInfo.get_api_key",
            return_value=FAKE_API_KEY,
        ):
            headers = self.config.validate_environment(
                headers={"anthropic-beta": ANTHROPIC_SKILLS_API_BETA_VERSION},
                litellm_params=self._make_litellm_params(),
            )
        beta = headers["anthropic-beta"]
        if isinstance(beta, list):
            assert beta.count(ANTHROPIC_SKILLS_API_BETA_VERSION) == 1
        else:
            assert beta == ANTHROPIC_SKILLS_API_BETA_VERSION

    def test_raises_without_api_key(self):
        with patch(
            "litellm.llms.anthropic.common_utils.AnthropicModelInfo.get_api_key",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                self.config.validate_environment(
                    headers={}, litellm_params=self._make_litellm_params(api_key=None)
                )


class TestAnthropicSkillsConfigCreateRequestTransformation:
    def setup_method(self):
        self.config = AnthropicSkillsConfig()
        self.litellm_params = GenericLiteLLMParams(api_key=FAKE_API_KEY)

    def test_display_title_included(self):
        create_request: CreateSkillRequest = {"display_title": "My Skill"}
        body = self.config.transform_create_skill_request(
            create_request=create_request,
            litellm_params=self.litellm_params,
            headers={},
        )
        assert body["display_title"] == "My Skill"

    def test_none_values_excluded(self):
        create_request: CreateSkillRequest = {"display_title": None, "files": None}
        body = self.config.transform_create_skill_request(
            create_request=create_request,
            litellm_params=self.litellm_params,
            headers={},
        )
        assert "display_title" not in body
        assert "files" not in body

    def test_empty_request_produces_empty_body(self):
        create_request: CreateSkillRequest = {}
        body = self.config.transform_create_skill_request(
            create_request=create_request,
            litellm_params=self.litellm_params,
            headers={},
        )
        assert body == {}


class TestAnthropicSkillsConfigListRequestTransformation:
    def setup_method(self):
        self.config = AnthropicSkillsConfig()
        self.litellm_params = GenericLiteLLMParams(api_key=FAKE_API_KEY)

    def test_limit_included_in_query_params(self):
        list_params: ListSkillsParams = {"limit": 25}
        with patch(
            "litellm.llms.anthropic.common_utils.AnthropicModelInfo.get_api_base",
            return_value=FAKE_API_BASE,
        ):
            url, query_params = self.config.transform_list_skills_request(
                list_params=list_params,
                litellm_params=self.litellm_params,
                headers={},
            )
        assert query_params["limit"] == 25
        assert url == f"{FAKE_API_BASE}/v1/skills"

    def test_source_filter_included(self):
        list_params: ListSkillsParams = {"source": "custom"}
        with patch(
            "litellm.llms.anthropic.common_utils.AnthropicModelInfo.get_api_base",
            return_value=FAKE_API_BASE,
        ):
            _, query_params = self.config.transform_list_skills_request(
                list_params=list_params,
                litellm_params=self.litellm_params,
                headers={},
            )
        assert query_params["source"] == "custom"

    def test_empty_params_produce_empty_query(self):
        list_params: ListSkillsParams = {}
        with patch(
            "litellm.llms.anthropic.common_utils.AnthropicModelInfo.get_api_base",
            return_value=FAKE_API_BASE,
        ):
            _, query_params = self.config.transform_list_skills_request(
                list_params=list_params,
                litellm_params=self.litellm_params,
                headers={},
            )
        assert query_params == {}


class TestAnthropicSkillsConfigResponseTransformation:
    def setup_method(self):
        self.config = AnthropicSkillsConfig()
        self.logging_obj = MagicMock()

    def test_create_skill_response_parses_skill(self):
        payload = _make_skill_payload()
        raw = _make_mock_response(payload)
        skill = self.config.transform_create_skill_response(
            raw_response=raw, logging_obj=self.logging_obj
        )
        assert isinstance(skill, Skill)
        assert skill.id == "skill_abc123"
        assert skill.source == "custom"
        assert skill.display_title == "Test Skill"

    def test_get_skill_response_parses_skill(self):
        payload = _make_skill_payload(id="skill_xyz", display_title="Another")
        raw = _make_mock_response(payload)
        skill = self.config.transform_get_skill_response(
            raw_response=raw, logging_obj=self.logging_obj
        )
        assert isinstance(skill, Skill)
        assert skill.id == "skill_xyz"
        assert skill.display_title == "Another"

    def test_list_skills_response_parses_list(self):
        payload = {
            "data": [_make_skill_payload(), _make_skill_payload(id="skill_def456")],
            "has_more": False,
            "next_page": None,
        }
        raw = _make_mock_response(payload)
        result = self.config.transform_list_skills_response(
            raw_response=raw, logging_obj=self.logging_obj
        )
        assert isinstance(result, ListSkillsResponse)
        assert len(result.data) == 2
        assert result.data[0].id == "skill_abc123"
        assert result.data[1].id == "skill_def456"
        assert result.has_more is False

    def test_list_skills_response_with_pagination(self):
        payload = {
            "data": [_make_skill_payload()],
            "has_more": True,
            "next_page": "page_token_xyz",
        }
        raw = _make_mock_response(payload)
        result = self.config.transform_list_skills_response(
            raw_response=raw, logging_obj=self.logging_obj
        )
        assert result.has_more is True
        assert result.next_page == "page_token_xyz"

    def test_delete_skill_response_parses_correctly(self):
        payload = {"id": "skill_abc123", "type": "skill_deleted"}
        raw = _make_mock_response(payload)
        result = self.config.transform_delete_skill_response(
            raw_response=raw, logging_obj=self.logging_obj
        )
        assert isinstance(result, DeleteSkillResponse)
        assert result.id == "skill_abc123"
        assert result.type == "skill_deleted"

    def test_skill_response_optional_fields_default(self):
        payload = {
            "id": "skill_minimal",
            "created_at": "2025-10-15T12:00:00Z",
            "updated_at": "2025-10-15T12:00:00Z",
            "source": "anthropic",
            "type": "skill",
        }
        raw = _make_mock_response(payload)
        skill = self.config.transform_create_skill_response(
            raw_response=raw, logging_obj=self.logging_obj
        )
        assert skill.display_title is None
        assert skill.latest_version is None
