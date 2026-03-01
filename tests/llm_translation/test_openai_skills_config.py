"""
Unit tests for OpenAI Skills API configuration and transformations
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from litellm.llms.openai.skills.transformation import OpenAISkillsConfig
from litellm.types.llms.anthropic_skills import (
    CreateSkillRequest,
    DeleteSkillResponse,
    ListSkillsParams,
    ListSkillsResponse,
    Skill,
)
from litellm.types.llms.openai_skills import (
    OpenAIDeletedSkill,
    OpenAISkill,
    OpenAISkillList,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


@pytest.fixture
def config():
    return OpenAISkillsConfig()


@pytest.fixture
def litellm_params():
    return GenericLiteLLMParams(
        api_key="test-openai-key-123",
        api_base="https://api.openai.com",
    )


@pytest.fixture
def litellm_params_custom_base():
    return GenericLiteLLMParams(
        api_key="test-openai-key-456",
        api_base="https://custom.openai.proxy.com",
    )


class TestOpenAISkillsConfigProvider:
    """Tests for provider identity."""

    def test_custom_llm_provider_is_openai(self, config):
        assert config.custom_llm_provider == LlmProviders.OPENAI


class TestOpenAISkillsConfigValidateEnvironment:
    """Tests for validate_environment (header generation)."""

    def test_adds_bearer_token(self, config, litellm_params):
        headers = config.validate_environment(
            headers={}, litellm_params=litellm_params
        )
        assert headers["Authorization"] == "Bearer test-openai-key-123"
        assert "Content-Type" not in headers  # httpx sets Content-Type per request type

    def test_raises_without_api_key(self, config):
        params = GenericLiteLLMParams(api_key=None, api_base=None)
        with patch("litellm.api_key", None), \
             patch("litellm.openai_key", None), \
             patch("litellm.secret_managers.main.get_secret_str", return_value=None):
            with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
                config.validate_environment(headers={}, litellm_params=params)

    def test_preserves_existing_headers(self, config, litellm_params):
        headers = config.validate_environment(
            headers={"X-Custom": "value"}, litellm_params=litellm_params
        )
        assert headers["X-Custom"] == "value"
        assert "Authorization" in headers


class TestOpenAISkillsConfigURLBuilding:
    """Tests for get_complete_url and get_api_base."""

    def test_skills_list_url(self, config):
        url = config.get_complete_url(
            api_base="https://api.openai.com", endpoint="skills"
        )
        assert url == "https://api.openai.com/v1/skills"

    def test_skills_detail_url(self, config):
        url = config.get_complete_url(
            api_base="https://api.openai.com",
            endpoint="skills",
            skill_id="sk_abc123",
        )
        assert url == "https://api.openai.com/v1/skills/sk_abc123"

    def test_default_api_base(self, config):
        url = config.get_complete_url(api_base=None, endpoint="skills")
        assert url == "https://api.openai.com/v1/skills"

    def test_custom_api_base(self, config):
        url = config.get_complete_url(
            api_base="https://custom.proxy.com", endpoint="skills"
        )
        assert url == "https://custom.proxy.com/v1/skills"

    def test_get_api_base_from_params(self, config, litellm_params):
        assert config.get_api_base(litellm_params) == "https://api.openai.com"

    def test_get_api_base_custom(self, config, litellm_params_custom_base):
        assert (
            config.get_api_base(litellm_params_custom_base)
            == "https://custom.openai.proxy.com"
        )

    def test_get_api_base_default_fallback(self, config):
        params = GenericLiteLLMParams(api_base=None)
        assert config.get_api_base(params) == "https://api.openai.com"


class TestOpenAISkillsConfigCreateSkill:
    """Tests for create skill transform."""

    def test_transform_create_request_passthrough(self, config, litellm_params):
        request: CreateSkillRequest = {"files": ["file_data"]}
        result = config.transform_create_skill_request(
            create_request=request,
            litellm_params=litellm_params,
            headers={},
        )
        assert result == {"files": ["file_data"]}

    def test_transform_create_request_strips_none(self, config, litellm_params):
        request: CreateSkillRequest = {
            "display_title": None,
            "files": ["data"],
        }
        result = config.transform_create_skill_request(
            create_request=request,
            litellm_params=litellm_params,
            headers={},
        )
        assert "display_title" not in result
        assert result["files"] == ["data"]

    def test_transform_create_response(self, config):
        response_data = {
            "id": "sk_test123",
            "created_at": 1709251200,
            "default_version": "1",
            "description": "Test skill",
            "latest_version": "1",
            "name": "my-skill",
            "object": "skill",
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data

        result = config.transform_create_skill_response(
            raw_response=mock_response,
            logging_obj=MagicMock(),
        )

        assert isinstance(result, Skill)
        assert result.id == "sk_test123"
        assert result.display_title == "my-skill"
        assert result.latest_version == "1"
        assert result.source == "custom"
        assert result.type == "skill"


class TestOpenAISkillsConfigListSkills:
    """Tests for list skills transform."""

    def test_transform_list_request_basic(self, config, litellm_params):
        params: ListSkillsParams = {"limit": 10}
        url, query_params = config.transform_list_skills_request(
            list_params=params,
            litellm_params=litellm_params,
            headers={},
        )
        assert url == "https://api.openai.com/v1/skills"
        assert query_params == {"limit": 10}

    def test_transform_list_request_maps_page_to_after(self, config, litellm_params):
        params: ListSkillsParams = {"limit": 5, "page": "sk_cursor_abc"}
        url, query_params = config.transform_list_skills_request(
            list_params=params,
            litellm_params=litellm_params,
            headers={},
        )
        assert query_params["after"] == "sk_cursor_abc"
        assert "page" not in query_params

    def test_transform_list_response(self, config):
        response_data = {
            "data": [
                {
                    "id": "sk_1",
                    "created_at": 1709251200,
                    "name": "skill-1",
                    "object": "skill",
                    "latest_version": "1",
                },
                {
                    "id": "sk_2",
                    "created_at": 1709251300,
                    "name": "skill-2",
                    "object": "skill",
                    "latest_version": "2",
                },
            ],
            "first_id": "sk_1",
            "last_id": "sk_2",
            "has_more": True,
            "object": "list",
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data

        result = config.transform_list_skills_response(
            raw_response=mock_response,
            logging_obj=MagicMock(),
        )

        assert isinstance(result, ListSkillsResponse)
        assert len(result.data) == 2
        assert result.data[0].id == "sk_1"
        assert result.data[0].display_title == "skill-1"
        assert result.next_page == "sk_2"
        assert result.has_more is True


class TestOpenAISkillsConfigGetSkill:
    """Tests for get skill transform."""

    def test_transform_get_request(self, config, litellm_params):
        url, headers = config.transform_get_skill_request(
            skill_id="sk_abc",
            api_base="https://api.openai.com",
            litellm_params=litellm_params,
            headers={"Authorization": "Bearer key"},
        )
        assert url == "https://api.openai.com/v1/skills/sk_abc"
        assert headers["Authorization"] == "Bearer key"

    def test_transform_get_response(self, config):
        response_data = {
            "id": "sk_abc",
            "created_at": 1709251200,
            "name": "my-skill",
            "description": "A test skill",
            "default_version": "1",
            "latest_version": "2",
            "object": "skill",
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data

        result = config.transform_get_skill_response(
            raw_response=mock_response,
            logging_obj=MagicMock(),
        )

        assert isinstance(result, Skill)
        assert result.id == "sk_abc"
        assert result.display_title == "my-skill"


class TestOpenAISkillsConfigDeleteSkill:
    """Tests for delete skill transform."""

    def test_transform_delete_request(self, config, litellm_params):
        url, headers = config.transform_delete_skill_request(
            skill_id="sk_to_delete",
            api_base="https://api.openai.com",
            litellm_params=litellm_params,
            headers={"Authorization": "Bearer key"},
        )
        assert url == "https://api.openai.com/v1/skills/sk_to_delete"

    def test_transform_delete_response(self, config):
        response_data = {
            "id": "sk_to_delete",
            "deleted": True,
            "object": "skill.deleted",
        }
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = response_data

        result = config.transform_delete_skill_response(
            raw_response=mock_response,
            logging_obj=MagicMock(),
        )

        assert isinstance(result, DeleteSkillResponse)
        assert result.id == "sk_to_delete"
        assert result.type == "skill_deleted"


class TestOpenAISkillCanonicalMapping:
    """Tests for _openai_skill_to_canonical helper."""

    def test_maps_name_to_display_title(self):
        data = {
            "id": "sk_1",
            "created_at": 1709251200,
            "name": "My Skill",
            "object": "skill",
        }
        result = OpenAISkillsConfig._openai_skill_to_canonical(data)
        assert result.display_title == "My Skill"

    def test_converts_unix_timestamp_to_iso(self):
        data = {
            "id": "sk_1",
            "created_at": 1709251200,
            "object": "skill",
        }
        result = OpenAISkillsConfig._openai_skill_to_canonical(data)
        assert "2024" in result.created_at
        assert "T" in result.created_at

    def test_defaults_source_to_custom(self):
        data = {
            "id": "sk_1",
            "created_at": 1709251200,
            "object": "skill",
        }
        result = OpenAISkillsConfig._openai_skill_to_canonical(data)
        assert result.source == "custom"

    def test_maps_default_version(self):
        data = {
            "id": "sk_1",
            "created_at": 1709251200,
            "default_version": "3",
            "object": "skill",
        }
        result = OpenAISkillsConfig._openai_skill_to_canonical(data)
        assert result.default_version == "3"

    def test_maps_int_default_version_to_str(self):
        data = {
            "id": "sk_1",
            "created_at": 1709251200,
            "default_version": 2,
            "object": "skill",
        }
        result = OpenAISkillsConfig._openai_skill_to_canonical(data)
        assert result.default_version == "2"

    def test_handles_missing_optional_fields(self):
        data = {
            "id": "sk_1",
            "created_at": 0,
            "object": "skill",
        }
        result = OpenAISkillsConfig._openai_skill_to_canonical(data)
        assert result.id == "sk_1"
        assert result.display_title is None
        assert result.latest_version is None
        assert result.default_version is None


# ──────────────────────────────────────────────
# Tests for OpenAI-specific extended endpoints
# ──────────────────────────────────────────────


class TestOpenAISkillsConfigUpdateSkill:
    """Tests for update_skill transforms."""

    def test_transform_update_skill_request_url(self):
        config = OpenAISkillsConfig()
        params = GenericLiteLLMParams(api_key="test-key")
        url, headers, body = config.transform_update_skill_request(
            skill_id="sk_123",
            update_data={"default_version": "3"},
            api_base="https://api.openai.com",
            litellm_params=params,
            headers={"Authorization": "Bearer test-key"},
        )
        assert url == "https://api.openai.com/v1/skills/sk_123"
        assert body == {"default_version": "3"}

    def test_transform_update_skill_request_strips_none(self):
        config = OpenAISkillsConfig()
        params = GenericLiteLLMParams(api_key="test-key")
        url, headers, body = config.transform_update_skill_request(
            skill_id="sk_123",
            update_data={"default_version": 2, "extra": None},
            api_base="https://api.openai.com",
            litellm_params=params,
            headers={},
        )
        assert body == {"default_version": "2"}

    def test_transform_update_skill_response(self):
        config = OpenAISkillsConfig()
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = {
            "id": "sk_1",
            "created_at": 1700000000,
            "name": "Updated Skill",
            "default_version": "3",
            "latest_version": "3",
            "object": "skill",
        }
        logging_obj = MagicMock()
        result = config.transform_update_skill_response(raw_response, logging_obj)
        assert isinstance(result, Skill)
        assert result.id == "sk_1"
        assert result.display_title == "Updated Skill"


class TestOpenAISkillsConfigGetSkillContent:
    """Tests for get_skill_content transforms."""

    def test_transform_get_skill_content_request(self):
        config = OpenAISkillsConfig()
        params = GenericLiteLLMParams(api_key="test-key")
        url, headers = config.transform_get_skill_content_request(
            skill_id="sk_abc",
            api_base="https://api.openai.com",
            litellm_params=params,
            headers={"Authorization": "Bearer test-key"},
        )
        assert url == "https://api.openai.com/v1/skills/sk_abc/content"

    def test_transform_get_skill_content_response(self):
        config = OpenAISkillsConfig()
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.headers = {"content-type": "application/json"}
        raw_response.json.return_value = {"content": "base64data", "type": "zip"}
        logging_obj = MagicMock()
        result = config.transform_get_skill_content_response(raw_response, logging_obj)
        assert result["content"] == "base64data"


class TestOpenAISkillsConfigCreateSkillVersion:
    """Tests for create_skill_version transforms."""

    def test_transform_create_skill_version_request(self):
        config = OpenAISkillsConfig()
        params = GenericLiteLLMParams(api_key="test-key")
        url, headers, body = config.transform_create_skill_version_request(
            skill_id="sk_1",
            create_request={"files": ["data"]},
            api_base="https://api.openai.com",
            litellm_params=params,
            headers={},
        )
        assert url == "https://api.openai.com/v1/skills/sk_1/versions"
        assert body == {"files": ["data"]}

    def test_transform_create_skill_version_response(self):
        config = OpenAISkillsConfig()
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = {
            "id": "sv_1",
            "created_at": 1700000000,
            "version": "2",
            "skill_id": "sk_1",
            "object": "skill.version",
        }
        logging_obj = MagicMock()
        result = config.transform_create_skill_version_response(raw_response, logging_obj)
        assert result["id"] == "sv_1"
        assert result["version"] == "2"


class TestOpenAISkillsConfigListSkillVersions:
    """Tests for list_skill_versions transforms."""

    def test_transform_list_skill_versions_request(self):
        config = OpenAISkillsConfig()
        params = GenericLiteLLMParams(api_key="test-key")
        url, headers, query_params = config.transform_list_skill_versions_request(
            skill_id="sk_1",
            list_params={"limit": 10, "after": "sv_5"},
            api_base="https://api.openai.com",
            litellm_params=params,
            headers={},
        )
        assert url == "https://api.openai.com/v1/skills/sk_1/versions"
        assert query_params == {"limit": 10, "after": "sv_5"}

    def test_transform_list_skill_versions_request_empty_params(self):
        config = OpenAISkillsConfig()
        params = GenericLiteLLMParams(api_key="test-key")
        url, headers, query_params = config.transform_list_skill_versions_request(
            skill_id="sk_1",
            list_params={},
            api_base="https://api.openai.com",
            litellm_params=params,
            headers={},
        )
        assert query_params == {}

    def test_transform_list_skill_versions_response(self):
        config = OpenAISkillsConfig()
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = {
            "data": [{"id": "sv_1", "version": "1"}],
            "has_more": True,
            "first_id": "sv_1",
            "last_id": "sv_1",
            "object": "list",
        }
        logging_obj = MagicMock()
        result = config.transform_list_skill_versions_response(raw_response, logging_obj)
        assert len(result["data"]) == 1
        assert result["has_more"] is True


class TestOpenAISkillsConfigGetSkillVersion:
    """Tests for get_skill_version transforms."""

    def test_transform_get_skill_version_request(self):
        config = OpenAISkillsConfig()
        params = GenericLiteLLMParams(api_key="test-key")
        url, headers = config.transform_get_skill_version_request(
            skill_id="sk_1",
            version="3",
            api_base="https://api.openai.com",
            litellm_params=params,
            headers={},
        )
        assert url == "https://api.openai.com/v1/skills/sk_1/versions/3"

    def test_transform_get_skill_version_response(self):
        config = OpenAISkillsConfig()
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = {
            "id": "sv_3",
            "version": "3",
            "skill_id": "sk_1",
            "object": "skill.version",
        }
        logging_obj = MagicMock()
        result = config.transform_get_skill_version_response(raw_response, logging_obj)
        assert result["id"] == "sv_3"


class TestOpenAISkillsConfigDeleteSkillVersion:
    """Tests for delete_skill_version transforms."""

    def test_transform_delete_skill_version_request(self):
        config = OpenAISkillsConfig()
        params = GenericLiteLLMParams(api_key="test-key")
        url, headers = config.transform_delete_skill_version_request(
            skill_id="sk_1",
            version="2",
            api_base="https://api.openai.com",
            litellm_params=params,
            headers={},
        )
        assert url == "https://api.openai.com/v1/skills/sk_1/versions/2"

    def test_transform_delete_skill_version_response(self):
        config = OpenAISkillsConfig()
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = {
            "id": "sv_2",
            "deleted": True,
            "object": "skill.version.deleted",
            "version": "2",
        }
        logging_obj = MagicMock()
        result = config.transform_delete_skill_version_response(raw_response, logging_obj)
        assert result["deleted"] is True
        assert result["id"] == "sv_2"


class TestOpenAISkillsConfigGetSkillVersionContent:
    """Tests for get_skill_version_content transforms."""

    def test_transform_get_skill_version_content_request(self):
        config = OpenAISkillsConfig()
        params = GenericLiteLLMParams(api_key="test-key")
        url, headers = config.transform_get_skill_version_content_request(
            skill_id="sk_1",
            version="4",
            api_base="https://api.openai.com",
            litellm_params=params,
            headers={},
        )
        assert url == "https://api.openai.com/v1/skills/sk_1/versions/4/content"

    def test_transform_get_skill_version_content_response(self):
        config = OpenAISkillsConfig()
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.headers = {"content-type": "application/json"}
        raw_response.json.return_value = {"content": "version-content-data"}
        logging_obj = MagicMock()
        result = config.transform_get_skill_version_content_response(raw_response, logging_obj)
        assert result["content"] == "version-content-data"


class TestBaseSkillsConfigNotImplemented:
    """Verify base class raises NotImplementedError for new methods."""

    def test_anthropic_raises_for_update_skill(self):
        from litellm.llms.anthropic.skills.transformation import AnthropicSkillsConfig
        config = AnthropicSkillsConfig()
        params = GenericLiteLLMParams(api_key="test-key")
        with pytest.raises(NotImplementedError):
            config.transform_update_skill_request(
                skill_id="sk_1",
                update_data={},
                api_base="https://api.anthropic.com",
                litellm_params=params,
                headers={},
            )

    def test_anthropic_raises_for_get_skill_content(self):
        from litellm.llms.anthropic.skills.transformation import AnthropicSkillsConfig
        config = AnthropicSkillsConfig()
        params = GenericLiteLLMParams(api_key="test-key")
        with pytest.raises(NotImplementedError):
            config.transform_get_skill_content_request(
                skill_id="sk_1",
                api_base="https://api.anthropic.com",
                litellm_params=params,
                headers={},
            )

    def test_anthropic_raises_for_create_skill_version(self):
        from litellm.llms.anthropic.skills.transformation import AnthropicSkillsConfig
        config = AnthropicSkillsConfig()
        params = GenericLiteLLMParams(api_key="test-key")
        with pytest.raises(NotImplementedError):
            config.transform_create_skill_version_request(
                skill_id="sk_1",
                create_request={},
                api_base="https://api.anthropic.com",
                litellm_params=params,
                headers={},
            )
