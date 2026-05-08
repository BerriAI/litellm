"""
Unit tests for OpenAI Skills API request/response transformation.

These tests keep the OpenAI skills provider support deterministic and do not
require a live OpenAI API key.
"""

from unittest.mock import MagicMock, patch

import httpx

import litellm
from litellm.llms.openai.skills.transformation import OpenAISkillsConfig
from litellm.types.llms.anthropic_skills import (
    CreateSkillRequest,
    DeleteSkillResponse,
    ListSkillsParams,
    ListSkillsResponse,
    Skill,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


FAKE_API_KEY = "sk-test-key"
FAKE_API_BASE = "https://api.openai.com"


def _make_mock_response(
    json_data: dict, status_code: int = 200, method: str = "POST"
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request(method, "https://api.openai.com/v1/skills"),
    )


def _make_openai_skill_payload(**kwargs) -> dict:
    defaults = {
        "id": "skill_abc123",
        "object": "skill",
        "created_at": 1780000000,
        "updated_at": 1780000100,
        "name": "test-skill",
        "latest_version": "1",
    }
    defaults.update(kwargs)
    return defaults


class TestOpenAISkillsConfigRegistration:
    def test_provider_config_manager_returns_openai_skills_config(self):
        config = ProviderConfigManager.get_provider_skills_api_config(
            provider=LlmProviders.OPENAI
        )

        assert isinstance(config, OpenAISkillsConfig)


class TestOpenAISkillsConfigURLConstruction:
    def setup_method(self):
        self.config = OpenAISkillsConfig()

    def test_url_without_skill_id(self):
        url = self.config.get_complete_url(
            api_base=FAKE_API_BASE,
            endpoint="skills",
        )

        assert url == f"{FAKE_API_BASE}/v1/skills"

    def test_url_with_skill_id_encodes_path_segment(self):
        url = self.config.get_complete_url(
            api_base=FAKE_API_BASE,
            endpoint="skills",
            skill_id="../../files?x=1#frag",
        )

        assert url == f"{FAKE_API_BASE}/v1/skills/..%2F..%2Ffiles%3Fx%3D1%23frag"

    def test_api_base_with_trailing_v1_is_not_duplicated(self):
        url = self.config.get_complete_url(
            api_base=f"{FAKE_API_BASE}/v1/",
            endpoint="skills",
        )

        assert url == f"{FAKE_API_BASE}/v1/skills"


class TestOpenAISkillsConfigHeaderValidation:
    def setup_method(self):
        self.config = OpenAISkillsConfig()

    def test_sets_bearer_auth_header(self):
        headers = self.config.validate_environment(
            headers={}, litellm_params=GenericLiteLLMParams(api_key=FAKE_API_KEY)
        )

        assert headers["Authorization"] == f"Bearer {FAKE_API_KEY}"
        assert "content-type" not in headers

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "openai_key", None)
        with patch(
            "litellm.llms.openai.skills.transformation.get_secret_str",
            return_value=None,
        ):
            try:
                self.config.validate_environment(
                    headers={}, litellm_params=GenericLiteLLMParams(api_key=None)
                )
            except ValueError as e:
                assert "OpenAI API key is required for Skills API" in str(e)
            else:
                raise AssertionError("Expected ValueError")


class TestOpenAISkillsConfigRequestTransformation:
    def setup_method(self):
        self.config = OpenAISkillsConfig()
        self.litellm_params = GenericLiteLLMParams(api_key=FAKE_API_KEY)

    def test_create_request_strips_anthropic_display_title(self):
        create_request: CreateSkillRequest = {
            "display_title": "Ignored by OpenAI",
            "files": ["skill.zip"],
        }

        body = self.config.transform_create_skill_request(
            create_request=create_request,
            litellm_params=self.litellm_params,
            headers={},
        )

        assert body == {"files": ["skill.zip"]}

    def test_openai_uses_singular_file_multipart_field(self):
        assert self.config.get_skill_file_field_name() == "file"

    def test_list_page_maps_to_after_cursor(self):
        list_params: ListSkillsParams = {"limit": 25, "page": "skill_prev"}

        url, query_params = self.config.transform_list_skills_request(
            list_params=list_params,
            litellm_params=self.litellm_params,
            headers={},
        )

        assert url == f"{FAKE_API_BASE}/v1/skills"
        assert query_params == {"limit": 25, "after": "skill_prev"}

    def test_list_extra_openai_cursors_pass_through(self):
        list_params: ListSkillsParams = {
            "limit": 10,
            "after": "skill_after",
            "before": "skill_before",
        }  # type: ignore[typeddict-unknown-key]

        _, query_params = self.config.transform_list_skills_request(
            list_params=list_params,
            litellm_params=self.litellm_params,
            headers={},
        )

        assert query_params == {
            "limit": 10,
            "after": "skill_after",
            "before": "skill_before",
        }


class TestOpenAISkillsConfigResponseTransformation:
    def setup_method(self):
        self.config = OpenAISkillsConfig()
        self.logging_obj = MagicMock()

    def test_create_skill_response_maps_openai_fields(self):
        raw = _make_mock_response(_make_openai_skill_payload())

        skill = self.config.transform_create_skill_response(
            raw_response=raw, logging_obj=self.logging_obj
        )

        assert isinstance(skill, Skill)
        assert skill.id == "skill_abc123"
        assert skill.display_title == "test-skill"
        assert skill.source == "openai"
        assert skill.type == "skill"
        assert skill.created_at == "2026-05-28T20:26:40Z"

    def test_get_skill_response_preserves_iso_timestamps(self):
        raw = _make_mock_response(
            _make_openai_skill_payload(
                created_at="2026-05-01T12:00:00Z",
                updated_at="2026-05-01T12:01:00Z",
            ),
            method="GET",
        )

        skill = self.config.transform_get_skill_response(
            raw_response=raw, logging_obj=self.logging_obj
        )

        assert skill.created_at == "2026-05-01T12:00:00Z"
        assert skill.updated_at == "2026-05-01T12:01:00Z"

    def test_list_skills_response_maps_openai_pagination(self):
        payload = {
            "object": "list",
            "data": [
                _make_openai_skill_payload(id="skill_first"),
                _make_openai_skill_payload(id="skill_last"),
            ],
            "has_more": True,
            "first_id": "skill_first",
            "last_id": "skill_last",
        }
        raw = _make_mock_response(payload, method="GET")

        result = self.config.transform_list_skills_response(
            raw_response=raw, logging_obj=self.logging_obj
        )

        assert isinstance(result, ListSkillsResponse)
        assert [skill.id for skill in result.data] == ["skill_first", "skill_last"]
        assert result.has_more is True
        assert result.next_page == "skill_last"

    def test_delete_skill_response_maps_deleted_object(self):
        raw = _make_mock_response(
            {"id": "skill_abc123", "object": "skill.deleted", "deleted": True},
            method="DELETE",
        )

        result = self.config.transform_delete_skill_response(
            raw_response=raw, logging_obj=self.logging_obj
        )

        assert isinstance(result, DeleteSkillResponse)
        assert result.id == "skill_abc123"
        assert result.type == "skill_deleted"

    def test_delete_skill_response_handles_canonical_type(self):
        raw = _make_mock_response(
            {"id": "skill_abc123", "type": "skill_deleted"},
            method="DELETE",
        )

        result = self.config.transform_delete_skill_response(
            raw_response=raw, logging_obj=self.logging_obj
        )

        assert result.type == "skill_deleted"
