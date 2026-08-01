"""Offline tests for native OpenAI and Azure OpenAI Skills transformations."""

from unittest.mock import MagicMock

import httpx

from litellm.llms.azure.skills.transformation import AzureOpenAISkillsConfig
from litellm.llms.custom_httpx import llm_http_handler as llm_http_handler_module
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.skills.transformation import OpenAISkillsConfig
from litellm.types.router import GenericLiteLLMParams


def test_openai_skill_urls_and_query_params():
    config = OpenAISkillsConfig()
    params = GenericLiteLLMParams(api_base="https://api.openai.test/v1", api_key="test-key")

    assert (
        config.get_complete_url(params.api_base, "skills", litellm_params=params) == "https://api.openai.test/v1/skills"
    )
    assert config.get_skill_operation_url("version_content", "skill_1", "v2", params) == (
        "https://api.openai.test/v1/skills/skill_1/versions/v2/content"
    )

    response = httpx.Response(
        200,
        json={"data": [], "first_id": None, "has_more": False, "last_id": None, "object": "list"},
        request=httpx.Request("GET", "https://api.openai.test/v1/skills"),
    )
    transformed = config.transform_list_skills_response(response, None)
    assert transformed.object == "list"


def test_openai_skill_auth_uses_litellm_params():
    headers = OpenAISkillsConfig().validate_environment(
        headers={},
        litellm_params=GenericLiteLLMParams(api_key="test-key"),
    )
    assert headers["Authorization"] == "Bearer test-key"


def test_azure_skill_url_and_auth_reuse_existing_azure_configuration():
    config = AzureOpenAISkillsConfig()
    params = GenericLiteLLMParams(
        api_base="https://resource.openai.azure.com",
        api_version="v1",
        api_key="test-key",
    )

    assert config.get_complete_url(None, "skills", litellm_params=params) == (
        "https://resource.openai.azure.com/openai/v1/skills?api-version=v1"
    )
    headers = config.validate_environment({}, params)
    assert headers["api-key"] == "test-key"


def test_skill_operation_handler_builds_json_request_without_network(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.kwargs = None

        def post(self, **kwargs):
            self.kwargs = kwargs
            return httpx.Response(
                200,
                json={"id": "skill_1", "created_at": 1, "object": "skill"},
                request=httpx.Request("POST", "https://api.openai.test/v1/skills/skill_1"),
            )

    fake_client = FakeClient()
    monkeypatch.setattr(llm_http_handler_module, "_get_httpx_client", lambda **_: fake_client)
    config = OpenAISkillsConfig()
    result = BaseLLMHTTPHandler().skill_operation_handler(
        method="POST",
        operation="update",
        url="https://api.openai.test/v1/skills/skill_1",
        skills_api_provider_config=config,
        custom_llm_provider="openai",
        litellm_params=GenericLiteLLMParams(api_key="test-key"),
        logging_obj=MagicMock(),
        request_body={"default_version": "v2"},
    )

    assert result.id == "skill_1"
    assert fake_client.kwargs["json"] == {"default_version": "v2"}
