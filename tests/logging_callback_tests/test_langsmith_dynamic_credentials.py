import pytest

from litellm.integrations.langsmith import LangsmithLogger


@pytest.mark.asyncio
async def test_get_credentials_from_env_does_not_use_env_for_dynamic_base_url(
    monkeypatch,
):
    monkeypatch.setenv("LANGSMITH_API_KEY", "global-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "global-project")
    monkeypatch.setenv("LANGSMITH_TENANT_ID", "global-tenant")
    logger = LangsmithLogger(
        langsmith_api_key="default-key",
        langsmith_project="default-project",
        langsmith_base_url="https://default.example",
    )

    credentials = logger.get_credentials_from_env(
        langsmith_base_url="https://attacker.example",
        allow_env_credentials=False,
    )

    assert credentials["LANGSMITH_API_KEY"] is None
    assert credentials["LANGSMITH_PROJECT"] == "litellm-completion"
    assert credentials["LANGSMITH_BASE_URL"] == "https://attacker.example"
    assert credentials["LANGSMITH_TENANT_ID"] is None


@pytest.mark.asyncio
async def test_dynamic_langsmith_base_url_does_not_inherit_default_api_key(
    monkeypatch,
):
    monkeypatch.setenv("LANGSMITH_API_KEY", "global-key")
    logger = LangsmithLogger(
        langsmith_api_key="default-key",
        langsmith_project="default-project",
        langsmith_base_url="https://default.example",
    )

    credentials = logger._get_credentials_to_use_for_request(
        kwargs={
            "standard_callback_dynamic_params": {
                "langsmith_base_url": "https://attacker.example"
            }
        }
    )

    assert credentials["LANGSMITH_API_KEY"] is None
    assert credentials["LANGSMITH_BASE_URL"] == "https://attacker.example"
