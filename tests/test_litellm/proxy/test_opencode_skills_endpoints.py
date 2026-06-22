from unittest.mock import AsyncMock
from io import BytesIO
from zipfile import ZipFile

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LiteLLM_SkillsTable, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


def _client(config: dict, auth: UserAPIKeyAuth | None = None) -> TestClient:
    from litellm.proxy.opencode_endpoints.skills_endpoints import (
        initialize_opencode_skills_endpoint,
    )

    app = FastAPI()
    if auth is not None:
        app.dependency_overrides[user_api_key_auth] = lambda: auth
    initialize_opencode_skills_endpoint(app=app, skills_gateway_config=config)
    return TestClient(app)


def test_should_not_register_opencode_skills_endpoint_when_disabled():
    client = _client({})

    response = client.get("/opencode/skills")

    assert response.status_code == 404


def test_should_return_opencode_skills_index_when_enabled(monkeypatch):
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    list_skills = AsyncMock(
        return_value=[
            LiteLLM_SkillsTable(
                skill_id="litellm_skill_writer",
                display_title="Writer",
                description="Draft clean release notes",
                instructions="Use this skill to draft release notes.",
            )
        ]
    )
    monkeypatch.setattr(LiteLLMSkillsHandler, "list_skills", list_skills)
    auth = UserAPIKeyAuth(user_id="user-1")
    client = _client(
        {"enabled": True, "opencode": {"enabled": True}},
        auth=auth,
    )

    response = client.get("/opencode/skills/index.json")

    assert response.status_code == 200
    assert response.json() == {
        "skills": [{"name": "litellm_skill_writer", "files": ["SKILL.md"]}]
    }
    list_skills.assert_awaited_once_with(limit=1000, offset=0, user_api_key_dict=auth)


def test_should_omit_disabled_litellm_skills(monkeypatch):
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "list_skills",
        AsyncMock(
            return_value=[
                LiteLLM_SkillsTable(
                    skill_id="litellm_skill_enabled",
                    metadata={"enabled": True},
                ),
                LiteLLM_SkillsTable(
                    skill_id="litellm_skill_disabled",
                    metadata={"enabled": False},
                ),
            ]
        ),
    )
    client = _client(
        {"enabled": True, "opencode": {"enabled": True}},
        auth=UserAPIKeyAuth(user_id="user-1"),
    )

    response = client.get("/opencode/skills")

    assert response.status_code == 200
    assert response.json() == {
        "skills": [{"name": "litellm_skill_enabled", "files": ["SKILL.md"]}]
    }


def test_should_serve_generated_skill_markdown(monkeypatch):
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "list_skills",
        AsyncMock(
            return_value=[
                LiteLLM_SkillsTable(
                    skill_id="litellm_skill_writer",
                    display_title="Writer",
                    description="Draft clean release notes",
                    instructions="Use this skill to draft release notes.",
                )
            ]
        ),
    )
    client = _client(
        {"enabled": True, "opencode": {"enabled": True}},
        auth=UserAPIKeyAuth(user_id="user-1"),
    )

    response = client.get("/opencode/skills/litellm_skill_writer/SKILL.md")

    assert response.status_code == 200
    assert response.text == (
        "---\n"
        "name: litellm-skill-writer\n"
        "description: Draft clean release notes\n"
        "---\n\n"
        "# Writer\n\n"
        "Use this skill to draft release notes.\n"
    )


def test_should_serve_zip_backed_skill_files_from_custom_path(monkeypatch):
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr(
            "writer/SKILL.md",
            "---\nname: writer\ndescription: Draft notes\n---\n\nUse me.",
        )
        zip_file.writestr("writer/references/example.txt", "example")

    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "list_skills",
        AsyncMock(
            return_value=[
                LiteLLM_SkillsTable(
                    skill_id="litellm_skill_writer",
                    file_content=zip_buffer.getvalue(),
                    file_name="writer.zip",
                )
            ]
        ),
    )
    client = _client(
        {
            "enabled": True,
            "opencode": {"enabled": True, "path": "/custom/skills"},
        },
        auth=UserAPIKeyAuth(user_id="user-1"),
    )

    index = client.get("/custom/skills/index.json")
    skill_md = client.get("/custom/skills/litellm_skill_writer/SKILL.md")
    reference = client.get("/custom/skills/litellm_skill_writer/references/example.txt")

    assert index.status_code == 200
    assert index.json() == {
        "skills": [
            {
                "name": "litellm_skill_writer",
                "files": ["SKILL.md", "references/example.txt"],
            }
        ]
    }
    assert skill_md.status_code == 200
    assert (
        skill_md.text == "---\nname: writer\ndescription: Draft notes\n---\n\nUse me."
    )
    assert reference.status_code == 200
    assert reference.text == "example"
