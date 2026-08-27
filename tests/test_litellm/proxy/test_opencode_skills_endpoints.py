from io import BytesIO
from unittest.mock import AsyncMock
from zipfile import ZipFile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LiteLLM_SkillsTable, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


def _client(config: dict, auth: UserAPIKeyAuth | None = None) -> TestClient:
    from litellm.proxy.opencode_endpoints.skills_endpoints import (
        initialize_agent_skills_endpoint,
        initialize_opencode_remote_config_endpoint,
        initialize_opencode_skills_endpoint,
    )

    app = FastAPI()
    if auth is not None:
        app.dependency_overrides[user_api_key_auth] = lambda: auth
    initialize_opencode_skills_endpoint(app=app, skills_gateway_config=config)
    initialize_opencode_remote_config_endpoint(app=app, skills_gateway_config=config)
    initialize_agent_skills_endpoint(app=app, skills_gateway_config=config)
    return TestClient(app)


def test_should_not_register_opencode_skills_endpoint_when_disabled():
    client = _client({})

    response = client.get("/opencode/skills")

    assert response.status_code == 404


def test_should_not_register_opencode_remote_config_when_disabled():
    client = _client({"enabled": True, "opencode": {"enabled": True}})

    response = client.get("/.well-known/opencode")

    assert response.status_code == 404


def test_should_not_register_agent_skills_endpoint_when_disabled():
    client = _client({"enabled": True})

    response = client.get("/.well-known/agent-skills/index.json")

    assert response.status_code == 404


def test_should_handle_skills_endpoint_config_edge_cases():
    from litellm.proxy.opencode_endpoints.skills_endpoints import (
        OPENCODE_SKILLS_DEFAULT_PATH,
        _agent_skills_config_enabled,
        _one_line,
        _opencode_config_enabled,
        _opencode_path,
    )

    assert _opencode_config_enabled(None) is False
    assert _agent_skills_config_enabled(None) is False
    assert _opencode_path(None) == OPENCODE_SKILLS_DEFAULT_PATH
    assert _opencode_path({"enabled": True, "opencode": "invalid"}) == (
        OPENCODE_SKILLS_DEFAULT_PATH
    )
    assert _one_line(None) == ""


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


def test_should_return_opencode_remote_config_when_enabled_without_auth(monkeypatch):
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    monkeypatch.delenv("SERVER_ROOT_PATH", raising=False)
    client = _client(
        {
            "enabled": True,
            "opencode": {
                "enabled": True,
                "remote_config": {"enabled": True},
            },
        }
    )

    response = client.get("/.well-known/opencode")

    assert response.status_code == 200
    assert response.json() == {
        "config": {
            "$schema": "https://opencode.ai/config.json",
            "skills": {
                "urls": ["http://testserver/opencode/skills"],
            },
        },
    }


def test_should_return_opencode_remote_config_with_server_root_path(monkeypatch):
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    monkeypatch.setenv("SERVER_ROOT_PATH", "/my-custom-path")
    client = _client(
        {
            "enabled": True,
            "opencode": {
                "enabled": True,
                "remote_config": {"enabled": True},
            },
        }
    )

    response = client.get("/.well-known/opencode")

    assert response.status_code == 200
    assert response.json()["config"]["skills"]["urls"] == [
        "http://testserver/my-custom-path/opencode/skills"
    ]


def test_should_normalize_opencode_path_without_leading_slash(monkeypatch):
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "list_skills",
        AsyncMock(return_value=[]),
    )
    client = _client(
        {"enabled": True, "opencode": {"enabled": True, "path": "custom/skills"}},
        auth=UserAPIKeyAuth(user_id="user-1"),
    )

    response = client.get("/custom/skills/index.json")

    assert response.status_code == 200
    assert response.json() == {"skills": []}


def test_should_return_opencode_remote_config_with_custom_paths_and_proxy_base_url(
    monkeypatch,
):
    monkeypatch.setenv("PROXY_BASE_URL", "https://litellm.example.com/root")
    monkeypatch.delenv("SERVER_ROOT_PATH", raising=False)
    client = _client(
        {
            "enabled": True,
            "opencode": {
                "enabled": True,
                "path": "native/skills",
                "remote_config": {
                    "enabled": True,
                    "path": "/custom/opencode",
                },
            },
        }
    )

    response = client.get("/custom/opencode")

    assert response.status_code == 200
    assert response.json()["config"]["skills"]["urls"] == [
        "https://litellm.example.com/root/native/skills"
    ]


def test_should_return_agent_skills_well_known_index_when_enabled(monkeypatch):
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
        {"enabled": True, "agent_skills": {"enabled": True}},
        auth=auth,
    )

    response = client.get("/.well-known/agent-skills/index.json")

    assert response.status_code == 200
    assert response.json() == {
        "skills": [
            {
                "name": "litellm-skill-writer",
                "description": "Draft clean release notes",
                "files": ["SKILL.md"],
            }
        ]
    }
    list_skills.assert_awaited_once_with(limit=1000, offset=0, user_api_key_dict=auth)


def test_should_serve_agent_skills_markdown_from_well_known_path(monkeypatch):
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
        {"enabled": True, "agent_skills": {"enabled": True}},
        auth=UserAPIKeyAuth(user_id="user-1"),
    )

    response = client.get("/.well-known/agent-skills/litellm-skill-writer/SKILL.md")

    assert response.status_code == 200
    assert response.text == (
        "---\n"
        "name: litellm-skill-writer\n"
        "description: Draft clean release notes\n"
        "---\n\n"
        "# Writer\n\n"
        "Use this skill to draft release notes.\n"
    )


def test_should_return_opencode_404_when_skill_file_is_missing(monkeypatch):
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "list_skills",
        AsyncMock(
            return_value=[
                LiteLLM_SkillsTable(skill_id="other_skill"),
                LiteLLM_SkillsTable(skill_id="litellm_skill_writer"),
            ]
        ),
    )
    client = _client(
        {"enabled": True, "opencode": {"enabled": True}},
        auth=UserAPIKeyAuth(user_id="user-1"),
    )

    response = client.get("/opencode/skills/litellm_skill_writer/missing.txt")

    assert response.status_code == 404
    assert response.json() == {"detail": "Skill file not found"}


def test_should_return_opencode_404_when_no_skills_exist(monkeypatch):
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "list_skills",
        AsyncMock(return_value=[]),
    )
    client = _client(
        {"enabled": True, "opencode": {"enabled": True}},
        auth=UserAPIKeyAuth(user_id="user-1"),
    )

    response = client.get("/opencode/skills/missing_skill/SKILL.md")

    assert response.status_code == 404
    assert response.json() == {"detail": "Skill file not found"}


def test_should_return_agent_404_when_skill_file_is_missing(monkeypatch):
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "list_skills",
        AsyncMock(
            return_value=[
                LiteLLM_SkillsTable(skill_id="other_skill"),
                LiteLLM_SkillsTable(skill_id="litellm_skill_writer"),
            ]
        ),
    )
    client = _client(
        {"enabled": True, "agent_skills": {"enabled": True}},
        auth=UserAPIKeyAuth(user_id="user-1"),
    )

    response = client.get("/.well-known/agent-skills/litellm-skill-writer/missing.txt")

    assert response.status_code == 404
    assert response.json() == {"detail": "Skill file not found"}


def test_should_return_agent_404_when_no_skills_exist(monkeypatch):
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    monkeypatch.setattr(
        LiteLLMSkillsHandler,
        "list_skills",
        AsyncMock(return_value=[]),
    )
    client = _client(
        {"enabled": True, "agent_skills": {"enabled": True}},
        auth=UserAPIKeyAuth(user_id="user-1"),
    )

    response = client.get("/.well-known/agent-skills/missing-skill/SKILL.md")

    assert response.status_code == 404
    assert response.json() == {"detail": "Skill file not found"}


def test_should_not_duplicate_routes_when_initialized_twice():
    from litellm.proxy.opencode_endpoints.skills_endpoints import (
        initialize_opencode_remote_config_endpoint,
        initialize_opencode_skills_endpoint,
    )

    app = FastAPI()
    config = {
        "enabled": True,
        "opencode": {"enabled": True, "remote_config": {"enabled": True}},
    }

    initialize_opencode_skills_endpoint(app=app, skills_gateway_config=config)
    initialize_opencode_remote_config_endpoint(app=app, skills_gateway_config=config)
    initialize_opencode_skills_endpoint(app=app, skills_gateway_config=config)
    initialize_opencode_remote_config_endpoint(app=app, skills_gateway_config=config)

    route_paths = [getattr(route, "path", None) for route in app.routes]
    assert route_paths.count("/opencode/skills") == 1
    assert route_paths.count("/opencode/skills/index.json") == 1
    assert route_paths.count("/opencode/skills/{skill_name}/{file_path:path}") == 1
    assert route_paths.count("/.well-known/opencode") == 1


def test_should_serve_legacy_agent_skills_well_known_alias(monkeypatch):
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
        {"enabled": True, "agent_skills": {"enabled": True}},
        auth=UserAPIKeyAuth(user_id="user-1"),
    )

    response = client.get("/.well-known/skills/index.json")

    assert response.status_code == 200
    assert response.json() == {
        "skills": [
            {
                "name": "litellm-skill-enabled",
                "description": "litellm_skill_enabled",
                "files": ["SKILL.md"],
            }
        ]
    }


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


@pytest.mark.asyncio
async def test_should_register_skills_endpoints_from_proxy_config_load(
    tmp_path, monkeypatch
):
    from litellm.proxy import proxy_server

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "model_list: []\n"
        "general_settings: {}\n"
        "litellm_settings: {}\n"
        "skills_gateway:\n"
        "  enabled: true\n"
        "  opencode:\n"
        "    enabled: true\n"
        "    path: /native/skills\n"
        "    remote_config:\n"
        "      enabled: true\n"
        "  agent_skills:\n"
        "    enabled: true\n"
    )
    app = FastAPI()
    monkeypatch.setattr(proxy_server, "app", app)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)

    proxy_config = proxy_server.ProxyConfig()
    await proxy_config.load_config(router=None, config_file_path=str(config_file))

    route_paths = {getattr(route, "path", None) for route in app.routes}
    assert "/native/skills" in route_paths
    assert "/native/skills/index.json" in route_paths
    assert "/.well-known/opencode" in route_paths
    assert "/.well-known/agent-skills/index.json" in route_paths
