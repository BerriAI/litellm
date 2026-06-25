import re

from fastapi import Depends, FastAPI, HTTPException, Request, Response

from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler
from litellm.llms.litellm_proxy.skills.prompt_injection import (
    SkillPromptInjectionHandler,
)
from litellm.proxy._types import LiteLLM_SkillsTable, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

OPENCODE_SKILLS_DEFAULT_PATH = "/opencode/skills"
OPENCODE_REMOTE_CONFIG_DEFAULT_PATH = "/.well-known/opencode"
OPENCODE_CONFIG_SCHEMA = "https://opencode.ai/config.json"
AGENT_SKILLS_PATHS = ("/.well-known/agent-skills", "/.well-known/skills")
_MAX_SKILLS = 1000


def _opencode_config_enabled(skills_gateway_config: dict | None) -> bool:
    if not isinstance(skills_gateway_config, dict):
        return False
    opencode_config = skills_gateway_config.get("opencode", {})
    return (
        skills_gateway_config.get("enabled") is True
        and isinstance(opencode_config, dict)
        and opencode_config.get("enabled") is True
    )


def _opencode_remote_config_enabled(skills_gateway_config: dict | None) -> bool:
    if not _opencode_config_enabled(skills_gateway_config):
        return False
    opencode_config = skills_gateway_config.get("opencode", {})
    remote_config = opencode_config.get("remote_config", {})
    return isinstance(remote_config, dict) and remote_config.get("enabled") is True


def _agent_skills_config_enabled(skills_gateway_config: dict | None) -> bool:
    if not isinstance(skills_gateway_config, dict):
        return False
    agent_skills_config = skills_gateway_config.get("agent_skills", {})
    return (
        skills_gateway_config.get("enabled") is True
        and isinstance(agent_skills_config, dict)
        and agent_skills_config.get("enabled") is True
    )


def _normalized_route_path(path: object, default: str) -> str:
    path = str(path or default).strip() or default
    if not path.startswith("/"):
        path = f"/{path}"
    return path.rstrip("/") or default


def _opencode_path(skills_gateway_config: dict | None) -> str:
    if not isinstance(skills_gateway_config, dict):
        return OPENCODE_SKILLS_DEFAULT_PATH
    opencode_config = skills_gateway_config.get("opencode", {})
    if not isinstance(opencode_config, dict):
        return OPENCODE_SKILLS_DEFAULT_PATH
    return _normalized_route_path(
        opencode_config.get("path"),
        OPENCODE_SKILLS_DEFAULT_PATH,
    )


def _opencode_remote_config_path(skills_gateway_config: dict | None) -> str:
    if not isinstance(skills_gateway_config, dict):
        return OPENCODE_REMOTE_CONFIG_DEFAULT_PATH
    opencode_config = skills_gateway_config.get("opencode", {})
    if not isinstance(opencode_config, dict):
        return OPENCODE_REMOTE_CONFIG_DEFAULT_PATH
    remote_config = opencode_config.get("remote_config", {})
    if not isinstance(remote_config, dict):
        return OPENCODE_REMOTE_CONFIG_DEFAULT_PATH
    return _normalized_route_path(
        remote_config.get("path"),
        OPENCODE_REMOTE_CONFIG_DEFAULT_PATH,
    )


def _skill_enabled(skill: LiteLLM_SkillsTable) -> bool:
    metadata = skill.metadata if isinstance(skill.metadata, dict) else {}
    return metadata.get("enabled") is not False


def _opencode_skill_name(skill: LiteLLM_SkillsTable) -> str:
    return skill.skill_id


def _agent_skill_name(skill: LiteLLM_SkillsTable) -> str:
    return _slug(skill.skill_id)[:64].strip("-") or "litellm-skill"


def _agent_skill_description(skill: LiteLLM_SkillsTable) -> str:
    return _one_line(
        skill.description or skill.instructions or skill.display_title or skill.skill_id
    )[:1024]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "litellm-skill"


def _one_line(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).split())


def _generated_skill_md(skill: LiteLLM_SkillsTable) -> bytes:
    title = skill.display_title or skill.skill_id
    description = _one_line(skill.description or skill.instructions or title)
    instructions = (skill.instructions or skill.description or title).strip()
    return (
        "---\n"
        f"name: {_slug(skill.skill_id)}\n"
        f"description: {description}\n"
        "---\n\n"
        f"# {title}\n\n"
        f"{instructions}\n"
    ).encode("utf-8")


def _skill_files(skill: LiteLLM_SkillsTable) -> dict[str, bytes]:
    files = SkillPromptInjectionHandler().extract_all_files(skill)
    if "SKILL.md" not in files:
        files["SKILL.md"] = _generated_skill_md(skill)
    return files


async def _enabled_skills(
    user_api_key_dict: UserAPIKeyAuth,
) -> list[LiteLLM_SkillsTable]:
    skills = await LiteLLMSkillsHandler.list_skills(
        limit=_MAX_SKILLS,
        offset=0,
        user_api_key_dict=user_api_key_dict,
    )
    return [skill for skill in skills if _skill_enabled(skill)]


def _sorted_files(files: dict[str, bytes]) -> list[str]:
    return sorted(files)


async def opencode_skills_index(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    skills = await _enabled_skills(user_api_key_dict)
    return {
        "skills": [
            {
                "name": _opencode_skill_name(skill),
                "files": _sorted_files(_skill_files(skill)),
            }
            for skill in skills
        ]
    }


def _opencode_remote_config_response(
    request: Request,
    skills_gateway_config: dict | None,
) -> dict:
    from litellm.proxy.utils import get_custom_url

    skills_url = get_custom_url(
        request_base_url=str(request.base_url),
        route=_opencode_path(skills_gateway_config),
    )
    return {
        "config": {
            "$schema": OPENCODE_CONFIG_SCHEMA,
            "skills": {"urls": [skills_url]},
        }
    }


async def opencode_skill_file(
    skill_name: str,
    file_path: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    for skill in await _enabled_skills(user_api_key_dict):
        if _opencode_skill_name(skill) != skill_name:
            continue
        files = _skill_files(skill)
        content = files.get(file_path)
        if content is None:
            break
        media_type = (
            "text/markdown; charset=utf-8"
            if file_path.endswith(".md")
            else "application/octet-stream"
        )
        return Response(content=content, media_type=media_type)
    raise HTTPException(status_code=404, detail="Skill file not found")


async def agent_skills_index(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    skills = await _enabled_skills(user_api_key_dict)
    return {
        "skills": [
            {
                "name": _agent_skill_name(skill),
                "description": _agent_skill_description(skill),
                "files": _sorted_files(_skill_files(skill)),
            }
            for skill in skills
        ]
    }


async def agent_skill_file(
    skill_name: str,
    file_path: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    for skill in await _enabled_skills(user_api_key_dict):
        if _agent_skill_name(skill) != skill_name:
            continue
        files = _skill_files(skill)
        content = files.get(file_path)
        if content is None:
            break
        media_type = (
            "text/markdown; charset=utf-8"
            if file_path.endswith(".md")
            else "application/octet-stream"
        )
        return Response(content=content, media_type=media_type)
    raise HTTPException(status_code=404, detail="Skill file not found")


def _add_route(app: FastAPI, path: str, endpoint):
    for route in app.routes:
        if getattr(route, "path", None) == path and "GET" in getattr(
            route, "methods", set()
        ):
            return
    app.add_api_route(path=path, endpoint=endpoint, methods=["GET"])


def initialize_opencode_skills_endpoint(
    app: FastAPI,
    skills_gateway_config: dict | None,
) -> None:
    if not _opencode_config_enabled(skills_gateway_config):
        return

    path = _opencode_path(skills_gateway_config)
    _add_route(app, path, opencode_skills_index)
    _add_route(app, f"{path}/index.json", opencode_skills_index)
    _add_route(app, f"{path}/{{skill_name}}/{{file_path:path}}", opencode_skill_file)


def initialize_opencode_remote_config_endpoint(
    app: FastAPI,
    skills_gateway_config: dict | None,
) -> None:
    if not _opencode_remote_config_enabled(skills_gateway_config):
        return

    async def opencode_remote_config(request: Request):
        return _opencode_remote_config_response(
            request=request,
            skills_gateway_config=skills_gateway_config,
        )

    _add_route(
        app,
        _opencode_remote_config_path(skills_gateway_config),
        opencode_remote_config,
    )


def initialize_agent_skills_endpoint(
    app: FastAPI,
    skills_gateway_config: dict | None,
) -> None:
    if not _agent_skills_config_enabled(skills_gateway_config):
        return

    for path in AGENT_SKILLS_PATHS:
        _add_route(app, f"{path}/index.json", agent_skills_index)
        _add_route(app, f"{path}/{{skill_name}}/{{file_path:path}}", agent_skill_file)
