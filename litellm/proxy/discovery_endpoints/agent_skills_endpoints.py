"""Serve skills stored on the proxy as an Agent Skills well-known discovery index.

``npx skills add <proxy url> -a <agent>`` reads ``/.well-known/agent-skills/index.json``
and downloads each entry's archive. Discovery clients send no credentials, so both
routes are unauthenticated and stay off until ``litellm_settings.public_skills_index``
is enabled, which publishes every stored skill to anyone who can reach the proxy.
"""

import re
from collections.abc import Sequence
from itertools import groupby
from operator import itemgetter
from types import MappingProxyType
from typing import Final

from fastapi import APIRouter, Depends, HTTPException, Request, Response

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.models.skills import LiteLLM_SkillsTable
from litellm.proxy.discovery_endpoints.agent_skills_archive import SkillArchive, build_skill_archive
from litellm.types.proxy.discovery_endpoints.agent_skills_endpoints import (
    MAX_SKILL_DESCRIPTION_LENGTH,
    MAX_SKILL_NAME_LENGTH,
    AgentSkillsIndex,
    AgentSkillsIndexEntry,
)

MAX_INDEXED_SKILLS: Final = 1000

_NON_SLUG_PATTERN: Final = re.compile(r"[^a-z0-9]+")
_FALLBACK_SKILL_NAME: Final = "skill"

router: Final = APIRouter(tags=["public", "skills"])  # mutable-ok: fastapi types tags as list[str | Enum]


def ensure_index_enabled() -> None:
    if litellm.public_skills_index is not True:
        raise HTTPException(status_code=404, detail="Not Found")


async def stored_skills() -> Sequence[LiteLLM_SkillsTable]:
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    return await LiteLLMSkillsHandler.list_skills(limit=MAX_INDEXED_SKILLS)


async def stored_skill(skill_id: str) -> LiteLLM_SkillsTable | None:
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    try:
        return await LiteLLMSkillsHandler.get_skill(skill_id)
    except ValueError:
        return None


@router.get(
    "/.well-known/agent-skills/index.json",
    response_model=AgentSkillsIndex,
    dependencies=(Depends(ensure_index_enabled),),
)
@router.get(
    "/.well-known/skills/index.json",
    response_model=AgentSkillsIndex,
    dependencies=(Depends(ensure_index_enabled),),
    include_in_schema=False,
)
async def agent_skills_index(
    request: Request,
    skills: Sequence[LiteLLM_SkillsTable] = Depends(stored_skills),
) -> AgentSkillsIndex:
    """Agent Skills v0.2.0 discovery index over every skill stored on this proxy."""
    from litellm.proxy.utils import get_custom_url

    installable: Final = tuple(
        (skill, archive)
        for skill, archive in ((skill, _archive_for(skill)) for skill in reversed(skills))
        if archive is not None
    )
    names: Final = _deduplicated(tuple(_base_name(skill, archive) for skill, archive in installable))

    return AgentSkillsIndex(
        skills=tuple(
            AgentSkillsIndexEntry(
                name=name,
                type="archive",
                description=_description(skill, archive, name),
                url=get_custom_url(
                    request_base_url=str(request.base_url),
                    route=f"v1/skills/{skill.skill_id}/archive",
                ),
                digest=archive.digest,
            )
            for (skill, archive), name in zip(installable, names, strict=True)
        )
    )


@router.get(
    "/v1/skills/{skill_id}/archive",
    dependencies=(Depends(ensure_index_enabled),),
)
async def agent_skills_archive(
    skill_id: str,
    skill: LiteLLM_SkillsTable | None = Depends(stored_skill),
) -> Response:
    """Stored skill upload, repacked so SKILL.md sits at the archive root."""
    archive: Final = _archive_for(skill) if skill is not None else None
    if archive is None:
        raise HTTPException(status_code=404, detail=f"No installable skill archive for: {skill_id}")

    return Response(
        content=archive.content,
        media_type="application/zip",
        headers=MappingProxyType({"Content-Disposition": f'attachment; filename="{skill_id}.zip"'}),
    )


def _archive_for(skill: LiteLLM_SkillsTable) -> SkillArchive | None:
    if skill.file_content is None:
        return None

    archive: Final = build_skill_archive(skill.file_content)
    if archive is None:
        verbose_proxy_logger.warning(
            "Agent Skills index: skipping skill %s, its upload is not a zip holding SKILL.md at the root of a "
            "single top-level folder",
            skill.skill_id,
        )
    return archive


def _base_name(skill: LiteLLM_SkillsTable, archive: SkillArchive) -> str:
    candidates: Final = (archive.declared_name, skill.display_title, skill.skill_id)
    return next(
        (slug for slug in (_slugify(candidate) for candidate in candidates) if slug is not None),
        _FALLBACK_SKILL_NAME,
    )


def _slugify(raw: str | None) -> str | None:
    if raw is None:
        return None
    return _NON_SLUG_PATTERN.sub("-", raw.lower()).strip("-")[:MAX_SKILL_NAME_LENGTH].rstrip("-") or None


def _deduplicated(names: Sequence[str]) -> tuple[str, ...]:
    ordinals: Final = MappingProxyType(
        {
            position: ordinal
            for _, duplicates in groupby(sorted(enumerate(names), key=itemgetter(1)), key=itemgetter(1))
            for ordinal, (position, _) in enumerate(duplicates)
        }
    )
    return tuple(_with_ordinal(name, ordinals[position]) for position, name in enumerate(names))


def _with_ordinal(name: str, ordinal: int) -> str:
    if ordinal == 0:
        return name
    suffix: Final = f"-{ordinal + 1}"
    return f"{name[: MAX_SKILL_NAME_LENGTH - len(suffix)].rstrip('-')}{suffix}"


def _description(skill: LiteLLM_SkillsTable, archive: SkillArchive, name: str) -> str:
    candidates: Final = (archive.declared_description, skill.description, skill.display_title)
    chosen: Final = next(
        (candidate.strip() for candidate in candidates if candidate is not None and candidate.strip()),
        name,
    )
    return chosen[:MAX_SKILL_DESCRIPTION_LENGTH]
