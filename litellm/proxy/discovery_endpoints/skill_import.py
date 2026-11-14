import io
import re
import zipfile
from dataclasses import dataclass
from typing import Final
from urllib.parse import quote, unquote, urlsplit

import httpx

from litellm.proxy.discovery_endpoints.agent_skills_archive import (
    MAX_ARCHIVE_ENTRIES,
    MAX_ARCHIVE_UNPACKED_BYTES,
    MAX_SKILL_MANIFEST_BYTES,
    SKILL_ARCHIVE_ERRORS,
    SkillArchive,
    agent_skill_frontmatter,
    build_agent_skill_archive,
    read_skill_archive_entry,
    safe_skill_archive_entry,
    skill_archive_entries,
)

MAX_IMPORT_BYTES: Final = 10 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SkillImportError:
    status: int
    message: str


@dataclass(frozen=True, slots=True)
class GitHubSkillSource:
    owner: str
    repository: str
    ref: str
    skill: str | None
    path: str | None

    @property
    def archive_url(self) -> str:
        return f"https://codeload.github.com/{self.owner}/{self.repository}/zip/{quote(self.ref, safe='')}"


def parse_skill_source(
    url: str, *, skill: str | None, ref: str, path: str | None
) -> GitHubSkillSource | SkillImportError:
    try:
        parsed: Final = urlsplit(url)
    except ValueError:
        return SkillImportError(400, "Invalid skill source URL")
    if parsed.scheme != "https" or parsed.netloc not in ("skills.sh", "github.com") or parsed.query or parsed.fragment:
        return SkillImportError(400, "Use an HTTPS skills.sh skill URL or a public github.com repository URL")
    parts: Final = tuple(unquote(part) for part in parsed.path.strip("/").split("/"))
    if len(parts) < 2 or any(re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None for part in parts[:2]):
        return SkillImportError(400, "The source URL must identify a GitHub owner and repository")
    if parsed.netloc == "skills.sh":
        if len(parts) != 3 or (skill is not None and skill != parts[2]):
            return SkillImportError(400, "Use skills.sh/<owner>/<repository>/<skill> with a matching skill selection")
        return GitHubSkillSource(parts[0], parts[1], ref, parts[2], path)
    if len(parts) == 2:
        return GitHubSkillSource(parts[0], parts[1].removesuffix(".git"), ref, skill, path)
    if len(parts) < 5 or parts[2] != "tree" or path is not None or ref != "HEAD":
        return SkillImportError(400, "Use a repository URL with ref and path, or a GitHub tree URL")
    return GitHubSkillSource(parts[0], parts[1], parts[3], skill, "/".join(parts[4:]))


async def fetch_skill_source(source: GitHubSkillSource, client: httpx.AsyncClient) -> bytes | SkillImportError:
    buffer: Final = io.BytesIO()
    request: Final = httpx.Request(
        "GET",
        source.archive_url,
        headers={"Accept": "application/zip"},
        extensions={"timeout": httpx.Timeout(30).as_dict()},
    )
    try:
        response: Final = await client.send(request, stream=True, follow_redirects=False, auth=None)
        try:
            if response.status_code != 200:
                return SkillImportError(422, "Unable to download the public GitHub repository at the requested ref")
            async for chunk in response.aiter_bytes():
                if buffer.tell() + len(chunk) > MAX_IMPORT_BYTES:
                    return SkillImportError(413, "Repository ZIP must be 10 MiB or smaller")
                buffer.write(chunk)
        finally:
            await response.aclose()
    except httpx.HTTPError:
        return SkillImportError(502, "The skill source could not be reached")
    return buffer.getvalue()


def select_imported_skill(content: bytes, source: GitHubSkillSource) -> SkillArchive | SkillImportError:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos: Final = archive.infolist()
            if len(infos) > MAX_ARCHIVE_ENTRIES or sum(info.file_size for info in infos) > MAX_ARCHIVE_UNPACKED_BYTES:
                return SkillImportError(413, "Repository exceeds the skill archive limits")
            entries: Final = skill_archive_entries(archive, for_upload=True)
            if entries is None:
                return SkillImportError(422, "Repository contains unsafe archive entries")
            candidates: Final = tuple(info for info, path in entries if path.endswith("/SKILL.md"))
            if any(not safe_skill_archive_entry(info) for info in candidates):
                return SkillImportError(422, "Repository contains unsafe skill manifests")
            selected: Final = tuple(info for info in candidates if _matches_selection(info, archive, source))
            if len(selected) != 1:
                return SkillImportError(422, "Select exactly one skill using its name or repository-relative path")
            root: Final = selected[0].filename.rsplit("/", 1)[0]
            skill: Final = build_agent_skill_archive(content, skill_path=root)
            return (
                skill
                if skill is not None
                else SkillImportError(422, "The selected directory is not a valid Agent Skill")
            )
    except SKILL_ARCHIVE_ERRORS:
        return SkillImportError(422, "The repository did not return a readable skill ZIP")


def _matches_selection(info: zipfile.ZipInfo, archive: zipfile.ZipFile, source: GitHubSkillSource) -> bool:
    if info.file_size > MAX_SKILL_MANIFEST_BYTES:
        return False
    relative: Final = info.filename.split("/", 1)[1].rpartition("/")[0]
    if source.path is not None and relative != source.path.strip("/"):
        return False
    if source.skill is None:
        return True
    return agent_skill_frontmatter(read_skill_archive_entry(archive, info)).get("name") == source.skill
