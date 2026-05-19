"""
xct skill ZIP parsing (S2-07).

A skill ZIP is a small archive with up to four well-known files at the root:

    manifest.yaml             metadata (required)
    SKILL.md                  system prompt template (Jinja2 / format-string)
    tools.json                OpenAI-shape tools array (optional)
    README.md                 free-form description (optional, used as fallback
                              for the description field when manifest lacks one)

Anything else in the archive is preserved verbatim in ``file_content`` so the
operator can re-download the original.

Constraints:
    - Total uncompressed size ≤ MAX_SKILL_ZIP_SIZE (zip-bomb guard)
    - Total archive size ≤ MAX_SKILL_ZIP_SIZE
    - manifest.yaml must parse and carry a non-empty ``display_title``
    - No symlinks, no entries with ``..`` in the path

This module is intentionally transport-agnostic: ``parse_skill_zip(bytes)``
returns a structured ``ParsedSkill`` dataclass that the FastAPI endpoint
then maps onto the Prisma create_data dict. Virus-scan hook is a callable
the caller can pass in (default no-op).
"""

import io
import os
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import yaml

from litellm._logging import verbose_proxy_logger

MAX_SKILL_ZIP_SIZE_BYTES = int(
    os.environ.get(
        "LITELLM_SKILL_UPLOAD_MAX_SIZE_BYTES", str(10 * 1024 * 1024)  # 10 MB
    )
)


class SkillZipError(ValueError):
    """Raised by parse_skill_zip with a human-friendly message."""


@dataclass
class ParsedSkill:
    """Structured skill payload extracted from a ZIP."""

    display_title: str
    description: Optional[str] = None
    instructions: Optional[str] = None
    system_prompt_template: Optional[str] = None
    tool_schema: Optional[List[Dict[str, Any]]] = None
    version: Optional[str] = "1"
    is_public: bool = False
    xct_metadata: Dict[str, Any] = field(default_factory=dict)
    # Raw archive bytes, persisted into LiteLLM_SkillsTable.file_content so
    # operators can re-download or hand off to a tool that expects the
    # original zip.
    file_content: bytes = b""
    file_name: Optional[str] = None
    file_type: str = "application/zip"


VirusScanHook = Callable[[bytes], None]
"""A virus-scan hook is a callable that raises on suspicious content.

Default implementation (``_noop_scan``) does nothing; enterprise can wire
ClamAV, S3 antivirus tags, or whatever else. The hook runs BEFORE parsing
so an infected archive never gets unzipped server-side.
"""


def _noop_scan(_payload: bytes) -> None:  # pragma: no cover — trivial default
    return None


def _safe_read(zf: zipfile.ZipFile, name: str) -> Optional[str]:
    """Read a member as UTF-8 text. Returns None when absent."""
    try:
        info = zf.getinfo(name)
    except KeyError:
        return None
    if info.is_dir():
        return None
    return zf.read(info).decode("utf-8", errors="replace")


def _validate_archive_safety(zf: zipfile.ZipFile, raw_size: int) -> None:
    """Reject obviously hostile archives BEFORE we read any member.

    Guards against:
      - zip-bombs: uncompressed total ≤ MAX_SKILL_ZIP_SIZE_BYTES
      - path traversal: no member name contains ``..`` or absolute paths
      - symlinks: zip can encode symlinks via the external_attr field;
        we reject any non-regular entry
    """
    if raw_size > MAX_SKILL_ZIP_SIZE_BYTES:
        raise SkillZipError(
            f"ZIP too large ({raw_size} > {MAX_SKILL_ZIP_SIZE_BYTES} bytes)."
        )

    total_uncompressed = 0
    for info in zf.infolist():
        # path traversal / absolute path
        if (
            info.filename.startswith("/")
            or "\\" in info.filename
            or ".." in info.filename.split("/")
        ):
            raise SkillZipError(f"Refusing unsafe path in archive: {info.filename!r}")
        # symlink detection (Unix permission bits in external_attr upper 16)
        unix_mode = (info.external_attr >> 16) & 0xF000
        if unix_mode == 0xA000:  # S_IFLNK
            raise SkillZipError(f"Symlinks not allowed in skill ZIP: {info.filename!r}")
        total_uncompressed += info.file_size
        if total_uncompressed > MAX_SKILL_ZIP_SIZE_BYTES:
            raise SkillZipError(
                "Uncompressed contents exceed " f"{MAX_SKILL_ZIP_SIZE_BYTES} bytes."
            )


def parse_skill_zip(
    payload: bytes,
    *,
    file_name: Optional[str] = None,
    virus_scan: Optional[VirusScanHook] = None,
) -> ParsedSkill:
    """Parse a skill ZIP and return a structured ParsedSkill."""
    if not payload:
        raise SkillZipError("Empty file uploaded.")
    (virus_scan or _noop_scan)(payload)

    try:
        zf = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as e:
        raise SkillZipError(f"Not a valid ZIP archive: {e}") from e

    _validate_archive_safety(zf, raw_size=len(payload))

    manifest_raw = _safe_read(zf, "manifest.yaml") or _safe_read(zf, "manifest.yml")
    if not manifest_raw:
        raise SkillZipError("manifest.yaml is required at the archive root.")
    try:
        manifest: Dict[str, Any] = yaml.safe_load(manifest_raw) or {}
    except yaml.YAMLError as e:
        raise SkillZipError(f"manifest.yaml is not valid YAML: {e}") from e
    if not isinstance(manifest, dict):
        raise SkillZipError("manifest.yaml must be a YAML mapping.")

    display_title = str(
        manifest.get("display_title") or manifest.get("name") or ""
    ).strip()
    if not display_title:
        raise SkillZipError(
            "manifest.yaml must include a non-empty 'display_title' (or 'name')."
        )

    description = manifest.get("description")
    if not description:
        # Fall back to README.md if present.
        readme = _safe_read(zf, "README.md")
        if readme:
            # First non-empty line as the description.
            for line in readme.splitlines():
                line = line.strip().lstrip("#").strip()
                if line:
                    description = line
                    break

    system_prompt_template = _safe_read(zf, "SKILL.md") or _safe_read(
        zf, "system_prompt.md"
    )
    instructions = system_prompt_template  # mirror for legacy callers

    tools_raw = _safe_read(zf, "tools.json")
    tool_schema: Optional[List[Dict[str, Any]]] = None
    if tools_raw:
        try:
            import json as _json

            parsed = _json.loads(tools_raw)
        except Exception as e:
            raise SkillZipError(f"tools.json is not valid JSON: {e}") from e
        if isinstance(parsed, list):
            tool_schema = [t for t in parsed if isinstance(t, dict)]
        else:
            raise SkillZipError("tools.json must be a JSON array of tool dicts.")

    # The manifest may define arbitrary extension fields under `xct` —
    # everything else (category, tags, domain, model_recommendations …)
    # is stashed there so we don't ossify the schema on day one.
    xct_metadata: Dict[str, Any] = {}
    if isinstance(manifest.get("xct"), dict):
        xct_metadata.update(manifest["xct"])
    # Promote a few common top-level fields if they appear at the manifest
    # root and not under `xct`.
    for k in ("category", "tags", "domain", "model_recommendations"):
        if k in manifest and k not in xct_metadata:
            xct_metadata[k] = manifest[k]

    version = str(manifest.get("version") or "1").strip() or "1"
    is_public = bool(manifest.get("is_public", False))

    verbose_proxy_logger.debug(
        "parsed skill ZIP: title=%s version=%s tools=%d size=%d",
        display_title,
        version,
        len(tool_schema or []),
        len(payload),
    )

    return ParsedSkill(
        display_title=display_title,
        description=str(description) if description else None,
        instructions=instructions,
        system_prompt_template=system_prompt_template,
        tool_schema=tool_schema,
        version=version,
        is_public=is_public,
        xct_metadata=xct_metadata,
        file_content=payload,
        file_name=file_name,
        file_type="application/zip",
    )
