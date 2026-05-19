"""Tests for the skill ZIP parser + /v1/xct-skills/upload endpoint (S2-07)."""

import io
import json
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.skill_endpoints.endpoints import router
from litellm.proxy.skill_endpoints.skill_zip import (
    SkillZipError,
    parse_skill_zip,
)


def _zip_with(members: dict, *, mode=0o644) -> bytes:
    """Build a ZIP archive from a {name: text-or-bytes} dict."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in members.items():
            if isinstance(content, str):
                content = content.encode()
            info = zipfile.ZipInfo(filename=name)
            info.external_attr = (mode & 0xFFFF) << 16
            zf.writestr(info, content)
    return buf.getvalue()


def _zip_with_symlink(link_name: str, target: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo(filename=link_name)
        info.external_attr = 0xA1FF0000  # symlink mode (S_IFLNK in high bits)
        zf.writestr(info, target.encode())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# parse_skill_zip — unit
# ---------------------------------------------------------------------------


def test_parse_minimum_valid_zip():
    payload = _zip_with(
        {
            "manifest.yaml": "display_title: Hello\ndescription: greet\n",
            "SKILL.md": "You are a polite assistant.",
        }
    )
    parsed = parse_skill_zip(payload, file_name="hello.zip")
    assert parsed.display_title == "Hello"
    assert parsed.description == "greet"
    assert parsed.system_prompt_template == "You are a polite assistant."
    assert parsed.instructions == "You are a polite assistant."
    assert parsed.version == "1"
    assert parsed.file_content == payload
    assert parsed.file_name == "hello.zip"


def test_parse_extracts_tools_json():
    tools = [{"type": "function", "function": {"name": "ping"}}]
    payload = _zip_with(
        {
            "manifest.yaml": "display_title: T\n",
            "tools.json": json.dumps(tools),
        }
    )
    parsed = parse_skill_zip(payload)
    assert parsed.tool_schema == tools


def test_parse_promotes_top_level_metadata_to_xct_metadata():
    payload = _zip_with(
        {
            "manifest.yaml": (
                "display_title: T\n"
                "category: research\n"
                "tags: [a, b]\n"
                "xct:\n"
                "  domain: science\n"
            ),
        }
    )
    parsed = parse_skill_zip(payload)
    # xct.* is inherited, then top-level category/tags promoted IF not in xct.
    assert parsed.xct_metadata.get("domain") == "science"
    assert parsed.xct_metadata.get("category") == "research"
    assert parsed.xct_metadata.get("tags") == ["a", "b"]


def test_parse_rejects_empty_payload():
    with pytest.raises(SkillZipError):
        parse_skill_zip(b"")


def test_parse_rejects_not_a_zip():
    with pytest.raises(SkillZipError):
        parse_skill_zip(b"not a zip")


def test_parse_requires_manifest():
    payload = _zip_with({"SKILL.md": "no manifest!"})
    with pytest.raises(SkillZipError):
        parse_skill_zip(payload)


def test_parse_requires_display_title():
    payload = _zip_with({"manifest.yaml": "description: nope\n"})
    with pytest.raises(SkillZipError):
        parse_skill_zip(payload)


def test_parse_rejects_path_traversal():
    payload = _zip_with(
        {
            "manifest.yaml": "display_title: T\n",
            "../escape.txt": "evil",
        }
    )
    with pytest.raises(SkillZipError):
        parse_skill_zip(payload)


def test_parse_rejects_absolute_path():
    payload = _zip_with({"manifest.yaml": "display_title: T\n", "/etc/passwd": "no"})
    with pytest.raises(SkillZipError):
        parse_skill_zip(payload)


def test_parse_rejects_symlink():
    payload = _zip_with_symlink("evil-link", "/etc/passwd")
    with pytest.raises(SkillZipError):
        parse_skill_zip(payload)


def test_parse_invalid_yaml_manifest():
    payload = _zip_with({"manifest.yaml": "display_title: [unterminated\n"})
    with pytest.raises(SkillZipError):
        parse_skill_zip(payload)


def test_parse_invalid_tools_json():
    payload = _zip_with(
        {
            "manifest.yaml": "display_title: T\n",
            "tools.json": "{not json",
        }
    )
    with pytest.raises(SkillZipError):
        parse_skill_zip(payload)


# ---------------------------------------------------------------------------
# /v1/xct-skills/upload — endpoint
# ---------------------------------------------------------------------------


def _client(role=LitellmUserRoles.PROXY_ADMIN):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="sk-x", user_id="u-1", team_id="t-1", user_role=role
    )
    return TestClient(app)


def _mock_row(**overrides):
    base = dict(
        skill_id="sk-new",
        display_title="Hello",
        description=None,
        instructions=None,
        system_prompt_template=None,
        tool_schema=None,
        source="custom",
        version="1",
        is_public=False,
        team_id="t-1",
        user_id="u-1",
        created_by="u-1",
        xct_metadata={},
    )
    base.update(overrides)
    m = MagicMock()
    for k, v in base.items():
        setattr(m, k, v)
    m.model_dump = MagicMock(return_value=base)
    return m


def test_upload_endpoint_parses_zip_and_persists_row():
    payload = _zip_with(
        {
            "manifest.yaml": "display_title: From ZIP\nversion: 2\n",
            "SKILL.md": "Be brief.",
        }
    )
    prisma = MagicMock()
    prisma.db.litellm_skillstable.create = AsyncMock(
        return_value=_mock_row(skill_id="sk-99", display_title="From ZIP", version="2")
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _client().post(
            "/v1/xct-skills/upload",
            files={"file": ("skill.zip", payload, "application/zip")},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 200
    create_data = prisma.db.litellm_skillstable.create.call_args.kwargs["data"]
    assert create_data["display_title"] == "From ZIP"
    assert create_data["system_prompt_template"] == "Be brief."
    assert create_data["version"] == "2"
    assert create_data["source"] == "custom"
    assert create_data["file_content"] == payload
    assert create_data["file_name"] == "skill.zip"


def test_upload_endpoint_rejects_bad_zip_with_400():
    prisma = MagicMock()
    prisma.db.litellm_skillstable.create = AsyncMock()
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _client().post(
            "/v1/xct-skills/upload",
            files={"file": ("evil.zip", b"not a zip", "application/zip")},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 400
    prisma.db.litellm_skillstable.create.assert_not_awaited()


def test_upload_endpoint_rejects_wrong_content_type():
    prisma = MagicMock()
    prisma.db.litellm_skillstable.create = AsyncMock()
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _client().post(
            "/v1/xct-skills/upload",
            files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 400


def test_upload_endpoint_honors_is_public_override():
    payload = _zip_with({"manifest.yaml": "display_title: T\nis_public: false\n"})
    prisma = MagicMock()
    prisma.db.litellm_skillstable.create = AsyncMock(
        return_value=_mock_row(is_public=True)
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _client().post(
            "/v1/xct-skills/upload",
            files={"file": ("t.zip", payload, "application/zip")},
            data={"is_public_override": "true"},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 200
    create_data = prisma.db.litellm_skillstable.create.call_args.kwargs["data"]
    assert create_data["is_public"] is True
