import json
import re
import struct
import zipfile
from typing import Final

from litellm.proxy.discovery_endpoints.agent_skills_archive import pack_skill_archive, read_skill_archive_members

MAX_SKILL_UPLOAD_BYTES: Final = 10 * 1024 * 1024
MAX_SKILL_ARCHIVE_COMPRESSION_RATIO: Final = 50


def build_uploaded_skill_archive(content: bytes, name: str) -> bytes | None:
    if len(content) > MAX_SKILL_UPLOAD_BYTES or re.fullmatch(r"[a-z0-9-]+", name) is None:
        return None
    try:
        skill: Final = read_skill_archive_members(content, for_upload=True)
        if skill is None:
            return None
        members: Final = ((".claude-plugin/plugin.json", json.dumps({"name": name}).encode(), 0o644),) + tuple(
            (f"skills/{name}/{path}", data, mode) for path, data, mode in skill
        )
        compressed: Final = pack_skill_archive(members)
        return (
            pack_skill_archive(members, zipfile.ZIP_STORED)
            if sum(len(data) for _, data, _ in members) > len(compressed) * MAX_SKILL_ARCHIVE_COMPRESSION_RATIO
            else compressed
        )
    except (struct.error, UnicodeEncodeError):
        return None
