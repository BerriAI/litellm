"""
Security tests for Skills ZIP extraction path traversal vulnerability.
"""

import os
import zipfile
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from litellm.llms.litellm_proxy.skills.prompt_injection import (
    SkillPromptInjectionHandler,
)


def _make_skill_with_zip(entries: dict) -> MagicMock:
    """Create a mock skill with a ZIP containing the given {name: content} entries."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    skill = MagicMock()
    skill.file_content = buf.getvalue()
    skill.skill_id = "test-skill"
    return skill


class TestExtractAllFilesPathTraversal:
    handler = SkillPromptInjectionHandler()

    def test_normal_extraction(self):
        """Normal ZIP entries should extract fine."""
        skill = _make_skill_with_zip({
            "my-skill/main.py": "print('hello')",
            "my-skill/lib/utils.py": "x = 1",
        })
        files = self.handler.extract_all_files(skill)
        assert "main.py" in files
        assert "lib/utils.py" in files

    def test_rejects_dotdot_traversal(self):
        """ZIP entries with ../ should be skipped."""
        skill = _make_skill_with_zip({
            "my-skill/../../../etc/cron.d/evil": "malicious",
            "my-skill/legit.py": "safe = True",
        })
        files = self.handler.extract_all_files(skill)
        # Malicious entry should be skipped
        assert not any("etc" in k for k in files)
        assert not any(".." in k for k in files)
        # Legit file should still be extracted
        assert "legit.py" in files

    def test_rejects_absolute_path(self):
        """ZIP entries with absolute paths should be skipped."""
        skill = _make_skill_with_zip({
            "my-skill//etc/passwd": "root:x:0:0",
            "my-skill/ok.py": "ok = True",
        })
        files = self.handler.extract_all_files(skill)
        assert not any("etc" in k for k in files)
        assert "ok.py" in files

    def test_rejects_deep_traversal(self):
        """ZIP entries with deeply nested traversal should be skipped."""
        skill = _make_skill_with_zip({
            "skill/../../../../tmp/pwn.sh": "#!/bin/bash\nrm -rf /",
        })
        files = self.handler.extract_all_files(skill)
        assert len(files) == 0
