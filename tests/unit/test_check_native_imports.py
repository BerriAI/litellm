import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

from scripts.check_native_imports import main, violations

CHECKER: Final = Path(__file__).resolve().parents[2] / "scripts/check_native_imports.py"


@pytest.mark.parametrize(
    "path, source",
    (
        ("litellm/native/chat.py", "from litellm.main import execute_chat"),
        ("litellm/native/chat.py", "from litellm import main"),
        ("litellm/native/chat.py", "import litellm.main"),
        ("litellm/native/chat.py", "import litellm.main as legacy"),
        ("litellm/native/chat.py", "from ..main import completion"),
        ("litellm/native/chat.py", "from .. import main"),
        ("litellm/native/chat_providers/azure.py", "from ...main import completion"),
        ("litellm/native/chat.py", "import litellm\n\ndef run():\n    return litellm.main.completion()"),
    ),
)
def test_native_module_importing_main_is_rejected(path, source):
    assert violations(Path(path), source)


@pytest.mark.parametrize(
    "path, source",
    (
        ("litellm/native/chat.py", "from litellm.native.chat_providers.azure import complete_azure"),
        ("litellm/native/chat.py", "import litellm\n\ndef run():\n    return litellm.completion"),
        ("litellm/native/chat.py", "from litellm.maintenance import x"),
        ("litellm/native/chat.py", "from .chat_mock import mock_completion"),
        ("litellm/native/chat_providers/azure.py", "from ..chat_mock import mock_completion"),
        ("litellm/native/chat.py", "from litellm import utils"),
    ),
)
def test_native_module_without_main_dependency_passes(path, source):
    assert violations(Path(path), source) == ()


def test_command_reports_the_offending_line_and_passes_once_it_is_gone(tmp_path, monkeypatch, capsys):
    package: Final = tmp_path / "litellm/native/chat_providers"
    package.mkdir(parents=True)
    module: Final = package / "azure.py"
    module.write_text("import os\nfrom litellm.main import completion\n")
    monkeypatch.chdir(tmp_path)
    assert main() == 1
    assert "litellm/native/chat_providers/azure.py:2:" in capsys.readouterr().err
    rejected: Final = subprocess.run(
        [sys.executable, "-I", str(CHECKER)], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert rejected.returncode == 1

    module.write_text("import os\n")
    assert main() == 0
    accepted: Final = subprocess.run(
        [sys.executable, "-I", str(CHECKER)], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert accepted.returncode == 0


def test_the_native_package_has_no_main_dependency(monkeypatch):
    monkeypatch.chdir(CHECKER.parents[1])
    assert main() == 0
