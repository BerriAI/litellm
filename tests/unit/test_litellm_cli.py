import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

_RUN_ENTRY_POINT: Final = """
import json, sys
from {module} import cli
sys.argv = ["lite", *sys.argv[1:]]
try:
    cli()
except SystemExit as stopped:
    code = stopped.code
else:
    code = 0
print(json.dumps({{"exit": code, "sdk_initialized": "completion" in vars(sys.modules["litellm"])}}))
"""

_USE_SDK_AFTER_HELP: Final = """
import builtins, json, sys
interpreter_import = builtins.__import__
from {module} import cli
sys.argv = ["lite", "--help"]
try:
    cli()
except SystemExit:
    pass
before = "completion" in vars(sys.modules["litellm"])
{first_use}
from litellm.router import Router
import litellm
import litellm.responses.main
from litellm import acompletion
print(json.dumps({{
    "before": before,
    "router": Router.__name__,
    "completion": callable(litellm.completion),
    "acompletion": acompletion is litellm.acompletion,
    "responses_is_function": callable(litellm.responses) and not isinstance(litellm.responses, type(litellm)),
    "namespace": sorted(vars(litellm)),
    "module_type": type(litellm).__name__,
    "import_restored": builtins.__import__ is interpreter_import,
}}))
"""

_FIRST_USE_FROM_THREADS: Final = """
import json, sys, threading
from litellm_cli.main import cli
sys.argv = ["lite", "--help"]
try:
    cli()
except SystemExit:
    pass
import litellm
start = threading.Barrier(4)
seen = []
def use_sdk():
    start.wait()
    try:
        seen.append(callable(litellm.completion))
    except AttributeError as error:
        seen.append(str(error))
threads = [threading.Thread(target=use_sdk) for _ in range(4)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print(json.dumps({"seen": seen}))
"""


def _run(source: str, argv: tuple[str, ...], home: Path) -> tuple[str, dict[str, object]]:
    result: Final = subprocess.run(
        [sys.executable, "-I", "-c", source, *argv],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "HOME": str(home), "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    assert result.returncode == 0, result.stderr
    output, _, summary = result.stdout.rstrip("\n").rpartition("\n")
    return output, json.loads(summary)


@pytest.mark.parametrize(
    "argv",
    [
        ("--help",),
        ("logout", "--help"),
        ("login", "--help"),
        ("models", "list", "--help"),
        ("claude", "--help"),
        ("opencode", "--help"),
        ("codex", "--help"),
        ("pi", "--help"),
        ("keys", "list", "--help"),
    ],
)
def test_lite_runs_commands_without_initializing_the_sdk(argv: tuple[str, ...], tmp_path: Path):
    thin_output, thin = _run(_RUN_ENTRY_POINT.format(module="litellm_cli.main"), argv, tmp_path)
    full_output, full = _run(_RUN_ENTRY_POINT.format(module="litellm.proxy.client.cli.main"), argv, tmp_path)

    assert thin == {"exit": full["exit"], "sdk_initialized": False}
    assert full["sdk_initialized"] is True
    assert thin_output == full_output
    assert "Usage:" in thin_output


@pytest.mark.parametrize(
    "first_use",
    [
        "import litellm; litellm.completion",
        "from litellm import acompletion",
        "from litellm.router import Router",
        "from litellm.types import utils",
        "import litellm.responses.main",
        'exec("from .types import utils", {"__name__": "litellm.lazy_import", "__package__": "litellm"})',
    ],
)
def test_sdk_initializes_in_place_on_first_use_after_lite_imports(first_use: str, tmp_path: Path):
    _, thin = _run(_USE_SDK_AFTER_HELP.format(module="litellm_cli.main", first_use=first_use), (), tmp_path)
    _, full = _run(
        _USE_SDK_AFTER_HELP.format(module="litellm.proxy.client.cli.main", first_use=first_use), (), tmp_path
    )

    assert thin["before"] is False
    assert full["before"] is True
    assert {**thin, "before": True} == full
    assert (thin["router"], thin["completion"], thin["acompletion"], thin["responses_is_function"]) == (
        "Router",
        True,
        True,
        True,
    )


def test_threads_that_use_the_sdk_at_once_all_wait_for_one_initialization(tmp_path: Path):
    _, used = _run(_FIRST_USE_FROM_THREADS, (), tmp_path)

    assert used == {"seen": [True, True, True, True]}
