import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm_cli.main import cli

_HELP_ARGV: Final = (
    ("--help",),
    ("logout", "--help"),
    ("login", "--help"),
    ("models", "list", "--help"),
    ("claude", "--help"),
    ("opencode", "--help"),
    ("codex", "--help"),
    ("pi", "--help"),
    ("keys", "list", "--help"),
)

_COMMAND_ARGV: Final = (("logout",), ("whoami",))

_RUN_EACH_ARGV: Final = """
import contextlib, io, json, sys
from {module} import cli
runs = {{}}
for argv in json.loads(sys.argv[1]):
    sys.argv = ["lite", *argv]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            cli()
        except SystemExit as stopped:
            code = stopped.code
        else:
            code = 0
    runs[" ".join(argv)] = {{"exit": code, "output": output.getvalue()}}
print(json.dumps({{"runs": runs, "sdk_initialized": "completion" in vars(sys.modules["litellm"])}}))
"""

_USE_SDK_AFTER_HELP: Final = """
import builtins, importlib, json, sys
interpreter_imports = (builtins.__import__, importlib.import_module)
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
    "imports_restored": (builtins.__import__, importlib.import_module) == interpreter_imports,
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

_STAR_IMPORT_AFTER_HELP: Final = """
import json, sys
from litellm_cli.main import cli
sys.argv = ["lite", "--help"]
try:
    cli()
except SystemExit:
    pass
before = "completion" in vars(sys.modules["litellm"])
from litellm import *
print(json.dumps({"before": before, "completion": callable(globals().get("completion")), "acompletion": callable(globals().get("acompletion"))}))
"""

_SDK_WHOSE_SUBMODULE_READS_IT_WHILE_LOADING: Final = {
    "sdk_sync.py": """
import threading
initializing = threading.Event()
submodule_loading = threading.Event()
""",
    "litellm/__init__.py": """
import sdk_sync
sdk_sync.initializing.set()
sdk_sync.submodule_loading.wait(2)
from litellm import submodule
ready = True
""",
    "litellm/submodule.py": """
import litellm
import sdk_sync
sdk_sync.submodule_loading.set()
ready_when_loaded = getattr(litellm, "ready", False)
""",
    "litellm/_process_defaults.py": "def apply_process_defaults() -> None: ...\n",
    "litellm/proxy/__init__.py": "",
    "litellm/proxy/client/__init__.py": "",
    "litellm/proxy/client/cli/__init__.py": "",
    "litellm/proxy/client/cli/main.py": """
import json, threading
import litellm
import sdk_sync

def cli():
    def initialize_sdk():
        litellm.ready

    def import_submodule_once_initialization_starts():
        sdk_sync.initializing.wait()
        import litellm.submodule

    threads = [threading.Thread(target=initialize_sdk, daemon=True),
               threading.Thread(target=import_submodule_once_initialization_starts, daemon=True)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    print(json.dumps({"finished": [not thread.is_alive() for thread in threads]}))

litellm_proxy_cli = cli
""",
}

_SDK_WHOSE_INIT_FAILS: Final = {
    "litellm/__init__.py": 'raise ImportError("a dependency of the SDK is broken")\n',
    "litellm/utils.py": "",
    "litellm/_process_defaults.py": "def apply_process_defaults() -> None: ...\n",
    "litellm/proxy/__init__.py": "",
    "litellm/proxy/client/__init__.py": "",
    "litellm/proxy/client/cli/__init__.py": "",
    "litellm/proxy/client/cli/main.py": """
import json
import litellm

def error_from(use):
    try:
        use()
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None

def import_sdk_code():
    import litellm.utils

def cli():
    print(json.dumps([error_from(lambda: litellm.completion), error_from(lambda: litellm.completion), error_from(import_sdk_code)]))

litellm_proxy_cli = cli
""",
}

_RUN_LITE_AGAINST_SDK_AT: Final = """
import sys
sys.path.insert(0, sys.argv[1])
from litellm_cli.main import cli
cli()
"""

_COMMAND_MODULES_LOADED_BY: Final = """
import json, sys
from litellm_cli.main import cli
sys.argv = ["lite", *sys.argv[1:]]
try:
    cli()
except SystemExit:
    pass
prefix = "litellm.proxy.client.cli.commands."
print(json.dumps({"modules": sorted(name[len(prefix):] for name in sys.modules if name.startswith(prefix))}))
"""


def _run(source: str, argv: tuple[str, ...], home: Path) -> dict[str, object]:
    result: Final = subprocess.run(
        [sys.executable, "-I", "-c", source, *argv],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "HOME": str(home), "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.rstrip("\n").rpartition("\n")[2])


@pytest.fixture(scope="module")
def cli_runs(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, dict[str, object]]:
    argv: Final = (json.dumps(_HELP_ARGV + _COMMAND_ARGV),)
    return {
        "thin": _run(_RUN_EACH_ARGV.format(module="litellm_cli.main"), argv, tmp_path_factory.mktemp("thin")),
        "full": _run(
            _RUN_EACH_ARGV.format(module="litellm.proxy.client.cli.main"), argv, tmp_path_factory.mktemp("full")
        ),
    }


def test_lite_runs_commands_without_initializing_the_sdk(cli_runs: Mapping[str, dict[str, object]]):
    assert cli_runs["thin"]["sdk_initialized"] is False
    assert cli_runs["full"]["sdk_initialized"] is True


@pytest.mark.parametrize("argv", _HELP_ARGV + _COMMAND_ARGV, ids=" ".join)
def test_lite_prints_what_the_full_sdk_cli_prints(argv: tuple[str, ...], cli_runs: Mapping[str, dict[str, object]]):
    thin: Final = cli_runs["thin"]["runs"]
    full: Final = cli_runs["full"]["runs"]
    assert isinstance(thin, dict) and isinstance(full, dict)

    assert thin[" ".join(argv)] == full[" ".join(argv)]
    assert thin[" ".join(argv)]["output"]


@pytest.mark.parametrize(
    ("argv", "loaded", "not_loaded"),
    [
        (("logout", "--help"), {"auth"}, {"agents", "autoroute.commands", "models", "keys"}),
        (("models", "list", "--help"), {"models"}, {"agents", "autoroute.commands", "keys"}),
        (("claude", "--help"), {"agents"}, {"autoroute.commands", "models", "keys"}),
    ],
)
def test_lite_imports_only_the_command_it_runs(
    argv: tuple[str, ...], loaded: set[str], not_loaded: set[str], tmp_path: Path
):
    modules: Final = _run(_COMMAND_MODULES_LOADED_BY, argv, tmp_path)["modules"]
    assert isinstance(modules, list)

    assert loaded <= set(modules)
    assert not not_loaded & set(modules)


@pytest.fixture(scope="module")
def sdk_used_after_full_cli(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    return _run(
        _USE_SDK_AFTER_HELP.format(module="litellm.proxy.client.cli.main", first_use="pass"),
        (),
        tmp_path_factory.mktemp("full"),
    )


@pytest.mark.parametrize(
    "first_use",
    [
        "import litellm; litellm.completion",
        "import litellm; litellm.types.utils.ModelResponse",
        "from litellm import acompletion",
        "from litellm.router import Router",
        "from litellm.types import utils",
        "import litellm.responses.main",
        'importlib.import_module("litellm.router")',
        'importlib.import_module(".utils", "litellm")',
        'exec("from .types import utils", {"__name__": "litellm.lazy_import", "__package__": "litellm"})',
    ],
)
def test_sdk_initializes_in_place_on_first_use_after_lite_imports(
    first_use: str, sdk_used_after_full_cli: dict[str, object], tmp_path: Path
):
    thin: Final = _run(_USE_SDK_AFTER_HELP.format(module="litellm_cli.main", first_use=first_use), (), tmp_path)

    assert thin["before"] is False
    assert sdk_used_after_full_cli["before"] is True
    assert {**thin, "before": True} == sdk_used_after_full_cli
    assert (thin["router"], thin["completion"], thin["acompletion"], thin["responses_is_function"]) == (
        "Router",
        True,
        True,
        True,
    )


def test_star_import_after_lite_imports_gets_the_whole_sdk(tmp_path: Path):
    assert _run(_STAR_IMPORT_AFTER_HELP, (), tmp_path) == {"before": False, "completion": True, "acompletion": True}


def test_threads_that_use_the_sdk_at_once_all_wait_for_one_initialization(tmp_path: Path):
    used: Final = _run(_FIRST_USE_FROM_THREADS, (), tmp_path)

    assert used == {"seen": [True, True, True, True]}


def _write_sdk(root: Path, files: Mapping[str, str]) -> Path:
    for relative_path, source in files.items():
        (root / relative_path).parent.mkdir(parents=True, exist_ok=True)
        (root / relative_path).write_text(source)
    return root


def test_importing_sdk_code_while_another_thread_initializes_the_sdk_does_not_hang(tmp_path: Path):
    sdk_root: Final = _write_sdk(tmp_path / "sdk", _SDK_WHOSE_SUBMODULE_READS_IT_WHILE_LOADING)

    assert _run(_RUN_LITE_AGAINST_SDK_AT, (str(sdk_root),), tmp_path) == {"finished": [True, True]}


def test_every_use_after_a_failed_sdk_init_raises_the_init_error(tmp_path: Path):
    sdk_root: Final = _write_sdk(tmp_path / "sdk", _SDK_WHOSE_INIT_FAILS)

    assert _run(_RUN_LITE_AGAINST_SDK_AT, (str(sdk_root),), tmp_path) == ["ImportError: a dependency of the SDK is broken"] * 3


def test_lite_keeps_an_sdk_that_was_already_imported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setitem(sys.modules, "litellm", litellm)
    monkeypatch.setattr(litellm, "success_callback", ["kept"])
    monkeypatch.setattr(sys, "argv", ["lite", "--help"])

    with pytest.raises(SystemExit) as exited:
        cli()

    assert exited.value.code == 0
    assert "LiteLLM Proxy CLI - Manage your LiteLLM proxy server" in capsys.readouterr().out
    assert sys.modules["litellm"] is litellm
    assert litellm.success_callback == ["kept"]
