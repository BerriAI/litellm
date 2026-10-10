from pathlib import Path

import pytest

from scripts.check_layer_imports import main


def build_tree(root: Path, files: dict[str, str], allowlist: str = "") -> None:
    for relative, source in {
        "litellm/proxy/_types.py": "",
        "litellm/main.py": "",
        "litellm/router.py": "",
        **files,
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts/layer_imports_allowlist.txt").write_text(allowlist)


def run_check(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base_allowed: frozenset[str] | None = None,
) -> tuple[int, str]:
    monkeypatch.chdir(root)
    code = main(["--base", "origin/main"], lambda _ref: base_allowed)
    return code, capsys.readouterr().err


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    (
        (
            "litellm/integrations/foo.py",
            "from litellm.proxy._types import UserAPIKeyAuth\n",
            "litellm/integrations/foo.py:1: L1 forbids litellm.integrations.foo importing litellm.proxy._types (module level)",
        ),
        (
            "litellm/integrations/foo.py",
            "def f():\n    from litellm import proxy\n",
            "litellm/integrations/foo.py:2: L1 forbids litellm.integrations.foo importing litellm.proxy (function level)",
        ),
        (
            "litellm/llms/foo/handler.py",
            "import os\nfrom litellm import main\n",
            "litellm/llms/foo/handler.py:2: L2 forbids litellm.llms.foo.handler importing litellm.main (module level)",
        ),
        (
            "litellm/llms/foo/handler.py",
            "import litellm.router\n",
            "litellm/llms/foo/handler.py:1: L2 forbids litellm.llms.foo.handler importing litellm.router (module level)",
        ),
        (
            "litellm/types/foo.py",
            "from litellm.integrations.custom_logger import CustomLogger\n",
            "litellm/types/foo.py:1: L3 forbids litellm.types.foo importing litellm.integrations.custom_logger (module level)",
        ),
        (
            "litellm/types/foo.py",
            "import litellm\n",
            "litellm/types/foo.py:1: L3 forbids litellm.types.foo importing litellm (module level)",
        ),
        (
            "litellm/types/sub/foo.py",
            "from ...proxy import _types\n",
            "litellm/types/sub/foo.py:1: L3 forbids litellm.types.sub.foo importing litellm.proxy._types (module level)",
        ),
        (
            "litellm/litellm_core_utils/duration_parser.py",
            "def f():\n    from litellm.utils import x\n",
            "litellm/litellm_core_utils/duration_parser.py:2: LEAF forbids litellm.litellm_core_utils.duration_parser "
            "importing litellm.utils (function level)",
        ),
        (
            "litellm/litellm_core_utils/llm_response_utils/get_headers.py",
            "from litellm.router_strategy import x\n",
            "litellm/litellm_core_utils/llm_response_utils/get_headers.py:1: LEAF forbids "
            "litellm.litellm_core_utils.llm_response_utils.get_headers importing litellm.router_strategy (module level)",
        ),
        (
            "litellm/integrations/foo.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\nelse:\n    import litellm.proxy.utils\n",
            "litellm/integrations/foo.py:5: L1 forbids litellm.integrations.foo importing litellm.proxy.utils (module level)",
        ),
    ),
)
def test_new_layer_violation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    path: str,
    source: str,
    expected: str,
) -> None:
    build_tree(tmp_path, {path: source})
    code, err = run_check(tmp_path, monkeypatch, capsys)
    assert code == 1
    assert expected in err.splitlines()


@pytest.mark.parametrize(
    ("path", "source"),
    (
        (
            "litellm/integrations/foo.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from litellm.proxy import _types\n",
        ),
        ("litellm/integrations/foo.py", "import typing\nif typing.TYPE_CHECKING:\n    import litellm.proxy.utils\n"),
        (
            "litellm/integrations/foo.py",
            "def f():\n    if TYPE_CHECKING:\n        from litellm.proxy._types import X\n",
        ),
        ("litellm/proxy/foo.py", "from litellm.proxy._types import X\nimport litellm.main\n"),
        ("litellm/llms/foo.py", "import litellm\nfrom litellm import Router\nfrom litellm.router_strategy import x\n"),
        (
            "litellm/types/foo.py",
            "from litellm.types.utils import X\nfrom litellm.models.team import Y\nfrom litellm.constants import Z\n"
            "from litellm._logging import verbose_logger\nfrom litellm._uuid import uuid\nimport pydantic\n"
            "from litellm.litellm_core_utils.completion_timeout import CompletionTimeout\n"
            "from litellm.litellm_core_utils.core_helpers import normalize_drop_params\n"
            "from litellm.litellm_core_utils.provider_affinity import validate_provider_affinity_header_name\n"
            "from litellm.litellm_core_utils.duration_parser import duration_in_seconds\n",
        ),
        (
            "litellm/litellm_core_utils/core_helpers.py",
            "from litellm.litellm_core_utils.llm_response_utils.get_headers import get_response_headers\n"
            "from litellm.types.utils import X\nfrom litellm._logging import verbose_logger\n",
        ),
        ("litellm/litellm_core_utils/other.py", "import litellm.router\nfrom litellm.utils import x\n"),
    ),
)
def test_allowed_imports_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], path: str, source: str
) -> None:
    build_tree(tmp_path, {path: source})
    assert run_check(tmp_path, monkeypatch, capsys) == (0, "")


def test_allowlisted_violation_passes_and_stale_entry_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    llms_source = "from litellm.proxy._types import A\nfrom litellm.proxy._types import B\n"
    entries = (
        "# header\n\nL1  module litellm/llms/foo.py litellm.proxy._types\n"
        "L2 module litellm/llms/foo.py litellm.proxy._types\n"
    )
    build_tree(tmp_path, {"litellm/llms/foo.py": llms_source}, entries)
    assert run_check(tmp_path, monkeypatch, capsys) == (0, "")

    build_tree(tmp_path, {"litellm/llms/foo.py": "def f():\n    from litellm.proxy._types import A\n"}, entries)
    code, err = run_check(tmp_path, monkeypatch, capsys)
    assert code == 1
    assert sorted(err.splitlines()) == [
        "litellm/llms/foo.py:2: L1 forbids litellm.llms.foo importing litellm.proxy._types (function level)",
        "litellm/llms/foo.py:2: L2 forbids litellm.llms.foo importing litellm.proxy._types (function level)",
        "scripts/layer_imports_allowlist.txt: stale entry 'L1 module litellm/llms/foo.py litellm.proxy._types' "
        "no longer matches an import, delete it",
        "scripts/layer_imports_allowlist.txt: stale entry 'L2 module litellm/llms/foo.py litellm.proxy._types' "
        "no longer matches an import, delete it",
    ]


def test_allowlist_entry_added_since_base_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_tree(
        tmp_path,
        {
            "litellm/types/foo.py": "from litellm.router import Router\n",
            "litellm/types/bar.py": "import litellm.main\n",
        },
        "L3 module litellm/types/foo.py litellm.router\nL3 module litellm/types/bar.py litellm.main\n",
    )
    base = frozenset({"L3 module litellm/types/foo.py litellm.router"})

    assert run_check(tmp_path, monkeypatch, capsys, base) == (
        1,
        "scripts/layer_imports_allowlist.txt: entry 'L3 module litellm/types/bar.py litellm.main' is not in the "
        "base allowlist, fix the import instead of allowlisting it\n",
    )
    assert run_check(tmp_path, monkeypatch, capsys, base | {"L3 module litellm/types/bar.py litellm.main"}) == (0, "")


def test_missing_base_allowlist_skips_growth_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_tree(
        tmp_path,
        {"litellm/types/foo.py": "from litellm.router import Router\n"},
        "L3 module litellm/types/foo.py litellm.router\n",
    )

    assert run_check(tmp_path, monkeypatch, capsys, None) == (0, "")
