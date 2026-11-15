import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest

from litellm.scripts.stub_prisma_runtime_types import main, stub_runtime_types

GENERATED_TYPES: Final = (
    "from typing import TypedDict\n\n\nclass LiteLLM_TeamTableWhereInput(TypedDict, total=False):\n    team_id: str\n"
)


def _generated_client(tmp_path: Path) -> Path:
    (tmp_path / "types.py").write_text(GENERATED_TYPES, encoding="utf-8")
    return tmp_path


def _imported_runtime_types(prisma_dir: Path) -> ModuleType:
    spec: Final = importlib.util.spec_from_file_location("stubbed_prisma_types", prisma_dir / "types.py")
    assert spec is not None and spec.loader is not None
    module: Final = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stubbed_types_define_no_classes_but_resolve_every_generated_name(tmp_path: Path) -> None:
    prisma_dir: Final = _generated_client(tmp_path)

    assert stub_runtime_types(prisma_dir) is True

    runtime_types: Final = _imported_runtime_types(prisma_dir)
    assert [name for name in vars(runtime_types) if name.startswith("LiteLLM_")] == []
    assert runtime_types.LiteLLM_TeamTableWhereInput == dict[str, Any]


def test_stubbed_types_still_build_the_dict_a_typeddict_call_builds(tmp_path: Path) -> None:
    prisma_dir: Final = _generated_client(tmp_path)
    _ = stub_runtime_types(prisma_dir)

    assert _imported_runtime_types(prisma_dir).DatasourceOverride(url="postgresql://db") == {"url": "postgresql://db"}


def test_stubbed_types_still_report_missing_dunders(tmp_path: Path) -> None:
    prisma_dir: Final = _generated_client(tmp_path)
    _ = stub_runtime_types(prisma_dir)

    with pytest.raises(AttributeError):
        _ = _imported_runtime_types(prisma_dir).__wrapped__


def test_generated_types_are_kept_for_type_checkers(tmp_path: Path) -> None:
    prisma_dir: Final = _generated_client(tmp_path)

    _ = stub_runtime_types(prisma_dir)

    assert (prisma_dir / "types.pyi").read_text(encoding="utf-8") == GENERATED_TYPES


def test_second_run_does_not_overwrite_the_kept_generated_types(tmp_path: Path) -> None:
    prisma_dir: Final = _generated_client(tmp_path)
    _ = stub_runtime_types(prisma_dir)

    assert stub_runtime_types(prisma_dir) is False
    assert (prisma_dir / "types.pyi").read_text(encoding="utf-8") == GENERATED_TYPES


def test_main_stubs_the_named_installed_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir: Final = tmp_path / "generated_client_under_test"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    _ = _generated_client(package_dir)
    monkeypatch.syspath_prepend(str(tmp_path))

    assert main("generated_client_under_test") == 0

    assert (package_dir / "types.pyi").read_text(encoding="utf-8") == GENERATED_TYPES
    assert _imported_runtime_types(package_dir).LiteLLM_TeamTableWhereInput == dict[str, Any]


def test_main_fails_when_the_client_is_not_installed() -> None:
    assert main("no_such_generated_client") == 1
