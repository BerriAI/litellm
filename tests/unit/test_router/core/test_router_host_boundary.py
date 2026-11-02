"""
Both router cores and the helpers they share reach the process only through their host: no router
module reads a ``litellm.<setting>`` or calls a ``litellm.<function>`` directly. Exception classes,
types and modules (``litellm.RateLimitError``, ``litellm.Usage``, ``litellm.utils``) are code, not
process state, and stay importable.
"""

import ast
import inspect
from pathlib import Path
from typing import Final

import litellm

_PACKAGE: Final = Path(litellm.__file__).parent
# router_config.py is the host's own view of the process settings; router_python_host.py is the host.
_HOST_MODULES: Final = frozenset({"router_config.py", "router_python_host.py"})


def _router_modules() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in (
                *_PACKAGE.glob("router_*.py"),
                *(_PACKAGE / "router_utils").rglob("*.py"),
                *(_PACKAGE / "router_strategy").rglob("*.py"),
            )
            if path.name not in _HOST_MODULES
        )
    )


def _is_code(name: str) -> bool:
    value: Final[object] = getattr(litellm, name, None)
    return inspect.isclass(value) or inspect.ismodule(value)


def _process_reads(path: Path) -> tuple[str, ...]:
    tree: Final = ast.parse(path.read_text(encoding="utf-8"))
    return tuple(
        f"{path.relative_to(_PACKAGE.parent)}:{node.lineno} litellm.{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "litellm"
        and not _is_code(node.attr)
    )


def test_router_code_reaches_litellm_settings_and_functions_only_through_the_host() -> None:
    modules: Final = _router_modules()
    assert len(modules) > 50
    assert [read for path in modules for read in _process_reads(path)] == []
