from __future__ import annotations

import importlib.metadata
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast


def _direct_url_failures(direct_url: str | None) -> tuple[str, ...]:
    if direct_url is None:
        return ()
    try:
        direct_url_data: Final = cast(object, json.loads(direct_url))
    except json.JSONDecodeError:
        return ("litellm distribution direct_url.json is invalid JSON",)
    if not isinstance(direct_url_data, Mapping):
        return ()
    direct_url_mapping: Final = cast(Mapping[str, object], direct_url_data)
    dir_info: Final = direct_url_mapping.get("dir_info")
    if not isinstance(dir_info, Mapping):
        return ()
    dir_info_mapping: Final = cast(Mapping[str, object], dir_info)
    return ("litellm distribution is editable",) if dir_info_mapping.get("editable") is True else ()


def _native_bridge_available() -> bool:
    try:
        from litellm.rust_bridge import native_bridge_available
    except Exception:
        return False
    return native_bridge_available()


def provenance_failures(
    *,
    prefix: Path,
    checkout: Path,
    module_files: Mapping[str, Path],
    direct_url: str | None,
    native_available: bool,
) -> tuple[str, ...]:
    resolved_prefix: Final = prefix.resolve()
    resolved_checkout: Final = checkout.resolve()
    resolved_module_files: Final = tuple(
        (module_name, module_path.resolve()) for module_name, module_path in module_files.items()
    )
    prefix_failures: Final = tuple(
        f"{module_name} is not under sys.prefix: {module_path}"
        for module_name, module_path in resolved_module_files
        if not module_path.is_relative_to(resolved_prefix)
    )
    checkout_failures: Final = tuple(
        f"{module_name} is under the checkout: {module_path}"
        for module_name, module_path in resolved_module_files
        if module_path.is_relative_to(resolved_checkout)
    )
    direct_url_failures: Final = _direct_url_failures(direct_url)
    native_failures: Final = ("litellm.rust_bridge.native_bridge_available() is false",) if not native_available else ()
    return prefix_failures + checkout_failures + direct_url_failures + native_failures


def main(argv: Sequence[str] | None = None) -> int:
    arguments: Final = tuple(sys.argv if argv is None else argv)
    if len(arguments) != 2:
        sys.stderr.write(f"usage: {Path(arguments[0]).name} CHECKOUT\n")
        return 2

    checkout: Final = Path(arguments[1])
    try:
        import litellm
    except Exception as error:
        sys.stderr.write(f"litellm import failed: {error}\n")
        return 1

    try:
        import litellm.rust_bridge._native as native
    except Exception as error:
        sys.stderr.write(f"litellm.rust_bridge._native import failed: {error}\n")
        return 1

    imported_modules: Final = (
        ("litellm", litellm),
        ("litellm.rust_bridge._native", native),
    )
    missing_module_files: Final = tuple(
        f"{module_name} has no __file__" for module_name, module in imported_modules if module.__file__ is None
    )
    module_files: Final = MappingProxyType(
        {
            module_name: Path(module.__file__).resolve()
            for module_name, module in imported_modules
            if module.__file__ is not None
        }
    )
    try:
        direct_url: Final = importlib.metadata.distribution("litellm").read_text("direct_url.json")
    except Exception as error:
        sys.stderr.write(f"litellm distribution lookup failed: {error}\n")
        return 1

    native_available: Final = _native_bridge_available()

    failures: Final = missing_module_files + provenance_failures(
        prefix=Path(sys.prefix),
        checkout=checkout,
        module_files=module_files,
        direct_url=direct_url,
        native_available=native_available,
    )
    if failures:
        sys.stderr.write("".join(f"{failure}\n" for failure in failures))
        return 1

    sys.stdout.write("".join(f"{module_name}: {module_path}\n" for module_name, module_path in module_files.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
