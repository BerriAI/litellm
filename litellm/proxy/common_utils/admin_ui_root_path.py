import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Final

LITELLM_ASSET_PREFIX: Final = "/litellm-asset-prefix"
RENDERED_UI_DIRNAME: Final = ".litellm-ui-rendered"

_UI_CONFIG_PATH: Final = b"/.well-known/litellm-ui-config"
_PLACEHOLDER_PATTERN: Final = re.compile(rb"/litellm-asset-prefix|/litellm/\.well-known/litellm-ui-config")
_RENDER_FORMAT: Final = b"1"
_BINARY_SUFFIXES: Final = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot")
_NON_ROUTE_DIRS: Final = frozenset({"_next", "litellm-asset-prefix", RENDERED_UI_DIRNAME})


@dataclass(frozen=True, slots=True)
class RenderedUI:
    directory: Path


@dataclass(frozen=True, slots=True)
class RenderUnavailable:
    errors: tuple[str, ...]


UIRenderResult = RenderedUI | RenderUnavailable


def _move_to_index(file_path: Path) -> None:
    target_dir: Final = file_path.with_suffix("")
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(file_path, target_dir / "index.html")
    except FileNotFoundError:
        return


def _restructure_directory(ui_root: Path, directory: Path, files: Sequence[str]) -> None:
    rel_root: Final = directory.relative_to(ui_root)
    if rel_root.parts[:1] and rel_root.parts[0] in _NON_ROUTE_DIRS:
        return
    for filename in files:
        if filename.endswith(".html") and filename != "index.html":
            _move_to_index(directory / filename)


def restructure_ui_html_files(ui_root: Path) -> None:
    for current_root, _, files in os.walk(ui_root):
        _restructure_directory(ui_root, Path(current_root), files)


def _files_in(source: Path, current_root: str, files: Sequence[str]) -> Iterator[Path]:
    relative_root: Final = Path(current_root).relative_to(source)
    return (relative_root / filename for filename in files)


def _source_files(source: Path) -> tuple[Path, ...]:
    walked: Final = chain.from_iterable(
        _files_in(source, current_root, files) for current_root, _, files in os.walk(source, followlinks=True)
    )
    return tuple(sorted(path for path in walked if path.parts[0] != RENDERED_UI_DIRNAME))


def _render_key(source: Path, server_root_path: str) -> str:
    fields: Final = (
        _RENDER_FORMAT,
        server_root_path.encode(),
        str(source.resolve()).encode(),
        *(
            b"%s %d %d" % (relative.as_posix().encode(), stat.st_size, stat.st_mtime_ns)
            for relative, stat in ((relative, (source / relative).stat()) for relative in _source_files(source))
        ),
    )
    return hashlib.sha256(b"\0".join(fields)).hexdigest()[:16]


def _substitute_root_path(content: bytes, server_root_path: str) -> bytes:
    root: Final = server_root_path.encode()

    def replacement(match: re.Match[bytes]) -> bytes:
        return root if match.group(0) == LITELLM_ASSET_PREFIX.encode() else root + _UI_CONFIG_PATH

    return _PLACEHOLDER_PATTERN.sub(replacement, content)


def _copy_rendered_file(source_file: Path, destination_file: Path, server_root_path: str) -> None:
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    if source_file.name.lower().endswith(_BINARY_SUFFIXES):
        shutil.copyfile(source_file, destination_file)
        return
    destination_file.write_bytes(_substitute_root_path(source_file.read_bytes(), server_root_path))


def _write_rendered_tree(source: Path, destination: Path, server_root_path: str) -> None:
    for relative in _source_files(source):
        _copy_rendered_file(source / relative, destination / relative, server_root_path)


def _render_into_parent(source: Path, server_root_path: str, key: str, parent: Path) -> UIRenderResult:
    output_root: Final = parent / RENDERED_UI_DIRNAME
    target: Final = output_root / key
    try:
        if target.is_dir():
            return RenderedUI(target)
        output_root.mkdir(parents=True, exist_ok=True)
        staging: Final = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=output_root))
    except OSError as error:
        return RenderUnavailable((f"{output_root}: {error}",))

    try:
        staging.chmod(0o755)
        _write_rendered_tree(source, staging, server_root_path)
        restructure_ui_html_files(staging)
        os.rename(staging, target)
    except OSError as error:
        if target.is_dir():
            return RenderedUI(target)
        return RenderUnavailable((f"{output_root}: {error}",))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return RenderedUI(target)


def _render_into_first_usable_parent(
    source: Path, server_root_path: str, key: str, output_parents: tuple[Path, ...]
) -> UIRenderResult:
    if not output_parents:
        return RenderUnavailable(())
    first: Final = _render_into_parent(source, server_root_path, key, output_parents[0])
    match first:
        case RenderedUI():
            return first
        case RenderUnavailable(errors=first_errors):
            rest: Final = _render_into_first_usable_parent(source, server_root_path, key, output_parents[1:])
            match rest:
                case RenderedUI():
                    return rest
                case RenderUnavailable(errors=rest_errors):
                    return RenderUnavailable(first_errors + rest_errors)


def render_ui_for_root_path(source: Path, server_root_path: str, output_parents: tuple[Path, ...]) -> UIRenderResult:
    if server_root_path in ("", "/"):
        return RenderedUI(source)
    try:
        key: Final = _render_key(source, server_root_path)
    except OSError as error:
        return RenderUnavailable((f"{source}: {error}",))
    return _render_into_first_usable_parent(source, server_root_path, key, output_parents)
