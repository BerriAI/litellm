import hashlib
import os
import shutil
from collections.abc import Mapping
from typing import Final
from urllib.parse import quote

from anyio import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.responses import FileResponse, Response
from starlette.staticfiles import NotModifiedResponse
from starlette.types import Scope

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.discovery_endpoints.ui_discovery_endpoints import LITELLM_UI_CONFIG_PATH, UI_CONFIG_PATH

_PACKAGED_UI_PATH: Final = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_experimental", "out")
_ASSET_PREFIX: Final = "/litellm-asset-prefix"
_BINARY_ASSETS: Final = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot")
_REWRITABLE_ASSETS: Final = (".html", ".js", ".css", ".json", ".txt", ".svg", ".webmanifest")


def _has_content(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        with os.scandir(path) as entries:
            return any(entries)
    except FileNotFoundError:
        return False


def _valid_ui_directory(path: str) -> bool:
    return os.path.exists(os.path.join(path, "index.html")) and os.path.isdir(os.path.join(path, "_next"))


def _is_pre_restructured(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, ".litellm_ui_ready")):
        return True
    if not os.path.exists(os.path.join(path, "index.html")):
        return False
    try:
        with os.scandir(path) as entries:
            return any(
                entry.is_dir()
                and not entry.name.startswith("_")
                and os.path.exists(os.path.join(entry.path, "index.html"))
                for entry in entries
            )
    except (FileNotFoundError, OSError):
        return False


def _select_ui_path() -> str:
    non_root: Final = os.getenv("LITELLM_NON_ROOT", "").lower() == "true"
    default_path: Final = "/var/lib/litellm/ui" if non_root else _PACKAGED_UI_PATH
    runtime_path: Final = os.getenv("LITELLM_UI_PATH", default_path)
    if runtime_path == _PACKAGED_UI_PATH:
        return _PACKAGED_UI_PATH
    if _has_content(runtime_path):
        return runtime_path
    try:
        os.makedirs(runtime_path, exist_ok=True)
        if not _has_content(runtime_path) and _has_content(_PACKAGED_UI_PATH):
            shutil.copytree(_PACKAGED_UI_PATH, runtime_path, dirs_exist_ok=True)
            return runtime_path
    except (OSError, shutil.Error) as exc:
        verbose_proxy_logger.warning("Failed to populate UI at %s: %s", runtime_path, exc)
    verbose_proxy_logger.warning("Falling back to packaged UI at %s", _PACKAGED_UI_PATH)
    return _PACKAGED_UI_PATH


def _with_server_root(content: str, root_path: str) -> str:
    return content.replace(_ASSET_PREFIX, root_path).replace(LITELLM_UI_CONFIG_PATH, f"{root_path}{UI_CONFIG_PATH}")


class _MountedUIFiles(StaticFiles):
    def __init__(self, mount_path: str, directory: str, *, html: bool = False) -> None:
        super().__init__(directory=directory, html=html)
        self.mount_path: Final = mount_path

    async def get_response(self, path: str, scope: Scope) -> Response:
        request_scope: Final[Mapping[str, object]] = scope
        mounted_root: Final = request_scope.get("root_path", "")
        root: Final = mounted_root.removesuffix(self.mount_path).rstrip("/") if isinstance(mounted_root, str) else ""
        if not root.startswith("/") or root.startswith("//"):
            return await super().get_response(path, scope)
        headers: Final = Headers(scope=scope)
        conditional_headers: Final = (b"if-none-match", b"if-modified-since")
        response: Final = await super().get_response(
            path, {**scope, "headers": [(key, value) for key, value in headers.raw if key not in conditional_headers]}
        )
        if not isinstance(response, FileResponse):
            return response
        may_rewrite: Final = str(response.path).endswith(_REWRITABLE_ASSETS)
        encoded_root: Final = quote(root)
        if may_rewrite:
            mounted_etag: Final = hashlib.sha256(f"{response.headers['etag']}:{encoded_root}".encode()).hexdigest()
            response.headers["etag"] = f'"{mounted_etag}"'
        if response.status_code == 200 and self.is_not_modified(response.headers, headers):
            return NotModifiedResponse(response.headers)
        if may_rewrite:
            rewritten: Final = await _rewrite_mounted_asset(
                response, encoded_root, head=request_scope["method"] == "HEAD"
            )
            if rewritten is not None:
                return rewritten
        return response


async def _rewrite_mounted_asset(response: FileResponse, root_path: str, *, head: bool) -> Response | None:
    try:
        content: Final = await Path(response.path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    rewritten: Final = _with_server_root(content, root_path)
    if rewritten == content:
        return None
    encoded: Final = rewritten.encode("utf-8")
    return Response(
        b"" if head else encoded,
        status_code=response.status_code,
        headers={
            "etag": response.headers["etag"],
            "last-modified": response.headers["last-modified"],
            "content-length": str(len(encoded)),
        },
        media_type=response.media_type,
    )


def _rewrite_server_root(path: str, server_root_path: str) -> None:
    if not server_root_path or server_root_path == "/":
        return
    if not os.access(path, os.W_OK):
        verbose_proxy_logger.warning(
            "Cannot apply server_root_path replacements to UI at %s: path is not writable.", path
        )
        return
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(_BINARY_ASSETS):
                continue
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, encoding="utf-8") as file:
                    content = file.read()
                rewritten = _with_server_root(content, server_root_path)
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(rewritten)
            except (UnicodeDecodeError, PermissionError, OSError):
                continue


def restructure_ui_html_files(path: str) -> None:
    for root, _, files in os.walk(path):
        first_segment = os.path.relpath(root, path).split(os.sep, 1)[0]
        if first_segment in {"_next", "litellm-asset-prefix"}:
            continue
        for filename in files:
            if not filename.endswith(".html") or filename == "index.html":
                continue
            file_path = os.path.join(root, filename)
            target_dir = os.path.splitext(file_path)[0]
            os.makedirs(target_dir, exist_ok=True)
            try:
                os.replace(file_path, os.path.join(target_dir, "index.html"))
            except FileNotFoundError:
                continue


def mount_admin_ui(app: FastAPI, server_root_path: str = "", *, request_root_path: bool = False) -> None:
    try:
        path: Final = _select_ui_path()
        if path != _PACKAGED_UI_PATH and not _valid_ui_directory(_PACKAGED_UI_PATH):
            verbose_proxy_logger.error("Packaged UI at %s is invalid or incomplete.", _PACKAGED_UI_PATH)
        if not _valid_ui_directory(path):
            verbose_proxy_logger.error("Selected UI path %s is invalid or incomplete.", path)
        if not request_root_path:
            _rewrite_server_root(path, server_root_path)
        assets: Final = os.path.join(path, "_next")
        for mount, directory, name in (
            ("/_next", assets, "next_static"),
            (f"{_ASSET_PREFIX}/_next", assets, "next_static"),
            ("/ui", path, "ui"),
        ):
            static = (
                _MountedUIFiles(mount, directory, html=name == "ui")
                if request_root_path
                else StaticFiles(directory=directory, html=name == "ui")
            )
            app.mount(mount, static, name=name)
        if not _is_pre_restructured(path):
            if os.access(path, os.W_OK):
                restructure_ui_html_files(path)
            else:
                verbose_proxy_logger.warning("Cannot restructure UI at %s: path is not writable.", path)
    except Exception:
        verbose_proxy_logger.exception("Unable to prepare the Admin UI static export")
