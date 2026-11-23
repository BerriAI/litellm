from html import escape
from pathlib import Path
from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from litellm.proxy.swagger_reference import mount_swagger_ui

SCALAR_ASSET_PATH: Final = "/scalar"
_ASSET_DIR: Final = Path(__file__).with_name("scalar")


def mount_api_references(
    app: FastAPI, docs_url: str | None, swagger_url: str | None, *, lazy_features: bool = False
) -> None:
    openapi_url: Final = app.openapi_url
    if not openapi_url:
        return

    if docs_url:
        _add_scalar_reference(app, docs_url, openapi_url)
    mount_swagger_ui(app, swagger_url, lazy_features=lazy_features)
    if docs_url:
        app.mount(SCALAR_ASSET_PATH, StaticFiles(directory=_ASSET_DIR), name="scalar")


def _add_scalar_reference(app: FastAPI, docs_url: str, openapi_url: str) -> None:
    async def scalar_reference(request: Request) -> HTMLResponse:
        scope_root_path: Final[object] = request.scope.get("root_path")
        root_path: Final = scope_root_path.rstrip("/") if isinstance(scope_root_path, str) else ""
        assets: Final = escape(f"{root_path}{SCALAR_ASSET_PATH}")
        schema: Final = escape(f"{root_path}{openapi_url}")
        title: Final = escape(app.title)
        server: Final = escape(root_path or "/")
        return HTMLResponse(
            f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>body {{ margin: 0; }}</style>
</head>
<body>
  <div id="reference" data-openapi-url="{schema}" data-server-url="{server}"></div>
  <script src="{assets}/scalar-1.81.0.js"></script>
  <script src="{assets}/reference.js"></script>
</body>
</html>"""
        )

    app.add_route(docs_url, scalar_reference, methods=["GET"], include_in_schema=False)
