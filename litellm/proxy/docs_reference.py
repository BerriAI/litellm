import html
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from litellm.proxy._lazy_features import lazy_tag_to_prefix

_ASSETS: Final = Path(__file__).parent / "api_reference"
_PLACEHOLDERS: Final = re.compile(r"\{\{(TITLE|ASSET_BASE|CONFIG)\}\}")
_SCRIPT_ESCAPES: Final = str.maketrans({"<": r"\u003c", ">": r"\u003e", "&": r"\u0026"})


def _root_path(scope: Mapping[str, object]) -> str:
    value: Final = scope.get("root_path", "")
    return value.rstrip("/") if isinstance(value, str) else ""


def mount_api_reference(app: FastAPI) -> None:
    docs_url: Final = app.docs_url
    openapi_url: Final = app.openapi_url
    if docs_url is None or openapi_url is None:
        return

    template: Final = (_ASSETS / "index.html").read_text(encoding="utf-8")
    lazy_tags: Final = tuple(lazy_tag_to_prefix())
    app.mount("/reference-assets", StaticFiles(directory=_ASSETS), name="reference-assets")

    async def api_reference(request: Request) -> HTMLResponse:
        root_path: Final = _root_path(request.scope)
        asset_base: Final = f"{root_path}/reference-assets"
        config: Final = json.dumps(
            {
                "title": app.title,
                "version": app.version,
                "apiBase": root_path,
                "referenceUrl": f"{root_path}/reference",
                "schemaUrl": f"{root_path}{openapi_url}",
                "docsUrl": f"{root_path}{docs_url}",
                "uiUrl": f"{root_path}/ui/",
                "assetBase": asset_base,
                "lazyTags": lazy_tags,
            }
        ).translate(_SCRIPT_ESCAPES)
        values: Final = {"TITLE": html.escape(app.title), "ASSET_BASE": html.escape(asset_base), "CONFIG": config}
        return HTMLResponse(_PLACEHOLDERS.sub(lambda match: values[match.group(1)], template))

    app.add_route("/reference", api_reference, methods=["GET"], include_in_schema=False)
