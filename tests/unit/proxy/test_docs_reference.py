import html
import re
from typing import Final

import httpx
import pytest
from fastapi import FastAPI
from pydantic import TypeAdapter

from litellm.proxy.docs_reference import mount_api_reference

_CONFIG: Final = TypeAdapter(dict[str, str | list[str]])
_OBJECT: Final = TypeAdapter(dict[str, object])


def _configuration(document: str) -> dict[str, str | list[str]]:
    script: Final = re.search(r'<script\b[^>]*\bid="reference-config"[^>]*>(.*?)</script>', document, flags=re.DOTALL)
    assert script is not None
    return _CONFIG.validate_json(script.group(1))


def _client(app: FastAPI, root_path: str = "") -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, root_path=root_path), base_url="http://proxy")


async def test_reference_uses_the_existing_schema_and_keeps_swagger_available() -> None:
    app: Final = FastAPI(docs_url="/", title="Gateway API", version="2.3.4")
    app.add_api_route("/visible", lambda: "visible")
    app.add_api_route("/private", lambda: "private", include_in_schema=False)
    mount_api_reference(app)

    async with _client(app) as client:
        page: Final = await client.get("/reference")
        config: Final = _configuration(page.text)
        schema_url: Final = config["schemaUrl"]
        assert isinstance(schema_url, str)
        schema: Final = await client.get(schema_url)
        swagger: Final = await client.get("/")
        stylesheet: Final = await client.get("/reference-assets/style.css")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert config["title"] == app.title
    assert config["version"] == app.version
    assert config["docsUrl"] == "/"
    assert config["apiBase"] == ""
    assert config["uiUrl"] == "/ui/"
    assert schema.status_code == 200
    schema_document: Final = _OBJECT.validate_json(schema.content)
    assert set(_OBJECT.validate_python(schema_document["paths"])) == {"/visible"}
    assert swagger.status_code == 200
    assert "SwaggerUIBundle" in swagger.text
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")


@pytest.mark.parametrize(("docs_url", "openapi_url"), ((None, "/openapi.json"), ("/", None), (None, None)))
async def test_disabled_docs_or_schema_do_not_expose_the_reference(
    docs_url: str | None, openapi_url: str | None
) -> None:
    app: Final = FastAPI(docs_url=docs_url, openapi_url=openapi_url)
    mount_api_reference(app)

    async with _client(app) as client:
        page: Final = await client.get("/reference")
        stylesheet: Final = await client.get("/reference-assets/style.css")

    assert page.status_code == 404
    assert stylesheet.status_code == 404


@pytest.mark.parametrize(("configured_root", "request_root"), (("/gateway", ""), ("", "/gateway")))
async def test_custom_schema_and_links_follow_the_request_root_path(configured_root: str, request_root: str) -> None:
    app: Final = FastAPI(root_path=configured_root, docs_url="/legacy-docs", openapi_url="/schema.json")
    mount_api_reference(app)

    async with _client(app, root_path=request_root) as client:
        page: Final = await client.get("/gateway/reference")
        stylesheet: Final = await client.get("/gateway/reference-assets/style.css")
        schema: Final = await client.get("/gateway/schema.json")

    assert page.status_code == 200
    config: Final = _configuration(page.text)
    assert config["apiBase"] == "/gateway"
    assert config["referenceUrl"] == "/gateway/reference"
    assert config["schemaUrl"] == "/gateway/schema.json"
    assert config["docsUrl"] == "/gateway/legacy-docs"
    assert config["uiUrl"] == "/gateway/ui/"
    assert config["assetBase"] == "/gateway/reference-assets"
    assert 'href="/gateway/reference-assets/style.css"' in page.text
    assert stylesheet.status_code == 200
    assert schema.status_code == 200


async def test_custom_title_stays_text_and_cannot_end_the_configuration_script() -> None:
    title: Final = '</script><script>alert("title")</script> & {{CONFIG}}'
    app: Final = FastAPI(title=title)
    mount_api_reference(app)

    async with _client(app) as client:
        page: Final = await client.get("/reference")

    assert page.status_code == 200
    assert _configuration(page.text)["title"] == title
    assert f"<title>{html.escape(title)}" in page.text
    assert title not in page.text
