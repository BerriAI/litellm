import json
from pathlib import Path
from typing import Final

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

SWAGGER_ASSET_DIR: Final = Path(__file__).with_name("swagger")


def _lazy_plugin_script() -> str:
    # On dropdown expand: one-time fetch to the prefix (triggers lazy load),
    # then spec re-download so real routes replace the stub. Raw JS (no
    # <script> tag) since it's injected inside the existing inline script.
    from litellm.proxy._lazy_features import lazy_tag_to_prefix

    return (
        "const TAG_TO_PREFIX = " + json.dumps(lazy_tag_to_prefix()) + ";"
        "const warmedTags = new Set();"
        "const LAZY_TAGS = new Set(Object.keys(TAG_TO_PREFIX));"
        "const hideStubRows = () => {"
        "document.querySelectorAll('.opblock').forEach(op => {"
        "const d = op.querySelector('.opblock-summary-description');"
        "if (d && LAZY_TAGS.has(d.textContent.trim())) op.style.display = 'none';"
        "});};"
        "const annotateLazyHeaders = () => {"
        "document.querySelectorAll('.opblock-tag').forEach(tagEl => {"
        "const m = (tagEl.id || '').match(/^operations-tag-(.+)$/);"
        "if (!m || !LAZY_TAGS.has(m[1])) return;"
        "const existing = tagEl.querySelector('.lazy-load-hint');"
        "if (warmedTags.has(m[1])) { if (existing) existing.remove(); return; }"
        "if (existing) return;"
        "const hint = document.createElement('small');"
        "hint.className = 'lazy-load-hint';"
        "hint.textContent = ' (expand to load routes)';"
        "hint.style.opacity = '0.6';"
        "hint.style.marginLeft = '6px';"
        "const target = tagEl.querySelector('a span') || tagEl.querySelector('span') || tagEl;"
        "target.appendChild(hint);"
        "});};"
        "setInterval(() => { hideStubRows(); annotateLazyHeaders(); }, 200);"
        "const LazyLoadPlugin = () => ({"
        "afterLoad:function(system){setTimeout(()=>{"
        "for(const tag of LAZY_TAGS)system.layoutActions.show(['operations-tag',tag],false);"
        "},200);},"
        "statePlugins:{layout:{wrapActions:{show:(ori,sys)=>(...args)=>{"
        "const thing=args[0];const shown=args[1];let tag=null;"
        "if(Array.isArray(thing)){for(const t of thing)if(TAG_TO_PREFIX[t])tag=t;}"
        "if(shown!==false&&tag&&!warmedTags.has(tag)){warmedTags.add(tag);"
        "fetch(LAZY_WARM_URL+tag,{method:'POST',credentials:'include'}).then(r=>r.json()).then(d=>{"
        "if(!d.paths||Object.keys(d.paths).length===0)return;"
        "const cur=sys.specSelectors.specJson().toJS();"
        "const merged={};let inserted=false;"
        "for(const k in (cur.paths||{})){"
        "if(k===d.stub_path){for(const nk in d.paths)merged[nk]=d.paths[nk];inserted=true;}"
        "else{merged[k]=cur.paths[k];}}"
        "if(!inserted)Object.assign(merged,d.paths);"
        "cur.paths=merged;"
        "cur.components=cur.components||{};"
        "cur.components.schemas=Object.assign(cur.components.schemas||{},(d.components||{}).schemas||{});"
        "sys.specActions.updateSpec(JSON.stringify(cur));"
        "}).catch(()=>{});}"
        "return ori(...args);}}}}});"
    )


def mount_swagger_ui(application: FastAPI, swagger_url: str | None, *, lazy_features: bool = False) -> None:
    openapi_url: Final = application.openapi_url
    if not swagger_url or not openapi_url:
        return
    swagger_static: Final = StaticFiles(directory=SWAGGER_ASSET_DIR)

    lazy_plugin: Final = _lazy_plugin_script() if lazy_features else None
    oauth2_redirect_url: Final = application.swagger_ui_oauth2_redirect_url

    async def swagger_ui_html(request: Request) -> HTMLResponse:
        scope_root_path: Final[object] = request.scope.get("root_path")
        root_path: Final = scope_root_path.rstrip("/") if isinstance(scope_root_path, str) else ""
        assets: Final = f"{root_path}/swagger"
        response: Final = get_swagger_ui_html(
            openapi_url=f"{root_path}{openapi_url}",
            title=f"{application.title} - Swagger UI",
            oauth2_redirect_url=f"{root_path}{oauth2_redirect_url}" if oauth2_redirect_url else None,
            init_oauth=application.swagger_ui_init_oauth,
            swagger_ui_parameters=application.swagger_ui_parameters,
            swagger_js_url=f"{assets}/swagger-ui-bundle.js",
            swagger_css_url=f"{assets}/swagger-ui.css",
            swagger_favicon_url=f"{assets}/favicon.png",
        )
        if lazy_plugin is None:
            return response
        warm_url: Final = json.dumps(f"{root_path}/lazy/warm/").replace("<", "\\u003c")
        body: Final = (
            bytes(response.body)
            .decode("utf-8")
            .replace(
                "const ui = SwaggerUIBundle({",
                f"const LAZY_WARM_URL = {warm_url};"
                + lazy_plugin
                + 'const ui = SwaggerUIBundle({plugins:[LazyLoadPlugin],tagsSorter:"alpha",',
                1,
            )
        )
        return HTMLResponse(content=body)

    async def swagger_ui_redirect(_request: Request) -> HTMLResponse:
        return get_swagger_ui_oauth2_redirect_html()

    application.add_route(swagger_url, swagger_ui_html, include_in_schema=False)
    if oauth2_redirect_url:
        application.add_route(oauth2_redirect_url, swagger_ui_redirect, include_in_schema=False)
    application.mount("/swagger", swagger_static, name="swagger")
