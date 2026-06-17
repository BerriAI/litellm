"""
Plugin proxy routes for litellm.

Enables external services to register as plugins and be proxied through
the litellm proxy server.

Config (in litellm config.yaml general_settings):
  plugins:
    - name: agent-control-plane
      url: "http://localhost:3210"
      display_name: "Agent Control Plane"
      plugin_key: "sk-..."   # optional: plugin's own auth key
"""

from fastapi import APIRouter, Request, Response

from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

router = APIRouter()

# In-memory plugin registry — populated from general_settings at startup
_plugin_registry: dict = {}


def register_plugins_from_config(general_settings: dict) -> None:
    """Call at startup to load plugin config from general_settings."""
    plugins = general_settings.get("plugins", [])
    for plugin in plugins:
        name = plugin.get("name")
        if name:
            _plugin_registry[name] = plugin


@router.get("/api/plugins", tags=["plugins"])
async def list_plugins(request: Request) -> list:
    """Return registered plugins for UI discovery.

    plugin_key is only returned to authenticated requests.
    """
    authed = bool(request.headers.get("Authorization") or request.cookies.get("token"))
    result = []
    for name, plugin in _plugin_registry.items():
        entry: dict = {
            "name": name,
            "display_name": plugin.get("display_name", name),
            "url": plugin.get("url", ""),
        }
        if authed and plugin.get("plugin_key"):
            entry["plugin_key"] = plugin["plugin_key"]
        result.append(entry)
    return result


@router.api_route(
    "/plugin-proxy/{plugin_name}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    tags=["plugins"],
    include_in_schema=False,
)
async def plugin_proxy(plugin_name: str, path: str, request: Request) -> Response:
    """Reverse-proxy any request to the registered plugin's backend."""
    plugin = _plugin_registry.get(plugin_name)
    if not plugin:
        return Response(
            content=f"Plugin '{plugin_name}' not registered",
            status_code=404,
        )

    target_url = f"{plugin['url'].rstrip('/')}/{path}"
    query = request.url.query
    if query:
        target_url = f"{target_url}?{query}"

    body = await request.body()

    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower()
        not in {
            "host",
            "connection",
            "transfer-encoding",
            "te",
            "trailers",
            "upgrade",
        }
    }

    handler = get_async_httpx_client(llm_provider=None)
    try:
        req = handler.client.build_request(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            content=body,
        )
        resp = await handler.client.send(req, follow_redirects=True)
    except Exception:
        return Response(
            content=f"Cannot connect to plugin '{plugin_name}' at {plugin['url']}",
            status_code=502,
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )
