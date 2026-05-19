"""
`GET /openapi-public.json` — a SDK-friendly subset of the full OpenAPI spec.

S5-02: the proxy's full `/openapi.json` includes admin / management /
key-rotation routes that an SDK should never expose. This endpoint filters
the same in-memory schema down to the paths a downstream xct-* app
actually needs:

    /v1/capabilities
    /.well-known/xct-capabilities
    /v1/agents (GET only)
    /v1/agents/{agent_id} (GET only)
    /v1/agents/{agent_id}/.well-known/agent-card.json
    /v1/a2a/*
    /a2a/*
    /v1/mcp/tools
    /v1/mcp/access_groups
    /mcp-rest/*
    /v1/xct-skills (GET, GET-by-id only — write paths stay admin-side)
    /v1/chat/completions
    /v1/embeddings
    /v1/responses
    /v1/messages
    /v1/models
    /oauth/authorize
    /oauth/token
    /oauth/revoke
    /oauth/introspect

Operation IDs are preserved verbatim so codegen output is stable.
"""

from typing import Any, Dict, List, Set, Tuple

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


# Exact paths that pass through unchanged.
_PUBLIC_PATH_PREFIXES: Tuple[str, ...] = (
    "/v1/capabilities",
    "/.well-known/xct-capabilities",
    "/v1/a2a/",
    "/a2a/",
    "/v1/mcp/tools",
    "/v1/mcp/access_groups",
    "/mcp-rest/",
    "/v1/chat/",
    "/chat/",
    "/v1/completions",
    "/completions",
    "/v1/embeddings",
    "/embeddings",
    "/v1/responses",
    "/v1/messages",
    "/messages",
    "/v1/models",
    "/models",
    "/oauth/",
)

# Paths where only certain methods are exposed.
# value = set of allowed lowercase HTTP methods.
_PUBLIC_RESTRICTED_PATHS: Dict[str, Set[str]] = {
    # Agents — read-only on the SDK; write stays admin-side.
    "/v1/agents": {"get"},
    "/v1/agents/{agent_id}": {"get"},
    "/v1/agents/{agent_id}/.well-known/agent-card.json": {"get"},
    "/v1/agents/{agent_id}/versions": {"get"},
    # XCT skills — read-only via SDK; write/upload/publish stay admin-side.
    "/v1/xct-skills": {"get"},
    "/v1/xct-skills/{skill_id}": {"get"},
}


def _path_is_public(path: str, methods: Set[str]) -> Tuple[bool, Set[str]]:
    """Decide whether to keep `path`, and which methods on it."""
    # Restricted-paths take precedence so we don't accidentally over-share
    # via a prefix match.
    if path in _PUBLIC_RESTRICTED_PATHS:
        allowed = methods & _PUBLIC_RESTRICTED_PATHS[path]
        return (bool(allowed), allowed)
    for prefix in _PUBLIC_PATH_PREFIXES:
        if path.startswith(prefix):
            return (True, methods)
    return (False, set())


def _filter_openapi(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a NEW dict carrying only the public surface."""
    out: Dict[str, Any] = {
        "openapi": schema.get("openapi", "3.1.0"),
        "info": {
            **schema.get("info", {}),
            "title": "xct-litellm public API",
            "x-xct-doc": (
                "Filtered subset of /openapi.json — only the routes downstream "
                "apps (xct-chat, xct-home, xct-agent-desktop) consume. The full "
                "schema is at /openapi.json for proxy admins."
            ),
        },
        "paths": {},
        "components": schema.get("components") or {},
        # Pass servers + security through unmodified.
        **({"servers": schema["servers"]} if "servers" in schema else {}),
    }
    src_paths = schema.get("paths") or {}
    referenced_schemas: Set[str] = set()
    for path, ops in src_paths.items():
        if not isinstance(ops, dict):
            continue
        method_keys = {
            m
            for m in ops.keys()
            if m.lower()
            in {
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "head",
                "options",
            }
        }
        keep, allowed_methods = _path_is_public(path, method_keys)
        if not keep:
            continue
        kept_ops = {"parameters": ops["parameters"]} if "parameters" in ops else {}
        for m in allowed_methods:
            if m in ops:
                kept_ops[m] = ops[m]
                _collect_refs(ops[m], referenced_schemas)
        # Skip the path entirely if every method got filtered out.
        if any(
            k in kept_ops
            for k in ("get", "post", "put", "patch", "delete", "head", "options")
        ):
            out["paths"][path] = kept_ops

    # Optionally prune unreferenced schemas in components.schemas to keep
    # the codegen output lean. We only prune when we can prove a schema
    # isn't referenced — partial graph traversal is safer than a full one
    # because `oneOf`/`allOf` chains can hide refs.
    if referenced_schemas and isinstance(out["components"].get("schemas"), dict):
        all_schemas = out["components"]["schemas"]
        # First-pass: keep referenced. Second-pass: keep anything referenced
        # transitively by the kept ones. Fixpoint in <=4 iterations for our
        # schema depth.
        kept = set(referenced_schemas)
        for _ in range(4):
            grew = set()
            for name in list(kept):
                if name in all_schemas:
                    _collect_refs(all_schemas[name], grew)
            new = grew - kept
            if not new:
                break
            kept |= new
        out["components"]["schemas"] = {
            name: schema_obj for name, schema_obj in all_schemas.items() if name in kept
        }
    return out


def _collect_refs(node: Any, acc: Set[str]) -> None:
    """Recurse the node, recording `#/components/schemas/<Name>` refs."""
    if isinstance(node, dict):
        for k, v in node.items():
            if (
                k == "$ref"
                and isinstance(v, str)
                and v.startswith("#/components/schemas/")
            ):
                acc.add(v.rsplit("/", 1)[-1])
            else:
                _collect_refs(v, acc)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, acc)


@router.get("/openapi-public.json", include_in_schema=False)
async def public_openapi() -> JSONResponse:
    """Return the SDK-facing subset of /openapi.json."""
    # Local import — proxy_server is heavy and we don't want circular import.
    from litellm.proxy.proxy_server import get_openapi_schema

    schema = get_openapi_schema()
    return JSONResponse(_filter_openapi(schema))
