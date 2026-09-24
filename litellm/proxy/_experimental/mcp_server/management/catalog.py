"""OpenAPI-driven tool catalog for the built-in management MCP server.

Every operation in the proxy's own ``app.openapi()`` spec becomes either a
management tool or a recorded exclusion. Data-plane, pass-through, public and
UI routes never become tools; mutations keep their real REST surface, so an
agent calling a tool sees exactly what the equivalent REST call would do.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import mcp.types as mcp_types
from pydantic import TypeAdapter

from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
    resolve_operation_params,
)
from litellm.proxy._types import LiteLLMRoutes

if TYPE_CHECKING:
    from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
        _OpenAPIComponents,  # pyright: ignore[reportPrivateUsage]  # spec-node types shared with the resolver
        _OpenAPIOperation,  # pyright: ignore[reportPrivateUsage]  # spec-node types shared with the resolver
        _OpenAPIPathItem,  # pyright: ignore[reportPrivateUsage]  # spec-node types shared with the resolver
    )

_METHODS: Final = ("get", "post", "put", "patch", "delete")

_EXCLUDED_ROUTE_GROUPS: Final = (
    frozenset(LiteLLMRoutes.llm_api_routes.value)
    .union(LiteLLMRoutes.mcp_routes.value, LiteLLMRoutes.ui_routes.value)
    .difference(LiteLLMRoutes.management_routes.value, LiteLLMRoutes.info_routes.value)
    .union(
        LiteLLMRoutes.public_routes.value,
        LiteLLMRoutes.apply_guardrail_routes.value,
        ("/apply_guardrail", "/usage/ai/chat"),
    )
)

_EXCLUDED_TAGS: Final = frozenset(
    {
        "llm_passthrough",
        "responses",
        "videos",
        "containers",
        "assistants",
        "files",
        "batch",
        "fine-tuning",
        "OpenAI Evals API",
        "OpenAI Evals API - Runs",
        "images",
        "realtime",
        "WebSocket",
        "OpenAI Pass-through",
        "pass-through",
        "google genai endpoints",
        "chat/completions",
        "completions",
        "embeddings",
        "audio",
        "rerank",
        "moderations",
        "ocr",
        "vector_stores",
        "search",
        "a2a",
        "gemini_agents",
        "interactions",
        "mcp_discoverable",
        "mcp_byok_oauth",
        "public",
        "health",
        "skills",
        "memory",
        "anthropic",
        "messages",
        "[beta] Anthropic",
        "llm utils",
        "generate content",
        "conversations",
        "Public Model Hub",
        "Public Model Hub Routes",
        "claude_code_marketplace",
    }
)

_EXCLUDED_PATH_PREFIXES: Final = (
    "/sso",
    "/login",
    "/logout",
    "/ui",
    "/health",
    "/.well-known",
    "/mcp",
    "/litellm-management",
    "/openai_passthrough",
    "/{provider}",
    "/v1/messages",
    "/messages",
    "/chat",
    "/v1/chat",
    "/openai",
    "/engines",
    "/gemini",
    "/vertex",
    "/bedrock",
    "/azure",
    "/anthropic",
    "/cohere",
    "/vllm",
    "/assemblyai",
    "/mistral",
    "/langfuse",
    "/eu.",
    "/oauth",
    "/api/",
    "/agents/",
    "/v1/agents/",
    "/a2a",
    "/v1/a2a",
    "/public/",
    "/memory",
    "/v1/memory",
    "/v1/evals",
)

_JSON_MEDIA_TYPE: Final = "application/json"
_SCHEMA_REF_PREFIX: Final = "#/components/schemas/"


_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])


def _json_object(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, object], value))  # cast-ok: spec JSON node  # mutable-ok: JSON object copy
    return {}  # mutable-ok: empty JSON object fallback


def _json_seq(value: object) -> tuple[object, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(cast(Sequence[object], value))  # cast-ok: spec JSON array
    return ()


@dataclass(frozen=True, slots=True)
class ManagementTool:
    name: str
    method: str
    path_template: str
    path_param_names: tuple[str, ...]
    query_param_names: tuple[str, ...]
    has_body: bool
    mcp_tool: mcp_types.Tool


@dataclass(frozen=True, slots=True)
class ManagementCatalog:
    tools: Mapping[str, ManagementTool]
    exclusions: Mapping[str, str]


def _params_object(parameters: Sequence[Mapping[str, object]], where: str) -> dict[str, object] | None:
    section: Final = tuple(p for p in parameters if p.get("in") == where and p.get("name"))
    if not section:
        return None
    properties: Final = {  # mutable-ok: JSON schema assembled for the tool descriptor
        str(p["name"]): _json_object(p.get("schema"))
        | {"description": p.get("description", "")}  # mutable-ok: rebuilt schema fragment for the MCP input schema
        for p in section
    }
    required: Final = tuple(str(p["name"]) for p in section if p.get("required"))
    return {  # mutable-ok: JSON schema assembled for the tool descriptor
        "type": "object",
        "properties": properties,
        "required": list(required),  # mutable-ok: JSON schema required is an array
        "additionalProperties": False,
    }


def _exclusion_reason(path: str, method: str, operation: Mapping[str, object]) -> str | None:
    if not operation.get("operationId"):
        return "no operationId"
    if path in _EXCLUDED_ROUTE_GROUPS:
        return "data plane, public or ui route group"
    if path.rstrip("/").endswith("/mcp"):
        return "MCP transport endpoint"
    if path.rstrip("/").endswith("/stream"):
        return "streaming endpoint"
    matching_tag: Final = next(
        iter(sorted(tag for tag in _json_seq(operation.get("tags")) if isinstance(tag, str) and tag in _EXCLUDED_TAGS)),
        None,
    )
    if matching_tag is not None:
        return f"tag:{matching_tag}"
    if path.startswith(_EXCLUDED_PATH_PREFIXES):
        return "excluded path prefix"
    request_body: Final = operation.get("requestBody")
    if isinstance(request_body, Mapping):
        body_map: Final = cast(  # cast-ok: spec JSON object
            Mapping[str, object], request_body
        )
        content: Final = _json_object(body_map.get("content"))
        if _JSON_MEDIA_TYPE not in content:
            return "non-JSON request body"
    elif method in ("post", "put", "patch"):
        return "undeclared request body"
    responses: Final = operation.get("responses")
    if isinstance(responses, Mapping):
        for response in cast(Mapping[str, object], responses).values():  # cast-ok: spec JSON object
            resp_map = _json_object(response)
            content_map = _json_object(resp_map.get("content"))
            for media_type in content_map:
                if media_type != _JSON_MEDIA_TYPE:
                    return "non-JSON response"
    return None


def _walk_schema_refs(node: object, found: list[str]) -> None:
    if isinstance(node, Mapping):
        node_map: Final = cast(Mapping[str, object], node)  # cast-ok: spec JSON object
        ref: Final = node_map.get("$ref")
        if isinstance(ref, str) and ref.startswith(_SCHEMA_REF_PREFIX):
            name = ref[len(_SCHEMA_REF_PREFIX) :]
            if name not in found:
                found.append(name)
        for value in node_map.values():
            _walk_schema_refs(value, found)
    elif isinstance(node, (list, tuple)):
        for item in cast(Sequence[object], node):  # cast-ok: spec JSON array
            _walk_schema_refs(item, found)


def _rewrite_schema_refs(node: object) -> object:
    if isinstance(node, Mapping):
        return {  # mutable-ok: rebuilt schema fragment for the MCP input schema
            key: (
                f"#/$defs/{value[len(_SCHEMA_REF_PREFIX) :]}"
                if key == "$ref" and isinstance(value, str) and value.startswith(_SCHEMA_REF_PREFIX)
                else _rewrite_schema_refs(value)
            )
            for key, value in cast(Mapping[str, object], node).items()  # cast-ok: spec JSON object
        }
    if isinstance(node, (list, tuple)):
        return [  # mutable-ok: rebuilt schema fragment for the MCP input schema
            _rewrite_schema_refs(item)
            for item in cast(Sequence[object], node)  # cast-ok: spec JSON array
        ]
    return node


def _collect_defs(fragments: Sequence[object], component_schemas: Mapping[str, object]) -> dict[str, object]:
    collected: Final[dict[str, object]] = {}  # mutable-ok: transitive $defs accumulator keyed by schema name
    pending: Final[list[str]] = []  # mutable-ok: depth-first work list over schema names
    for fragment in fragments:
        _walk_schema_refs(fragment, pending)
    while pending:
        schema_name = pending.pop()
        if schema_name in collected:
            continue
        definition = component_schemas.get(schema_name)
        if not isinstance(definition, Mapping):
            continue
        collected[schema_name] = _rewrite_schema_refs(cast(object, definition))  # cast-ok: erased spec node
        _walk_schema_refs(cast(object, definition), pending)  # cast-ok: erased spec node
    return collected


def _body_schema(request_body: Mapping[str, object]) -> dict[str, object]:
    content: Final = _json_object(request_body.get("content"))
    media: Final = _json_object(content.get(_JSON_MEDIA_TYPE))
    schema: Final = _json_object(media.get("schema"))
    return schema if schema else {"type": "object"}  # mutable-ok: JSON schema fragment for the tool descriptor


def _build_tool(
    name: str,
    method: str,
    path: str,
    path_item: Mapping[str, object],
    operation: Mapping[str, object],
    components: Mapping[str, object],
) -> ManagementTool:
    operation_typed: Final = cast(  # cast-ok: spec JSON node
        "_OpenAPIOperation",
        dict(operation),  # mutable-ok: node copy for resolver
    )
    path_item_typed: Final = cast(  # cast-ok: spec JSON node
        "_OpenAPIPathItem",
        dict(path_item),  # mutable-ok: node copy for resolver
    )
    components_typed: Final = cast(  # cast-ok: spec JSON node
        "_OpenAPIComponents",
        dict(components),  # mutable-ok: node copy for resolver
    )
    resolved: Final = resolve_operation_params(operation_typed, path_item_typed, components_typed)
    parameters: Final = tuple(resolved.get("parameters", ()))
    path_section: Final = _params_object(parameters, "path")
    query_section: Final = _params_object(parameters, "query")
    request_body: Final = operation.get("requestBody")
    has_body: Final = isinstance(request_body, Mapping)
    body_section: Final = (
        _body_schema(cast(Mapping[str, object], request_body))  # cast-ok: spec JSON object
        if isinstance(request_body, Mapping)
        else None
    )

    properties: Final[dict[str, object]] = {}  # mutable-ok: JSON schema assembled for the tool descriptor
    required: Final[list[str]] = []  # mutable-ok: JSON schema required is an array
    if path_section is not None:
        properties["path"] = path_section
        if path_section.get("required"):
            required.append("path")
    if query_section is not None:
        properties["query"] = query_section
        if query_section.get("required"):
            required.append("query")
    if body_section is not None:
        properties["body"] = body_section
        body_map: Final = cast(  # cast-ok: spec JSON object
            Mapping[str, object], request_body
        )
        if body_map.get("required"):
            required.append("body")

    fragments: Final = tuple(properties.values())
    component_schemas: Final = _json_object(components.get("schemas"))
    defs: Final = _collect_defs(fragments, component_schemas)
    schema: Final = {  # mutable-ok: JSON schema assembled for the tool descriptor
        "type": "object",
        "properties": {  # mutable-ok: JSON schema fragment for the tool descriptor
            key: _rewrite_schema_refs(value) for key, value in properties.items()
        },
        "required": required,
        "additionalProperties": False,
    }
    if defs:
        schema["$defs"] = defs  # mutable-ok: JSON schema assembled for the tool descriptor

    summary: Final = operation.get("summary")
    description: Final = operation.get("description")
    description_text: Final = ". ".join(str(part) for part in (summary, description) if part)
    mcp_tool: Final = mcp_types.Tool(
        name=name,
        title=str(summary or name),
        description=description_text or str(summary or name),
        input_schema=_OBJECT_ADAPTER.validate_python(schema),
        annotations=mcp_types.ToolAnnotations(
            read_only_hint=method == "get",
            destructive_hint=method != "get",
            idempotent_hint=method in ("get", "put", "delete"),
        ),
    )
    return ManagementTool(
        name=name,
        method=method.upper(),
        path_template=path,
        path_param_names=tuple(str(p["name"]) for p in parameters if p.get("in") == "path" and p.get("name")),
        query_param_names=tuple(str(p["name"]) for p in parameters if p.get("in") == "query" and p.get("name")),
        has_body=has_body,
        mcp_tool=mcp_tool,
    )


def build_catalog(spec: Mapping[str, object]) -> ManagementCatalog:
    """Turn the proxy's OpenAPI spec into the management tool catalog.

    Raises ValueError on a duplicate operationId: two REST operations behind
    one tool name would make dispatch ambiguous.
    """
    paths: Final = _json_object(spec.get("paths"))
    component_map: Final = _json_object(spec.get("components"))
    tools: Final[dict[str, ManagementTool]] = {}  # mutable-ok: catalog accumulator keyed by operationId
    exclusions: Final[dict[str, str]] = {}  # mutable-ok: exclusion accumulator keyed by "METHOD path"
    for path in sorted(paths):
        path_item = _json_object(paths[path])
        if not path_item:
            continue
        for method in _METHODS:
            raw_operation = path_item.get(method)
            if not isinstance(raw_operation, Mapping):
                continue
            operation = cast(Mapping[str, object], raw_operation)  # cast-ok: spec JSON object
            key = f"{method.upper()} {path}"
            reason = _exclusion_reason(path, method, operation)
            if reason is not None:
                exclusions[key] = reason
                continue
            name = str(operation["operationId"])
            if name in tools:
                raise ValueError(
                    f"duplicate OpenAPI operationId '{name}' at {key}; management MCP tool names must be unique"
                )
            tools[name] = _build_tool(name, method, path, path_item, operation, component_map)
    return ManagementCatalog(tools=MappingProxyType(tools), exclusions=MappingProxyType(exclusions))
