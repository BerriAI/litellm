"""Field accessors that read MCP SDK wire models across the 1.x/2.x rename.

The MCP Python SDK renamed its generated wire-model fields from camelCase to
snake_case in 2.0 (``inputSchema`` -> ``input_schema``, ``mimeType`` ->
``mime_type``). Construction keeps working on both lines because 2.x sets
``populate_by_name`` with a camelCase alias generator, so the camelCase keyword
is still accepted; only attribute *reads* have to branch on the SDK version.
"""

from pydantic import TypeAdapter, ValidationError

_MISSING = object()

_INPUT_SCHEMA_ADAPTER = TypeAdapter(dict[str, object])


def _read_renamed(model: object, snake_case: str, camel_case: str) -> object:
    value = getattr(model, snake_case, _MISSING)
    if value is not _MISSING:
        return value
    value = getattr(model, camel_case, _MISSING)
    if value is not _MISSING:
        return value
    raise AttributeError(
        f"{type(model).__name__} exposes neither {snake_case!r} nor {camel_case!r}; unsupported MCP SDK version"
    )


def mcp_tool_input_schema(tool: object) -> dict[str, object]:
    """The JSON Schema for an ``mcp.types.Tool``'s arguments."""
    schema = _read_renamed(tool, "input_schema", "inputSchema")
    try:
        return _INPUT_SCHEMA_ADAPTER.validate_python(schema)
    except ValidationError as exc:
        raise TypeError(f"MCP tool input schema must be a dict, got {type(schema).__name__}") from exc


def mcp_content_mime_type(content: object) -> str | None:
    """The MIME type of an MCP resource-contents or content block."""
    mime_type = _read_renamed(content, "mime_type", "mimeType")
    if mime_type is None:
        return None
    if isinstance(mime_type, str):
        return mime_type
    raise TypeError(f"MCP content mime type must be a str, got {type(mime_type).__name__}")
