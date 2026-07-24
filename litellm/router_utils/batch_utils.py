import io
import json
from os import PathLike
from typing import List, Optional

from litellm._logging import verbose_logger
from litellm.types.llms.openai import FileTypes, OpenAIFilesPurpose


class InMemoryFile(io.BytesIO):
    def __init__(self, content: bytes, name: str, content_type: str = "application/jsonl"):
        super().__init__(content)
        self.name = name
        self.content_type = content_type


def parse_jsonl_with_embedded_newlines(content: str) -> List[dict]:
    """
    Parse JSONL content that may contain JSON objects with embedded newlines in string values.

    Unlike splitlines(), this function properly handles cases where JSON string values
    contain literal newline characters, which would otherwise break simple line-based parsing.

    Args:
        content: The JSONL file content as a string

    Returns:
        List of parsed JSON objects

    Example:
        >>> content = '{"id":1,"msg":"Line 1\\nLine 2"}\\n{"id":2,"msg":"test"}'
        >>> parse_jsonl_with_embedded_newlines(content)
        [{"id":1,"msg":"Line 1\\nLine 2"}, {"id":2,"msg":"test"}]
    """
    json_objects = []
    buffer = ""

    for char in content:
        buffer += char
        if char == "\n":
            # Try to parse what we have so far
            try:
                json_object = json.loads(buffer.strip())
                json_objects.append(json_object)
                buffer = ""  # Reset buffer for next JSON object
            except json.JSONDecodeError:
                # Not a complete JSON object yet, keep accumulating
                continue

    # Handle any remaining content in buffer
    if buffer.strip():
        try:
            json_object = json.loads(buffer.strip())
            json_objects.append(json_object)
        except json.JSONDecodeError as e:
            verbose_logger.error(f"error parsing final buffer: {buffer[:100]}..., error: {e}")
            raise e

    return json_objects


def should_replace_model_in_jsonl(
    purpose: OpenAIFilesPurpose,
) -> bool:
    """
    Check if the model name should be replaced in the JSONL file for the deployment model name.

    Azure raises an error on create batch if the model name for deployment is not in the .jsonl.
    """
    if purpose == "batch":
        return True
    return False


def replace_model_in_jsonl(file_content: FileTypes, new_model_name: str) -> FileTypes:
    try:
        ## if pathlike, return the original file content
        if isinstance(file_content, PathLike):
            return file_content

        # Iterate the source line-by-line WITHOUT reading it all into memory. A
        # spooled upload handle (managed batches stream from it) is read straight
        # off its backing; bytes/str are wrapped so they iterate line-by-line.
        source = file_content[1] if isinstance(file_content, tuple) else file_content
        if hasattr(source, "read"):
            if hasattr(source, "seek"):
                try:
                    source.seek(0)  # type: ignore[attr-defined]
                except (OSError, ValueError):
                    pass
            line_iter: object = source
        elif isinstance(source, (bytes, bytearray)):
            line_iter = io.BytesIO(bytes(source))
        elif isinstance(source, str):
            line_iter = io.StringIO(source)
        else:
            return file_content

        # Rewrite one row at a time, writing straight into the output buffer
        # instead of holding every parsed row in a list. Peak memory stays at
        # ~one row plus the output rather than several full copies of the file,
        # which the managed-files path depends on (it re-runs this rewrite once
        # per target model). Lines are accumulated so JSON objects that span
        # multiple physical lines still parse. Streaming the handle also means
        # the model rewrite is actually applied to tuple-wrapped upload handles;
        # otherwise a restricted body.model would survive and bypass the batch
        # model allowlist (which validates the upload target alias).
        output = InMemoryFile(b"", name="modified_file.jsonl", content_type="application/jsonl")
        wrote_any = False
        buffer = ""
        for raw_line in line_iter:  # type: ignore[attr-defined]
            buffer += raw_line.decode("utf-8") if isinstance(raw_line, (bytes, bytearray)) else raw_line
            stripped = buffer.strip()
            if not stripped:
                buffer = ""
                continue
            try:
                json_object = json.loads(stripped)
            except json.JSONDecodeError:
                continue  # object not complete yet; keep accumulating
            if isinstance(json_object, dict) and isinstance(json_object.get("body"), dict):
                json_object["body"]["model"] = new_model_name
            output.write((("\n" if wrote_any else "") + json.dumps(json_object)).encode("utf-8"))
            wrote_any = True
            buffer = ""

        if buffer.strip():
            # A row never parsed (truncated/malformed, or it swallowed the rows
            # that followed it). Returning the partial `output` would silently
            # drop those rows; return the unchanged original so the provider
            # rejects the batch loudly instead of accepting a truncated one.
            verbose_logger.error(f"error parsing trailing batch content: {buffer[:100]}...")
            if hasattr(source, "seek"):
                try:
                    source.seek(0)  # type: ignore[attr-defined]
                except (OSError, ValueError):
                    pass
            return file_content

        # If no valid JSON objects were found, return the original content
        if not wrote_any:
            return file_content

        output.seek(0)
        return output  # type: ignore

    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        # return the original file content if there is an error replacing the model name
        return file_content


def _get_router_metadata_variable_name(function_name: Optional[str]) -> str:
    """
    Helper to return what the "metadata" field should be called in the request data

    For all /thread or /assistant endpoints we need to call this "litellm_metadata"

    For ALL other endpoints we call this "metadata
    """
    ROUTER_METHODS_USING_LITELLM_METADATA = set(
        [
            "batch",
            "generic_api_call",
            "_acreate_batch",
            "file",
            "_ageneric_api_call_with_fallbacks",
        ]
    )
    if function_name and any(method in function_name for method in ROUTER_METHODS_USING_LITELLM_METADATA):
        return "litellm_metadata"
    else:
        return "metadata"
