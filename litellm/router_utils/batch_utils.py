import io
import json
from os import PathLike
from typing import List, Optional

from litellm._logging import verbose_logger
from litellm.types.llms.openai import FileTypes, OpenAIFilesPurpose


class InMemoryFile(io.BytesIO):
    def __init__(
        self, content: bytes, name: str, content_type: str = "application/jsonl"
    ):
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
            verbose_logger.error(
                f"error parsing final buffer: {buffer[:100]}..., error: {e}"
            )
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

        # Decode the bytes to a string and split into lines
        # If file_content is a file-like object, read the bytes
        if hasattr(file_content, "read"):
            file_content_bytes = file_content.read()  # type: ignore
        elif isinstance(file_content, tuple):
            file_content_bytes = file_content[1]
        else:
            file_content_bytes = file_content

        # Decode the bytes to a string and split into lines
        if isinstance(file_content_bytes, bytes):
            file_content_str = file_content_bytes.decode("utf-8")
        elif isinstance(file_content_bytes, str):
            file_content_str = file_content_bytes
        else:

            return file_content

        # Parse JSONL properly, handling potential multiline JSON objects
        json_objects = parse_jsonl_with_embedded_newlines(file_content_str)

        # If no valid JSON objects were found, return the original content
        if len(json_objects) == 0:
            return file_content

        modified_lines = []
        for json_object in json_objects:
            # Replace the model name if it exists
            if "body" in json_object:
                json_object["body"]["model"] = new_model_name

            # Convert the modified JSON object back to a string
            modified_lines.append(json.dumps(json_object))

        # Reassemble the modified lines and return as bytes
        modified_file_content = "\n".join(modified_lines).encode("utf-8")

        return InMemoryFile(modified_file_content, name="modified_file.jsonl", content_type="application/jsonl")  # type: ignore

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
    if function_name and any(
        method in function_name for method in ROUTER_METHODS_USING_LITELLM_METADATA
    ):
        return "litellm_metadata"
    else:
        return "metadata"
