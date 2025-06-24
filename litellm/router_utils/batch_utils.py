import io
import json
from os import PathLike
from typing import Optional

from litellm.types.llms.openai import FileTypes, OpenAIFilesPurpose


class InMemoryFile(io.BytesIO):
    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name


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

        lines = file_content_str.splitlines()
        modified_lines = []
        for line in lines:
            # Parse each line as a JSON object
            json_object = json.loads(line.strip())

            # Replace the model name if it exists
            if "body" in json_object:
                json_object["body"]["model"] = new_model_name

            # Convert the modified JSON object back to a string
            modified_lines.append(json.dumps(json_object))

        # Reassemble the modified lines and return as bytes
        modified_file_content = "\n".join(modified_lines).encode("utf-8")
        return InMemoryFile(modified_file_content, name="modified_file.jsonl")  # type: ignore

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
        ["batch", "generic_api_call", "_acreate_batch", "file", "_ageneric_api_call_with_fallbacks"]
    )
    if function_name and any(
        method in function_name for method in ROUTER_METHODS_USING_LITELLM_METADATA
    ):
        return "litellm_metadata"
    else:
        return "metadata"
