import io
import json
from typing import IO, Optional, Tuple, Union


class InMemoryFile(io.BytesIO):
    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name


def replace_model_in_jsonl(
    file_content: Union[bytes, IO, Tuple[str, bytes, str]], new_model_name: str
) -> Optional[InMemoryFile]:
    try:
        # Decode the bytes to a string and split into lines
        # If file_content is a file-like object, read the bytes
        if hasattr(file_content, "read"):
            file_content_bytes = file_content.read()  # type: ignore
        elif isinstance(file_content, tuple):
            file_content_bytes = file_content[1]
        else:
            file_content_bytes = file_content

        # Decode the bytes to a string and split into lines
        file_content_str = file_content_bytes.decode("utf-8")
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

    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as e:
        return None


def _get_router_metadata_variable_name(function_name) -> str:
    """
    Helper to return what the "metadata" field should be called in the request data

    For all /thread or /assistant endpoints we need to call this "litellm_metadata"

    For ALL other endpoints we call this "metadata
    """
    if "batch" in function_name:
        return "litellm_metadata"
    else:
        return "metadata"
