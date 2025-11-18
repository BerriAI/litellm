import base64
import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, cast

from fastapi import Request

from litellm.types.utils import SpecialEnums

if TYPE_CHECKING:
    from litellm.types.llms.openai import FileExpiresAfter


def _is_base64_encoded_unified_file_id(b64_uid: str) -> Union[str, Literal[False]]:
    # Ensure b64_uid is a string and not a mock object
    if not isinstance(b64_uid, str):
        return False
    # Add padding back if needed
    padded = b64_uid + "=" * (-len(b64_uid) % 4)
    # Decode from base64
    try:
        decoded = base64.urlsafe_b64decode(padded).decode()
        if decoded.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            return decoded
        else:
            return False
    except Exception:
        return False


def convert_b64_uid_to_unified_uid(b64_uid: str) -> str:
    is_base64_unified_file_id = _is_base64_encoded_unified_file_id(b64_uid)
    if is_base64_unified_file_id:
        return is_base64_unified_file_id
    else:
        return b64_uid


def get_models_from_unified_file_id(unified_file_id: str) -> List[str]:
    """
    Extract model names from unified file ID.

    Example:
    unified_file_id = "litellm_proxy:application/octet-stream;unified_id,c4843482-b176-4901-8292-7523fd0f2c6e;target_model_names,gpt-4o-mini,gemini-2.0-flash"
    returns: ["gpt-4o-mini", "gemini-2.0-flash"]
    """
    try:
        # Ensure unified_file_id is a string and not a mock object
        if not isinstance(unified_file_id, str):
            return []
        match = re.search(r"target_model_names,([^;]+)", unified_file_id)
        if match:
            # Split on comma and strip whitespace from each model name
            return [model.strip() for model in match.group(1).split(",")]
        return []
    except Exception:
        return []


def get_model_id_from_unified_batch_id(file_id: str) -> Optional[str]:
    """
    Get the model_id from the file_id

    Expected format: litellm_proxy;model_id:{};llm_batch_id:{};llm_output_file_id:{}
    """
    ## use regex to get the model_id from the file_id
    try:
        # Ensure file_id is a string and not a mock object
        if not isinstance(file_id, str):
            return None
        return file_id.split("model_id:")[1].split(";")[0]
    except Exception:
        return None


def get_batch_id_from_unified_batch_id(file_id: str) -> str:
    ## use regex to get the batch_id from the file_id
    # Ensure file_id is a string and not a mock object
    if not isinstance(file_id, str):
        return ""
    if "llm_batch_id" in file_id:
        return file_id.split("llm_batch_id:")[1].split(",")[0]
    else:
        return file_id.split("generic_response_id:")[1].split(",")[0]


async def parse_expires_after_from_request(
    expires_after: Optional[Union[str, Dict[str, Any]]], request: Request
) -> Optional["FileExpiresAfter"]:
    """
    Parse expires_after from request form data.
    
    Handles two formats:
    1. Direct form parameter: expires_after as JSON string
    2. Bracket notation: expires_after[anchor] and expires_after[seconds] (or expires_after[days])
    """
    expires_after_dict: Optional[Dict[str, Any]] = None
    
    if expires_after:
        try:
            if isinstance(expires_after, str):
                expires_after_dict = json.loads(expires_after)
            elif isinstance(expires_after, dict):
                expires_after_dict = expires_after
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"Invalid expires_after format. Expected JSON object, got: {expires_after}. Error: {str(e)}"
            )
    else:
        # Try to get expires_after from bracket notation form fields
        # The OpenAI SDK sends expires_after as separate form fields: expires_after[anchor] and expires_after[seconds]
        try:
            form_data = await request.form()
            form_dict = dict(form_data)            
            anchor = form_dict.get("expires_after[anchor]")
            seconds = form_dict.get("expires_after[seconds]")
            
            if anchor:
                expires_after_dict = {"anchor": anchor}
                if seconds:
                    expires_after_dict["seconds"] = int(seconds) if isinstance(seconds, str) else seconds  
        except Exception:
            # If form parsing fails, continue without expires_after
            pass
    
    if expires_after_dict is not None:
        from litellm.types.llms.openai import FileExpiresAfter
        return cast(FileExpiresAfter, expires_after_dict)
    return None
