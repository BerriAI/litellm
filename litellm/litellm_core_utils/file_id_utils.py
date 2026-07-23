"""Lightweight helpers for LiteLLM model-embedded file and batch IDs."""

import base64
import re
from typing import Optional


def decode_model_from_file_id(encoded_id: str) -> Optional[str]:
    """Extract the model name from a model-embedded file or batch ID."""
    try:
        if not isinstance(encoded_id, str):
            return None

        if encoded_id.startswith("file-"):
            b64_part = encoded_id[5:]
        elif encoded_id.startswith("batch_"):
            b64_part = encoded_id[6:]
        else:
            b64_part = encoded_id

        padded = b64_part + "=" * (-len(b64_part) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode()
        if decoded.startswith("litellm:") and ";model," in decoded:
            match = re.search(r";model,([^;]+)", decoded)
            if match:
                return match.group(1).strip()
    except Exception:
        pass
    return None


def get_original_file_id(encoded_id: str) -> str:
    """Extract the provider ID from a model-embedded file or batch ID."""
    try:
        if not isinstance(encoded_id, str):
            return encoded_id

        if encoded_id.startswith("file-"):
            b64_part = encoded_id[5:]
        elif encoded_id.startswith("batch_"):
            b64_part = encoded_id[6:]
        else:
            b64_part = encoded_id

        padded = b64_part + "=" * (-len(b64_part) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode()
        if decoded.startswith("litellm:") and ";model," in decoded:
            match = re.search(r"litellm:([^;]+);model,", decoded)
            if match:
                return match.group(1)
    except Exception:
        pass
    return encoded_id


def is_model_embedded_id(file_id: str) -> bool:
    """Return whether a file or batch ID contains LiteLLM model metadata."""
    return decode_model_from_file_id(file_id) is not None
