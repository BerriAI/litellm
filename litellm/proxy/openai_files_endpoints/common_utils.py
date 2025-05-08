import base64
from typing import Literal, Union

from litellm.types.utils import SpecialEnums


def _is_base64_encoded_unified_file_id(b64_uid: str) -> Union[str, Literal[False]]:
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
