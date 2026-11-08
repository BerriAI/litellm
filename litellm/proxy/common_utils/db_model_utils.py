from collections import deque
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import BaseModel, TypeAdapter

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_SEQUENCE: Final = TypeAdapter(Sequence[object])


def decrypt_db_model_params(litellm_params: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key: decrypt_value_helper(value=value, key=key, return_original_value=True) if isinstance(value, str) else value
        for key, value in litellm_params.items()
    }


def reencrypt_db_model_params(litellm_params: Mapping[str, object], new_key: str) -> Mapping[str, object]:
    return {
        key: encrypt_value_helper(value=value, new_encryption_key=new_key) if isinstance(value, str) else value
        for key, value in decrypt_db_model_params(litellm_params).items()
    }


def team_model_params_are_allowed(
    litellm_params: Mapping[str, object], model_info: Mapping[str, object] | BaseModel | None
) -> bool:
    info: Final = model_info.model_dump() if isinstance(model_info, BaseModel) else model_info
    if info is None or info.get("team_id") is None:
        return True

    # A private queue keeps deep traversal linear without recursion or input mutation
    pending: Final = deque[object]((litellm_params,))
    while pending:
        match pending.popleft():
            case str() as value if value.startswith("os.environ/"):
                return False
            case Mapping() as value:
                pending.extend(_MAPPING.validate_python(value).values())
            case list() as value:
                pending.extend(_SEQUENCE.validate_python(value))
            case _:
                pass
    return True
