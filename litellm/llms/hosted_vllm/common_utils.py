import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

HOSTED_VLLM_CACHE_SALT_PARAM: Final = "hosted_vllm_cache_salt"
CACHE_SALT_FIELD: Final = "cache_salt"
_CACHE_SALT_DOMAIN: Final = "litellm/hosted_vllm/cache_salt/v1"


def _caller_key_hash(metadata: object) -> str | None:
    key_hash: Final = metadata.get("user_api_key_hash") if isinstance(metadata, Mapping) else None
    return key_hash if isinstance(key_hash, str) and key_hash else None


def get_cache_salt_caller(
    litellm_params: Mapping[str, object], metadata_keys: tuple[str, ...] = ("metadata", "litellm_metadata")
) -> str | None:
    """The calling virtual key's hash on a deployment opted in to `hosted_vllm_cache_salt`, else None."""
    if litellm_params.get(HOSTED_VLLM_CACHE_SALT_PARAM) is not True:
        return None
    return next(
        (key_hash for key in metadata_keys if (key_hash := _caller_key_hash(litellm_params.get(key))) is not None),
        None,
    )


def derive_cache_salt(caller: str, caller_salt: object) -> str:
    """vLLM `cache_salt` scoped to the calling key.

    vLLM folds the salt into the first KV block's hash, so only requests with an equal salt share
    prefix-cache hits; that closes the timing side channel where a fast first token reveals another
    tenant's cached prompt (vLLM RFC #16016, CVE-2025-46570). The value is stable per key, so prefix
    reuse within a key (and across proxy replicas) is kept. A salt the caller sent is mixed in rather
    than forwarded, so a caller can split its own cache further (e.g. per end user) but can never
    submit the value another key's requests use.
    """
    sub: Final = "" if caller_salt is None else caller_salt if isinstance(caller_salt, str) else json.dumps(caller_salt)
    return hashlib.sha256(f"{_CACHE_SALT_DOMAIN}\0{caller}\0{sub}".encode()).hexdigest()


def with_caller_cache_salt(extra_body: Mapping[str, object], caller: str | None) -> Mapping[str, object]:
    if caller is None or CACHE_SALT_FIELD not in extra_body:
        return extra_body
    return MappingProxyType({**extra_body, CACHE_SALT_FIELD: derive_cache_salt(caller, extra_body[CACHE_SALT_FIELD])})
