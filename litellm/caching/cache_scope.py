import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.caching import CacheScope, LiteLLMCacheType

if TYPE_CHECKING:
    from litellm.caching.caching import Cache
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

_CACHE_KEY_OVERRIDES: Final = frozenset({"cache_key", "preset_cache_key"})


def apply_cache_scope(data: Mapping[str, object], auth: "UserAPIKeyAuth", cache: "Cache") -> Mapping[str, object]:
    if cache.cache_scope == CacheScope.GLOBAL or cache.type in (
        LiteLLMCacheType.REDIS_SEMANTIC,
        LiteLLMCacheType.VALKEY_SEMANTIC,
        LiteLLMCacheType.QDRANT_SEMANTIC,
    ):
        return data

    controls: Final = as_str_mapping(data.get("cache")) or {}
    params: Final = as_str_mapping(data.get("litellm_params"))
    stripped: Final = {key: value for key, value in data.items() if key not in _CACHE_KEY_OVERRIDES}
    request: Final = (
        {**stripped, "litellm_params": {key: value for key, value in params.items() if key not in _CACHE_KEY_OVERRIDES}}
        if params is not None
        else stripped
    )
    credential: Final = auth.token if auth.is_session_token else auth.api_key
    kind, identity = (
        ("team", auth.team_id) if cache.cache_scope == CacheScope.TEAM and auth.team_id else ("key", credential)
    )
    if not identity:
        return {**request, "cache": {**controls, "no-cache": True, "no-store": True}}

    metadata: Final = as_str_mapping(data.get("metadata")) or {}
    namespace_value: Final = controls.get("namespace") or metadata.get("redis_namespace") or cache.namespace
    namespace: Final = namespace_value if isinstance(namespace_value, str) else None
    material: Final = json.dumps((kind, identity, namespace), separators=(",", ":")).encode()
    scoped_namespace: Final = f"scope:{hashlib.sha256(material).hexdigest()}"
    return {
        **request,
        "cache": {**controls, "namespace": (f"{namespace}:{scoped_namespace}" if namespace else scoped_namespace)},
    }
