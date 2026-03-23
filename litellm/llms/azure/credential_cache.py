"""
Module-level cache for Azure AD token provider callables.

Provides a shared cache used by get_azure_ad_token_from_entra_id and
get_azure_ad_token_from_username_password so that repeated calls with the same
credentials return the same provider callable rather than constructing a new one
each time.

Cache keys are tuples:
  ("entra", tenant_id, client_id, hmac(client_secret), scope)
  ("upw",   client_id, username,  hmac(password),      scope)

where hmac uses HMAC-SHA256 with a module-level key to avoid storing plaintext
credentials in cache keys.

If cachetools is available, the cache is a TTLCache(maxsize=128, ttl=3600) protected
by _cache_lock. Otherwise, it falls back to an unbounded dict with no TTL eviction.
"""

import hashlib
import hmac
import threading
from typing import Any

_CACHE_MAX_SIZE = 128
_CACHE_TTL_SECONDS = 3600  # 1 hour

# Deterministic HMAC key for cache keys — prevents raw credential strings from
# appearing verbatim in heap dumps, but is NOT a secret value and does NOT
# provide cryptographic confidentiality. Do not rely on this for security.
_CREDENTIAL_CACHE_HMAC_KEY = hashlib.sha256(
    b"litellm-azure-credential-cache-v1"
).digest()

_cache_lock = threading.Lock()

try:
    from cachetools import TTLCache

    _provider_cache: Any = TTLCache(maxsize=_CACHE_MAX_SIZE, ttl=_CACHE_TTL_SECONDS)
except ImportError:
    _provider_cache = {}  # unbounded fallback; install cachetools for TTL+LRU eviction


def _hash_secret(secret: str) -> str:
    # codeql[py/weak-cryptographic-algorithm]
    return hmac.new(
        _CREDENTIAL_CACHE_HMAC_KEY, secret.encode(), hashlib.sha256
    ).hexdigest()
