"""Stand-ins for the generated Redis scripts, for tests that fake Redis itself."""

from collections.abc import Awaitable, Callable

from litellm.caching import _redis_scripts
from litellm.caching.redis_cache import RedisScriptClient


def fake_scripts(**runners: Callable[..., Awaitable[object]]) -> RedisScriptClient:
    """A script client whose scripts run the given fakes instead of Redis.

    Each keyword is a script name from ``litellm.caching._redis_scripts`` and each value is
    called as ``run(keys=..., args=...)`` with the KEYS and ARGV the script would get: keys
    and arguments are strings, as on the wire. A script with no fake raises KeyError.
    """
    by_lua = {getattr(_redis_scripts, name.upper()): run for name, run in runners.items()}

    class _Registry:
        def async_register_script(self, script: str) -> Callable[..., Awaitable[object]]:
            return by_lua[script]

    return RedisScriptClient(_Registry())
