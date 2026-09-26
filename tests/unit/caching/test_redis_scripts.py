from pathlib import Path

import pytest

from litellm.caching import _redis_scripts
from litellm.caching.redis_cache import RedisScriptClient

REPO_ROOT = Path(__file__).parents[3]


def test_generated_scripts_match_their_python_source():
    """litellm/caching/_redis_scripts.py is what redis_scripts/scripts.py compiles to.

    redis-lua-py is not a litellm dependency, so this skips unless it is installed. If it fails,
    regenerate from the repository root with:
    uv run --with redis-lua-py==0.13.0 python -m redis_lua_py generate redis_scripts.scripts --out litellm/caching/_redis_scripts.py
    """
    codegen = pytest.importorskip("redis_lua_py.codegen")
    codegen.check("redis_scripts.scripts", REPO_ROOT / "litellm" / "caching" / "_redis_scripts.py")


@pytest.mark.asyncio
async def test_a_script_run_through_the_client_reaches_async_register_script_as_keys_and_argv():
    """The adapter hands a script's Lua, KEYS and wire-encoded ARGV to async_register_script, which
    owns namespacing and the circuit breaker, and drops the client the generated wrapper passes back."""
    registered = []
    calls = []

    class _Cache:
        def async_register_script(self, script):
            registered.append(script)

            async def run(keys, args, client=None):
                calls.append((keys, args, client))
                return 1

            return run

    result = await _redis_scripts.pexpire_if_owner(
        RedisScriptClient(_Cache()), key="lock", token="owner", milliseconds=1500
    )

    assert result == 1
    assert registered == [_redis_scripts.PEXPIRE_IF_OWNER]
    assert calls == [(["lock"], ["owner", "1500"], None)]
