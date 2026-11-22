import os
import subprocess
import sys
import textwrap

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def test_importing_caching_does_not_require_redis():
    # redis is an optional dependency (extra_proxy), so the base SDK can be
    # installed without it. Selecting valkey-semantic needs redis, but merely
    # importing litellm.caching.caching must not, or `import litellm` breaks for
    # every base-SDK user. This runs in a subprocess with redis blocked so the
    # check is not polluted by redis already being imported in this session.
    code = textwrap.dedent("""
        import sys
        for name in ("redis", "redis.asyncio", "redis.commands",
                     "redis.commands.search"):
            sys.modules[name] = None
        import litellm.caching.caching  # must not import redis at module top
        from litellm.types.caching import LiteLLMCacheType
        assert LiteLLMCacheType.VALKEY_SEMANTIC == "valkey-semantic"
        print("ok")
        """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _REPO_ROOT},
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
