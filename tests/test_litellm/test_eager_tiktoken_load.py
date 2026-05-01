"""
Test for LITELLM_DISABLE_LAZY_LOADING environment variable.

This test verifies that when LITELLM_DISABLE_LAZY_LOADING is set,
encoding is loaded at import time (pre-#18070 behavior) instead of lazy loading.

This addresses issue #18659: VCR cassette creation broken by lazy loading.
For now, this only affects encoding as it was the only reported issue.

Tests that need to clear sys.modules and re-import litellm run in subprocesses
to avoid contaminating the test process's module graph (which breaks mock.patch
for all subsequent tests on the same xdist worker).
"""

import subprocess
import sys
import textwrap

import pytest


def _run_python(
    script: str, env_override: dict | None = None
) -> subprocess.CompletedProcess:
    """Run a Python script in a subprocess and return the result."""
    import os

    env = os.environ.copy()
    # Remove the var so each test controls it explicitly
    env.pop("LITELLM_DISABLE_LAZY_LOADING", None)
    env.pop("TIKTOKEN_CACHE_DIR", None)
    if env_override:
        env.update(env_override)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        # Importing litellm can cold-load tiktoken/tokenizer assets and is
        # occasionally slow on CI runners; these tests validate behavior, not speed.
        timeout=180,
    )


def test_eager_loading_enabled():
    """Test that encoding is loaded at import time when env var is set"""
    result = _run_python(
        """
        import litellm
        assert hasattr(litellm, "encoding"), "Encoding should be available when eager loading is enabled"
        encoding = litellm.encoding
        assert encoding is not None, "Encoding should not be None"
        tokens = encoding.encode("Hello, world!")
        assert len(tokens) > 0, "Encoding should work"
        """,
        env_override={"LITELLM_DISABLE_LAZY_LOADING": "1"},
    )
    assert (
        result.returncode == 0
    ), f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_eager_loading_env_var_values():
    """Test that various truthy env var values all enable eager loading.

    All values are tested inside a single subprocess to avoid spawning one
    cold ``import litellm`` process per value (~78 s each on CI).  The
    subprocess re-imports litellm in isolated ``importlib`` reloads so each
    value gets a fresh module, but we only pay the process-start cost once.
    """
    result = _run_python(
        """
        import importlib, sys, os

        values = ["1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"]
        for value in values:
            # Set the env var for this iteration
            os.environ["LITELLM_DISABLE_LAZY_LOADING"] = value
            # Remove cached litellm modules so re-import picks up the new env
            mods_to_remove = [k for k in sys.modules if k == "litellm" or k.startswith("litellm.")]
            for m in mods_to_remove:
                del sys.modules[m]
            import litellm
            assert hasattr(litellm, "encoding"), f"Encoding missing for {value!r}"
            tokens = litellm.encoding.encode("test")
            assert len(tokens) > 0, f"Encoding broken for {value!r}"
        """,
        env_override={"LITELLM_DISABLE_LAZY_LOADING": "1"},
    )
    assert (
        result.returncode == 0
    ), f"Failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_lazy_loading_default():
    """Test that encoding is lazy loaded by default (when env var is not set)"""
    result = _run_python(
        """
        import litellm
        # Encoding should be accessible via __getattr__ (lazy loading)
        encoding = litellm.encoding
        tokens = encoding.encode("Hello, world!")
        assert len(tokens) > 0, "Encoding should work"
        """,
    )
    assert (
        result.returncode == 0
    ), f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_tiktoken_cache_dir_set_on_lazy_load():
    """Test that TIKTOKEN_CACHE_DIR is set when encoding is lazy loaded.

    This ensures the local tiktoken cache is used instead of downloading
    from the internet. Regression test for issue #19768.
    """
    result = _run_python(
        """
        import os
        import litellm
        # Access encoding (triggers lazy load)
        _ = litellm.encoding
        assert "TIKTOKEN_CACHE_DIR" in os.environ, "TIKTOKEN_CACHE_DIR should be set after lazy loading encoding"
        cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
        assert "tokenizers" in cache_dir, f"TIKTOKEN_CACHE_DIR should point to tokenizers directory, got: {cache_dir}"
        """,
    )
    assert (
        result.returncode == 0
    ), f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
