"""Test that litellm import time doesn't regress.

Ref https://github.com/BerriAI/litellm/issues/7605
"""

import subprocess
import sys

import pytest


def test_proxy_types_not_imported_eagerly():
    """Ensure proxy._types is not loaded during import litellm.

    Note: litellm_enterprise (if installed) may pull in proxy._types
    through its own imports. We skip in that case since the enterprise
    package is not part of the core import chain.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import litellm; import sys; "
                "enterprise = 'litellm_enterprise' in sys.modules; "
                "loaded = 'litellm.proxy._types' in sys.modules; "
                "print(f'{loaded},{enterprise}')"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    loaded, enterprise = result.stdout.strip().split(",")
    if enterprise == "True":
        pytest.skip("litellm_enterprise installed. Pulls in proxy._types independently")
    assert loaded == "False", (
        "litellm.proxy._types should not be loaded during import litellm"
    )


def test_import_safety():
    """Ensure `from litellm import *` still works after import changes."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from litellm import *; print('OK')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"from litellm import * failed: {result.stderr}"
    assert "OK" in result.stdout


def test_model_cost_loaded():
    """Ensure model_cost dict is populated from local backup at import time."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import litellm; print(len(litellm.model_cost))",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    count = int(result.stdout.strip())
    assert count > 100, f"model_cost should have >100 models, got {count}"
