"""
Regression tests: ``get_instance_fn`` refuses remote module loading
(``s3://``, ``gcs://``) when invoked without a ``config_file_path``.

The startup config-file load path passes ``config_file_path`` and is
unaffected — the documented ``litellm_settings.callbacks:
["s3://bucket/module.instance"]`` operator flow continues to work.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.proxy.types_utils.utils import get_instance_fn  # noqa: E402


@pytest.mark.parametrize("scheme", ["s3", "gcs"])
def test_remote_url_without_config_file_path_is_rejected(scheme):
    # The C1-Stage-B attack vector: admin endpoint receives an
    # s3:// / gcs:// instance specifier via the request body; no
    # ``config_file_path`` is in scope. Must refuse before the
    # ``exec_module`` sink is reached.
    with pytest.raises(ValueError, match="Remote module loading"):
        get_instance_fn(value=f"{scheme}://attacker-bucket/module.instance")


def test_remote_url_with_config_file_path_is_allowed():
    # Startup config-file load path: ``config_file_path`` is set, so
    # the gate doesn't fire. Documented operator feature must keep
    # working.
    with patch(
        "litellm.proxy.types_utils.utils._load_instance_from_remote_storage",
        return_value="loaded",
    ) as mock_loader:
        result = get_instance_fn(
            value="s3://my-bucket/m.inst",
            config_file_path="/etc/litellm/config.yaml",
        )

    assert result == "loaded"
    mock_loader.assert_called_once_with(
        "s3://my-bucket/m.inst", "/etc/litellm/config.yaml"
    )


def test_dotted_module_path_is_unaffected_by_gate():
    # Local dotted-name imports — the other branch of get_instance_fn —
    # have nothing to do with the remote-URL gate. Regression that the
    # gate doesn't accidentally affect them.
    with patch(
        "litellm.proxy.types_utils.utils.importlib.import_module"
    ) as mock_import:
        mock_module = type("M", (), {"my_instance": "loaded"})
        mock_import.return_value = mock_module

        result = get_instance_fn(value="my_module.my_instance")

    assert result == "loaded"
