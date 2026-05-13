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


def test_pass_through_route_threads_config_file_path():
    # ``create_pass_through_route`` must forward ``config_file_path`` so
    # an operator with ``custom_handler: s3://...`` declared in
    # ``config.yaml`` still resolves at startup. Callers that omit it
    # (DB-overlay / runtime admin API) fall through to the gate.
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte

    # ``get_instance_fn`` is imported lazily inside the function — patch
    # at the source so the deferred import resolves to the mock.
    with patch(
        "litellm.proxy.types_utils.utils.get_instance_fn", return_value=object()
    ) as mock_get:
        pte.create_pass_through_route(
            endpoint="/x",
            target="s3://bucket/mod.inst",
            config_file_path="/etc/litellm/config.yaml",
        )

    mock_get.assert_called_once_with(
        value="s3://bucket/mod.inst",
        config_file_path="/etc/litellm/config.yaml",
    )


def test_mcp_tool_registry_threads_config_file_path():
    # MCP tool handlers declared in ``config.yaml`` mcp_tools[].handler
    # may legitimately be ``s3://...``; the YAML-load path must thread
    # ``config_file_path`` so they resolve.
    from litellm.proxy._experimental.mcp_server import tool_registry as tr

    fake_handler = lambda **kwargs: None  # noqa: E731 — registry requires callable
    with patch.object(tr, "get_instance_fn", return_value=fake_handler) as mock_get:
        registry = tr.MCPToolRegistry()
        registry.load_tools_from_config(
            mcp_tools_config=[
                {
                    "name": "tool_a",
                    "description": "d",
                    "handler": "s3://bucket/mod.handler",
                }
            ],
            config_file_path="/etc/litellm/config.yaml",
        )

    mock_get.assert_called_once_with(
        "s3://bucket/mod.handler", "/etc/litellm/config.yaml"
    )
