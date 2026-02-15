import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.opik.opik import OpikLogger


def test_resolve_request_config_uses_static_values_by_default():
    logger = OpikLogger(
        project_name="static-project",
        url="https://static.opik/api",
        api_key="static-api-key",
        workspace="static-workspace",
    )

    project_name, base_url, headers, has_dynamic = logger._resolve_request_config({})

    assert project_name == "static-project"
    assert base_url == "https://static.opik/api"
    assert headers["authorization"] == "static-api-key"
    assert headers["Comet-Workspace"] == "static-workspace"
    assert has_dynamic is False


def test_resolve_request_config_applies_dynamic_opik_callback_vars():
    logger = OpikLogger(
        project_name="static-project",
        url="https://static.opik/api",
        api_key="static-api-key",
        workspace="static-workspace",
    )
    kwargs = {
        "standard_callback_dynamic_params": {
            "opik_api_key": "dynamic-api-key",
            "opik_workspace": "dynamic-workspace",
            "opik_project_name": "dynamic-project",
            "opik_url_override": "https://dynamic.opik/api",
        }
    }

    project_name, base_url, headers, has_dynamic = logger._resolve_request_config(kwargs)

    assert project_name == "dynamic-project"
    assert base_url == "https://dynamic.opik/api"
    assert headers["authorization"] == "dynamic-api-key"
    assert headers["Comet-Workspace"] == "dynamic-workspace"
    assert has_dynamic is True
