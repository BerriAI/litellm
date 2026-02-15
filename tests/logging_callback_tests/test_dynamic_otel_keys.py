import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    initialize_standard_callback_dynamic_params,
)


def test_dynamic_key_extraction_from_metadata():
    """
    Test extraction of langfuse keys from metadata in kwargs.
    This simulates a Proxy request where keys are passed in metadata.
    """
    kwargs = {
        "metadata": {
            "langfuse_public_key": "pk-test",
            "langfuse_secret_key": "sk-test",
            "langfuse_host": "https://test.langfuse.com",
            "opik_api_key": "opik-api-key",
            "opik_workspace": "opik-workspace",
            "opik_project_name": "opik-project",
            "opik_url_override": "https://www.comet.com/opik/api",
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-test"
    assert params.get("langfuse_secret_key") == "sk-test"
    assert params.get("langfuse_host") == "https://test.langfuse.com"
    assert params.get("opik_api_key") == "opik-api-key"
    assert params.get("opik_workspace") == "opik-workspace"
    assert params.get("opik_project_name") == "opik-project"
    assert params.get("opik_url_override") == "https://www.comet.com/opik/api"


def test_dynamic_key_extraction_from_litellm_params_metadata():
    """
    Test extraction of langfuse keys from litellm_params.metadata.
    """
    kwargs = {
        "litellm_params": {
            "metadata": {
                "langfuse_public_key": "pk-litellm",
                "langfuse_secret_key": "sk-litellm",
                "opik_api_key": "opik-api-key-litellm",
                "opik_workspace": "opik-workspace-litellm",
            }
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-litellm"
    assert params.get("langfuse_secret_key") == "sk-litellm"
    assert params.get("opik_api_key") == "opik-api-key-litellm"
    assert params.get("opik_workspace") == "opik-workspace-litellm"


if __name__ == "__main__":
    test_dynamic_key_extraction_from_metadata()
    test_dynamic_key_extraction_from_litellm_params_metadata()
