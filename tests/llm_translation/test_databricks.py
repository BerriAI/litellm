import pytest
import sys

import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.exceptions import BadRequestError


@pytest.mark.parametrize("set_base", [True, False])
def test_throws_if_api_base_or_api_key_not_set_without_databricks_sdk(
    monkeypatch, set_base
):
    # Simulate that the databricks SDK is not installed
    monkeypatch.setitem(sys.modules, "databricks.sdk", None)

    err_msg = ["the Databricks base URL and API key are not set", "Missing API Key"]

    if set_base:
        monkeypatch.setenv(
            "DATABRICKS_API_BASE",
            "https://my.workspace.cloud.databricks.com/serving-endpoints",
        )
        monkeypatch.delenv(
            "DATABRICKS_API_KEY",
        )
    else:
        monkeypatch.setenv("DATABRICKS_API_KEY", "dapimykey")
        monkeypatch.delenv(
            "DATABRICKS_API_BASE",
        )

    with pytest.raises(BadRequestError) as exc:
        litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages=[{"role": "user", "content": "How are you?"}],
        )
    assert any(msg in str(exc) for msg in err_msg)

    with pytest.raises(BadRequestError) as exc:
        litellm.embedding(
            model="databricks/bge-12312",
            input=["Hello", "World"],
        )
    assert any(msg in str(exc) for msg in err_msg)


