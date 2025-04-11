import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.caching import DualCache
from litellm.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles
from litellm.types.utils import SpecialEnums


def test_get_file_ids_and_decode_b64_to_unified_uid_from_messages():
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCxmYzdmMmVhNS0wZjUwLTQ5ZjYtODljMS03ZTZhNTRiMTIxMzg",
                    },
                },
            ],
        },
    ]
    file_ids = (
        proxy_managed_files.get_file_ids_and_decode_b64_to_unified_uid_from_messages(
            messages
        )
    )
    assert file_ids == [
        "litellm_proxy:application/pdf;unified_id,fc7f2ea5-0f50-49f6-89c1-7e6a54b12138"
    ]

    ## in place update
    assert messages[0]["content"][1]["file"]["file_id"].startswith(
        SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value
    )
