import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles

# def test_get_file_ids_and_decode_b64_to_unified_uid_from_messages():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles()
#     messages = [
#         {
#             "role": "user",
#             "content": [
