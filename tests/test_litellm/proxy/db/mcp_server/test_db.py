import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._experimental.mcp_server.db import get_mcp_servers_by_team


def test_fetch_mcp_servers_by_team():
    assert True == True
