# What is this?
## Unit tests for user_api_key_auth helper functions

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

import litellm


class Request:
    def __init__(self, client_ip: Optional[str] = None):
        self.client = MagicMock()
        self.client.host = client_ip


@pytest.mark.parametrize(
    "allowed_ips, client_ip, expected_result",
    [
        (None, "127.0.0.1", True),  # No IP restrictions, should be allowed
        (["127.0.0.1"], "127.0.0.1", True),  # IP in allowed list
        (["192.168.1.1"], "127.0.0.1", False),  # IP not in allowed list
        ([], "127.0.0.1", False),  # Empty allowed list, no IP should be allowed
        (["192.168.1.1", "10.0.0.1"], "10.0.0.1", True),  # IP in allowed list
        (
            ["192.168.1.1"],
            None,
            False,
        ),  # Request with no client IP should not be allowed
    ],
)
def test_check_valid_ip(
    allowed_ips: Optional[List[str]], client_ip: Optional[str], expected_result: bool
):
    from litellm.proxy.auth.user_api_key_auth import _check_valid_ip

    request = Request(client_ip)

    assert _check_valid_ip(allowed_ips, request) == expected_result  # type: ignore
