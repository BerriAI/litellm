import time

import pytest
import requests

from litellm.proxy.client.teams import TeamsManagementClient


def test_list_gives_up_at_the_timeout_instead_of_hanging(hanging_server):
    """
    A proxy that accepts the connection but never answers used to pin the caller's
    process forever, since the request carried no timeout at all.
    """
    client = TeamsManagementClient(base_url=hanging_server, api_key="sk-test", timeout=1)

    started = time.monotonic()
    with pytest.raises(requests.exceptions.Timeout):
        client.list()

    assert time.monotonic() - started < 10
