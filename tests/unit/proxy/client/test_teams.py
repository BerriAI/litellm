import time

import pytest
import requests
import responses

from litellm.proxy.client.exceptions import UnauthorizedError
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


@responses.activate
def test_list_v2_sends_filters_and_returns_the_page():
    client = TeamsManagementClient(base_url="http://localhost:8000", api_key="sk-test")
    page = {"teams": [{"team_id": "team-1"}], "total": 1, "page": 2, "page_size": 5, "total_pages": 1}
    responses.add(
        responses.GET,
        "http://localhost:8000/v2/team/list",
        json=page,
        status=200,
        match=[
            responses.matchers.query_param_matcher(
                {
                    "page": "2",
                    "page_size": "5",
                    "sort_order": "desc",
                    "user_id": "user-1",
                    "team_alias": "search",
                    "sort_by": "created_at",
                }
            )
        ],
    )

    result = client.list_v2(
        user_id="user-1", team_alias="search", page=2, page_size=5, sort_by="created_at", sort_order="desc"
    )

    assert result == page


@responses.activate
def test_list_v2_unauthorized_error():
    client = TeamsManagementClient(base_url="http://localhost:8000", api_key="sk-bad")
    responses.add(responses.GET, "http://localhost:8000/v2/team/list", status=401, json={"error": "Unauthorized"})

    with pytest.raises(UnauthorizedError):
        client.list_v2()
