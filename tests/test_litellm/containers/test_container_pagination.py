import pytest
from fastapi import HTTPException, Request

from litellm.proxy.container_endpoints.request_utils import (
    get_container_list_query_params,
)


def _request(query: str) -> Request:
    return Request(
        {
            "type": "http",
            "query_string": query.encode(),
        }
    )


def test_get_container_list_query_params_returns_provider_kwargs():
    assert get_container_list_query_params(_request("after=cursor&limit=20&order=desc")) == {
        "after": "cursor",
        "limit": 20,
        "order": "desc",
    }


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "limit=invalid", "order=middle"])
def test_get_container_list_query_params_rejects_invalid_values(query: str):
    with pytest.raises(HTTPException) as exc_info:
        get_container_list_query_params(_request(query))

    assert exc_info.value.status_code == 422
