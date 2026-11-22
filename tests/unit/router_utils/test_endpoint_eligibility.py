from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.router_utils.endpoint_eligibility import (
    warn_on_unenforced_endpoint_flag,
)

_DEPLOYMENT: Final = TypeAdapter(list[dict[str, object]])


@pytest.mark.parametrize(
    "info",
    [
        {"enforce_supported_endpoints": True},
        {"enforce_supported_endpoints": True, "supported_endpoints": ["/v1/embeddings", None]},
        {"enforce_supported_endpoints": "true", "supported_endpoints": 7},
    ],
)
def test_a_flag_without_a_usable_endpoint_list_is_reported(info: Mapping[str, object]) -> None:
    warning: Final = warn_on_unenforced_endpoint_flag("embed-only", info)
    assert warning is not None
    assert "'embed-only'" in warning


@pytest.mark.parametrize(
    "info",
    [
        {"enforce_supported_endpoints": True, "supported_endpoints": ["/v1/embeddings"]},
        {"enforce_supported_endpoints": True, "supported_endpoints": "/v1/embeddings"},
        {"enforce_supported_endpoints": False},
        {"supported_endpoints": ["/v1/embeddings", None]},
    ],
)
def test_an_enforceable_or_unflagged_deployment_is_not_reported(info: Mapping[str, object]) -> None:
    assert warn_on_unenforced_endpoint_flag("embed-only", info) is None
