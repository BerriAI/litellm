from collections.abc import Mapping
from typing import Final

import pytest

from litellm.router_utils.default_litellm_params import params_sent_to_deployment

_ROUTER_DEFAULTS: Final = {"vertex_location": "us-east5", "use_in_pass_through": True, "timeout": None}


@pytest.mark.parametrize(
    ("deployment", "request_params", "sent"),
    [
        ({"vertex_location": "eu"}, {}, {"vertex_location": "eu", "use_in_pass_through": True}),
        ({}, {}, {"vertex_location": "us-east5", "use_in_pass_through": True}),
        ({"vertex_location": None}, {}, {"vertex_location": "us-east5", "use_in_pass_through": True}),
        ({"vertex_location": "eu"}, {"vertex_location": "asia-northeast1"}, {"vertex_location": "asia-northeast1"}),
        ({"use_in_pass_through": False}, {}, {"vertex_location": "us-east5", "use_in_pass_through": True}),
    ],
    ids=[
        "deployment-over-router-default",
        "router-default-fills-unset",
        "router-default-fills-none",
        "request-over-both",
        "deployment-field-default-counts-as-unset",
    ],
)
def test_params_sent_to_deployment_prefer_the_request_then_the_deployment_then_the_router_default(
    deployment: Mapping[str, object], request_params: Mapping[str, object], sent: Mapping[str, object]
) -> None:
    merged: Final = params_sent_to_deployment(deployment, _ROUTER_DEFAULTS, request_params)

    assert {key: merged[key] for key in sent} == sent
    assert "timeout" not in merged
