"""Unit coverage for SplitTransport path routing (is_control_plane_path).

Model-management calls (/model/new, /model/delete, /model/info) must go to the
control plane: the data-plane gateway does not serve management routes, so a
misrouted /model/new 404s and takes down every suite that registers deployments
at runtime (llm_translation, batches, access_control). /models must stay on the
data plane; it is the OpenAI-compatible list-models route, not a management
route.
"""

import pytest

from transport import HttpTransport, SplitTransport, is_control_plane_path


@pytest.mark.parametrize(
    "path",
    [
        "/model/new",
        "/model/delete",
        "/model/update",
        "/model/info",
        "/key/generate",
        "/budget/new",
        "/spend/logs",
        "/end_user/daily/activity",
        "/user/daily/activity",
        "/team/daily/activity",
        "/tag/daily/activity",
    ],
)
def test_management_routes_go_to_the_control_plane(path: str) -> None:
    assert is_control_plane_path(path), (
        f"{path} is a management route; sending it to the data plane 404s"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/models",
        "/v1/models",
        "/chat/completions",
        "/v1/messages",
        "/embeddings",
        "/anthropic/v1/messages",
    ],
)
def test_llm_routes_stay_on_the_data_plane(path: str) -> None:
    assert not is_control_plane_path(path), (
        f"{path} is an LLM route; it must go to the data plane"
    )


def test_transport_repr_never_leaks_the_master_key() -> None:
    """pytest prints fixture values (Gateway -> SplitTransport -> HttpTransport)
    verbatim into failure output, which ships to CI logs and log aggregators, so
    the repr chain must never carry the master key."""
    secret = "sk-e2e-repr-leak-canary"
    transport = HttpTransport(base_url="http://localhost:4000", master_key=secret)
    split = SplitTransport(data=transport, control=transport)
    assert secret not in repr(transport)
    assert secret not in repr(split)
