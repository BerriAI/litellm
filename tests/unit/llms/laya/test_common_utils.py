from collections.abc import Mapping

import pytest

from litellm.llms.laya.common_utils import laya_response_model


@pytest.mark.parametrize(
    ("routing", "requested", "expected"),
    [
        ({"model": "multilingual"}, "english", "multilingual"),
        (None, "english", "english"),
        ({"model": 42}, "english", "english"),
        (None, None, "unknown"),
    ],
)
def test_laya_identity_tracks_the_checkpoint_not_the_shared_agent_name(
    routing: Mapping[str, object] | None, requested: str | None, expected: str
) -> None:
    assert laya_response_model({"model": "laya-rl-agent", "routing": routing}, requested) == expected
