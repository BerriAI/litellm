"""The shared chat config contract."""

from typing import Final

import litellm


def test_default_merge_extra_body_shallow_merges_and_lets_extra_body_replace_nested_metadata() -> None:
    cfg: Final = litellm.OpenAIGPTConfig()
    request: Final = {"model": "m", "messages": [], "metadata": {"from_request": "1"}, "temperature": 0.2}
    extra_body: Final = {"metadata": {"from_extra_body": "2"}, "vendor_only_field": "x"}

    assert cfg.merge_extra_body(dict(request), extra_body) == {
        "model": "m",
        "messages": [],
        "metadata": {"from_extra_body": "2"},
        "temperature": 0.2,
        "vendor_only_field": "x",
    }
    assert cfg.merge_extra_body(dict(request), None) == request
    assert cfg.merge_extra_body(dict(request), {}) == request
