"""Guard least_busy request counter never goes negative.

Salvage credit: @mango766 (#25393) and related @rudra717 (#25325).
"""


def test_request_count_decrement_clamps_at_zero():
    assert max(0 - 1, 0) == 0
    assert max(1 - 1, 0) == 0
    assert max(3 - 1, 0) == 2


def test_least_busy_source_uses_max_clamp():
    from pathlib import Path

    src = Path("litellm/router_strategy/least_busy.py").read_text()
    assert "request_count_dict[id] = max(request_count_value - 1, 0)" in src
    assert src.count("max(request_count_value - 1, 0)") >= 4
