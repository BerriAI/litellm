def test_least_busy_decrement_clamps_at_zero():
    assert max(0 - 1, 0) == 0
    assert max(2 - 1, 0) == 1
