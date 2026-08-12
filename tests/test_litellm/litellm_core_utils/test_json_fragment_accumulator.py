import json
import time
from unittest.mock import patch

from litellm.litellm_core_utils.json_fragment_accumulator import JSONFragmentAccumulator


def test_initial_state_is_empty():
    accumulator = JSONFragmentAccumulator()
    assert not accumulator
    assert accumulator.could_close_json() is False
    assert accumulator.snapshot() == ""


def test_could_close_json_true_only_when_last_fragment_closes_a_value():
    accumulator = JSONFragmentAccumulator()
    accumulator.append('{"a": ')
    assert accumulator.could_close_json() is False

    accumulator.append("1}")
    assert accumulator.could_close_json() is True


def test_could_close_json_looks_past_trailing_blank_fragments():
    """A whitespace-only or empty fragment (e.g. the flush call at end of
    stream) must not mask a real closing byte in an earlier fragment."""
    accumulator = JSONFragmentAccumulator()
    accumulator.append('{"a": 1}')
    accumulator.append("")
    accumulator.append("   \n")
    assert accumulator.could_close_json() is True


def test_pop_next_value_on_empty_buffer_returns_false_without_touching_state():
    accumulator = JSONFragmentAccumulator()
    found, value = accumulator.pop_next_value()
    assert found is False
    assert value is None


def test_pop_next_value_on_incomplete_buffer_leaves_buffer_untouched():
    accumulator = JSONFragmentAccumulator()
    accumulator.append('{"candidates": [{"content":')

    found, value = accumulator.pop_next_value()

    assert found is False
    assert value is None
    assert accumulator.snapshot() == '{"candidates": [{"content":'


def test_pop_next_value_decodes_single_complete_object_and_clears_buffer():
    accumulator = JSONFragmentAccumulator()
    accumulator.append('{"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}')

    found, value = accumulator.pop_next_value()

    assert found is True
    assert value == {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
    assert accumulator.snapshot() == ""
    assert not accumulator


def test_pop_next_value_reassembles_a_value_split_across_many_fragments():
    obj = {"candidates": [{"content": {"parts": [{"text": "x" * 5000}]}}]}
    blob = json.dumps(obj)
    fragments = [blob[i : i + 37] for i in range(0, len(blob), 37)]
    assert len(fragments) > 10, "need a genuinely multi-fragment payload"

    accumulator = JSONFragmentAccumulator()
    found = False
    value = None
    for fragment in fragments:
        accumulator.append(fragment)
        if accumulator.could_close_json():
            found, value = accumulator.pop_next_value()

    assert found is True
    assert value == obj


def test_pop_next_value_peels_one_value_and_keeps_remainder():
    """Two concatenated envelopes in the buffer must both surface, one per
    call, instead of json.loads's "Extra data" failure wedging the buffer."""
    obj = '{"a": 1}'
    accumulator = JSONFragmentAccumulator()
    accumulator.append(obj + obj)

    first_found, first_value = accumulator.pop_next_value()
    assert first_found is True
    assert first_value == {"a": 1}
    assert accumulator.snapshot() == obj, "second value must remain buffered"

    second_found, second_value = accumulator.pop_next_value()
    assert second_found is True
    assert second_value == {"a": 1}
    assert not accumulator


def test_pop_next_value_advances_past_a_non_dict_leading_value():
    accumulator = JSONFragmentAccumulator()
    accumulator.append("[1, 2]" + '{"a": 1}')

    first_found, first_value = accumulator.pop_next_value()
    assert first_found is True
    assert first_value == [1, 2]

    second_found, second_value = accumulator.pop_next_value()
    assert second_found is True
    assert second_value == {"a": 1}


def test_set_and_snapshot_roundtrip():
    accumulator = JSONFragmentAccumulator()
    accumulator.set('{"a": 1}')
    assert accumulator.snapshot() == '{"a": 1}'
    assert accumulator

    accumulator.set("")
    assert accumulator.snapshot() == ""
    assert not accumulator


def test_append_never_calls_raw_decode():
    """Appending must be O(1) bookkeeping only; the O(n) join+decode is
    deferred entirely to pop_next_value."""
    accumulator = JSONFragmentAccumulator()
    with patch.object(json.JSONDecoder, "raw_decode", autospec=True, side_effect=json.JSONDecoder.raw_decode) as spy:
        for fragment in ['{"a":', " 1", "}"]:
            accumulator.append(fragment)
        assert spy.call_count == 0


def test_pop_next_value_calls_raw_decode_at_most_once_per_value():
    accumulator = JSONFragmentAccumulator()
    accumulator.append('{"a": 1}' * 3)

    with patch.object(json.JSONDecoder, "raw_decode", autospec=True, side_effect=json.JSONDecoder.raw_decode) as spy:
        for _ in range(3):
            found, _ = accumulator.pop_next_value()
            assert found is True
        assert spy.call_count == 3


def test_accumulation_of_many_fragments_is_not_quadratic():
    """Regression guard: appending 1000 shards must stay O(n) total, not the
    O(n^2) cost of repeated `buffer += fragment` string concatenation."""
    accumulator = JSONFragmentAccumulator()
    shard = "x" * 2048

    start = time.perf_counter()
    for _ in range(1000):
        accumulator.append(shard)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 50, f"1000-fragment append took {elapsed_ms:.1f} ms (expected < 50 ms); O(n^2) regression?"
