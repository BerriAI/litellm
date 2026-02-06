from typing import Any, cast

import pytest

from litellm.proxy.common_utils.realtime_utils import _realtime_request_body
from litellm.proxy.proxy_server import _realtime_query_params_template


@pytest.fixture(autouse=True)
def clear_realtime_caches():
    _realtime_request_body.cache_clear()
    _realtime_query_params_template.cache_clear()
    yield
    _realtime_request_body.cache_clear()
    _realtime_query_params_template.cache_clear()


def test_realtime_request_body_returns_immutable_bytes():
    cached_body = _realtime_request_body("gpt-4o")

    with pytest.raises(TypeError):
        cast(Any, cached_body)[0] = ord("x")
        
        
def test_realtime_query_params_template_returns_immutable_tuples():
    cached_tuple = _realtime_query_params_template("gpt-4o", "intent-a")

    with pytest.raises(TypeError):
        cast(Any, cached_tuple)[0] = ("model", "mutated")


def test_realtime_request_body_caches_each_model_separately():
    gpt4o_body_first = _realtime_request_body("gpt-4o")
    gpt4o_body_second = _realtime_request_body("gpt-4o")
    gpt4o_mini_body = _realtime_request_body("gpt-4o-mini")

    assert gpt4o_body_first is gpt4o_body_second
    assert gpt4o_body_first == b'{"model": "gpt-4o"}'
    assert gpt4o_mini_body == b'{"model": "gpt-4o-mini"}'
    assert gpt4o_body_first is not gpt4o_mini_body


def test_realtime_query_params_template_caches_each_pair_separately():
    params_with_intent_first = _realtime_query_params_template("gpt-4o", "intent-a")
    params_with_intent_second = _realtime_query_params_template("gpt-4o", "intent-a")
    params_without_intent = _realtime_query_params_template("gpt-4o", None)

    assert params_with_intent_first is params_with_intent_second
    assert params_with_intent_first == (("model", "gpt-4o"), ("intent", "intent-a"))
    assert params_without_intent == (("model", "gpt-4o"),)
    assert params_with_intent_first is not params_without_intent


def test_realtime_query_params_dict_copies_do_not_leak_state():
    params_dict_one = dict(_realtime_query_params_template("gpt-4o", "intent-a"))
    params_dict_one["new"] = "value"

    params_dict_two = dict(_realtime_query_params_template("gpt-4o", "intent-a"))

    assert "new" not in params_dict_two
    assert params_dict_two == {"model": "gpt-4o", "intent": "intent-a"}

