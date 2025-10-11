import pytest
from litellm.proxy.hooks.proxy_track_cost_callback import _should_track_cost_callback

# Test: should track cost when any key is present
def test_should_track_cost_callback_valid():
    assert _should_track_cost_callback(user_api_key="key", user_id=None, team_id=None, end_user_id=None) is True
    assert _should_track_cost_callback(user_api_key=None, user_id="uid", team_id=None, end_user_id=None) is True
    assert _should_track_cost_callback(user_api_key=None, user_id=None, team_id="tid", end_user_id=None) is True
    assert _should_track_cost_callback(user_api_key=None, user_id=None, team_id=None, end_user_id="eid") is True
    assert _should_track_cost_callback(user_api_key="key", user_id="uid", team_id="tid", end_user_id="eid") is True

# Test: should raise Exception when all are None
def test_should_track_cost_callback_invalid():
    with pytest.raises(Exception, match="User API key and team id and user id missing from custom callback."):
        _should_track_cost_callback(user_api_key=None, user_id=None, team_id=None, end_user_id=None)
