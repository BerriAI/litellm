import os
import sys

import pytest

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy._types import CallInfo, Litellm_EntityType


def test_get_user_info_str_without_token():
    """
    CallInfo.token is Optional and model_dump(exclude_none=True) drops it when
    unset, so _get_user_info_str must not assume the key is present.

    It is unset for JWT-authenticated requests: that auth path builds
    UserAPIKeyAuth without a token, so a team soft-budget alert for a JWT caller
    reaches here with token=None. Popping it unconditionally raised
    KeyError: 'token', which propagated out of ProxyLogging.budget_alerts() and
    aborted the alert - including the email alert dispatched after the Slack one.
    """
    slack_alerting = SlackAlerting()

    user_info = CallInfo(
        spend=25.0,
        soft_budget=10.0,
        team_id="my-team",
        team_alias="My Team",
        event_group=Litellm_EntityType.TEAM,
    )

    msg = slack_alerting._get_user_info_str(user_info)

    assert "token" not in msg
    assert "*spend:* `25.0`" in msg
    assert "*soft_budget:* `10.0`" in msg
    assert "*team_id:* `my-team`" in msg
    # Litellm_EntityType is rendered as its value, not the enum repr
    assert "*event_group:* `team`" in msg


def test_get_user_info_str_omits_token_when_present():
    """A token, when set, is still excluded from the alert message."""
    slack_alerting = SlackAlerting()

    user_info = CallInfo(
        spend=5.0,
        max_budget=10.0,
        token="sk-hashed-token-value",
        user_id="my-user",
        event_group=Litellm_EntityType.KEY,
    )

    msg = slack_alerting._get_user_info_str(user_info)

    assert "sk-hashed-token-value" not in msg
    assert "token" not in msg
    assert "*user_id:* `my-user`" in msg
