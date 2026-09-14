"""Tests for the model deprecation email settings on SlackAlertingArgs."""

import pytest
from pydantic import ValidationError

from litellm.constants import EMAIL_MODEL_DEPRECATION_LOCK_ID
from litellm.types.integrations.slack_alerting import (
    SlackAlertingArgs,
    SlackAlertingCacheKeys,
)


def test_should_default_thresholds_to_30_7_0():
    assert SlackAlertingArgs().model_deprecation_email_thresholds == [30, 7, 0]


def test_should_default_ttl_to_90_days():
    assert SlackAlertingArgs().model_deprecation_email_ttl == 90 * 24 * 60 * 60


def test_should_dedupe_and_sort_thresholds_descending():
    args = SlackAlertingArgs(model_deprecation_email_thresholds=[0, 7, 7, 30])
    assert args.model_deprecation_email_thresholds == [30, 7, 0]


def test_should_allow_empty_thresholds_to_disable_emails():
    assert SlackAlertingArgs(model_deprecation_email_thresholds=[]).model_deprecation_email_thresholds == []


def test_should_reject_negative_thresholds():
    with pytest.raises(ValidationError, match="non-negative"):
        SlackAlertingArgs(model_deprecation_email_thresholds=[30, -1])


def test_should_reject_non_positive_ttl():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        SlackAlertingArgs(model_deprecation_email_ttl=0)


def test_should_expose_pass_cache_key_and_lock_id():
    assert SlackAlertingCacheKeys.deprecation_email_pass_key.value == "model_deprecation_email_pass_at"
    assert EMAIL_MODEL_DEPRECATION_LOCK_ID == "email_model_deprecation_warning"
