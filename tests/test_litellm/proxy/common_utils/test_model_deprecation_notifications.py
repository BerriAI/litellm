"""Tests for the model deprecation email notifications module."""

from __future__ import annotations

import pytest

from litellm.proxy.common_utils.model_deprecation_notifications import select_milestone


class TestSelectMilestone:
    @pytest.mark.parametrize(
        ("days_until", "expected"),
        [(45, None), (30, 30), (25, 30), (7, 7), (3, 7), (0, 0), (-12, 0)],
    )
    def test_should_pick_the_most_urgent_threshold_reached(self, days_until, expected):
        assert select_milestone(days_until, (30, 7, 0)) == expected

    def test_should_return_none_when_no_thresholds_configured(self):
        assert select_milestone(-5, ()) is None

    def test_should_not_depend_on_threshold_order(self):
        assert select_milestone(5, (0, 7, 30)) == 7
