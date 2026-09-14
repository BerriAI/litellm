"""Email team admins about the deprecating models their team can reach"""

from __future__ import annotations

from collections.abc import Sequence


def select_milestone(days_until: int, thresholds: Sequence[int]) -> int | None:
    """The most urgent threshold already reached, None while the first one is still ahead"""
    return min((threshold for threshold in thresholds if days_until <= threshold), default=None)
