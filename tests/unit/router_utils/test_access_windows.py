from datetime import datetime, time, timezone
from typing import Final


from litellm.router_utils.access_windows import (
    access_windows_config_error,
    filter_reserved_deployments,
    is_window_active,
)
from litellm.types.router import ModelAccessWindow

_NIGHT_NY: Final = ModelAccessWindow(
    start=time(22, 0),
    end=time(6, 0),
    timezone="America/New_York",
    team_ids=("team-nightly",),
)


def _window(start: str, end: str, tz: str = "UTC", team_ids=("team-a",)) -> ModelAccessWindow:
    return ModelAccessWindow(
        start=time.fromisoformat(start),
        end=time.fromisoformat(end),
        timezone=tz,
        team_ids=tuple(team_ids),
    )


def _deployment(windows: object = None) -> dict:
    if windows is None:
        return {"model_info": {}}
    return {"model_info": {"access_windows": windows}}


def test_same_day_window_active_and_inactive():
    window: Final = _window("09:00", "17:00")
    assert is_window_active(window, datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)) is True
    assert is_window_active(window, datetime(2026, 3, 9, 20, 0, tzinfo=timezone.utc)) is False


def test_cross_midnight_window():
    window: Final = _window("22:00", "06:00")
    assert is_window_active(window, datetime(2026, 3, 9, 23, 0, tzinfo=timezone.utc)) is True
    assert is_window_active(window, datetime(2026, 3, 10, 5, 59, tzinfo=timezone.utc)) is True
    assert is_window_active(window, datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)) is False


def test_start_boundary_inclusive_and_end_boundary_exclusive():
    window: Final = _window("22:00", "06:00")
    assert is_window_active(window, datetime(2026, 3, 9, 22, 0, tzinfo=timezone.utc)) is True
    assert is_window_active(window, datetime(2026, 3, 10, 6, 0, tzinfo=timezone.utc)) is False


def test_dst_spring_forward_gap_uses_real_local_time():
    window: Final = ModelAccessWindow(
        start=time(1, 30),
        end=time(3, 30),
        timezone="America/New_York",
        team_ids=("team-a",),
    )
    assert is_window_active(window, datetime(2026, 3, 8, 6, 30, tzinfo=timezone.utc)) is True
    assert is_window_active(window, datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc)) is True
    assert is_window_active(window, datetime(2026, 3, 8, 7, 30, tzinfo=timezone.utc)) is False


def test_naive_now_is_treated_as_utc():
    window: Final = _window("09:00", "17:00")
    assert is_window_active(window, datetime(2026, 3, 9, 12, 0)) is True


def test_team_in_second_window_is_kept():
    deployments: Final = (
        _deployment([
            {"start": "01:00", "end": "02:00", "timezone": "UTC", "team_ids": ["team-other"]},
            {"start": "20:00", "end": "23:59", "timezone": "UTC", "team_ids": ["team-a"]},
        ]),
    )
    result: Final = filter_reserved_deployments(
        deployments, "team-a", now=datetime(2026, 3, 9, 21, 0, tzinfo=timezone.utc)
    )
    assert result.deployments == deployments
    assert result.blocking_window is None


def test_unlisted_team_is_dropped_with_blocking_window():
    deployments: Final = (_deployment([_NIGHT_NY.model_dump()]),)
    result: Final = filter_reserved_deployments(
        deployments, "team-b", now=datetime(2026, 3, 10, 4, 0, tzinfo=timezone.utc)
    )
    assert result.deployments == ()
    assert result.blocking_window == _NIGHT_NY


def test_missing_team_id_is_dropped():
    result: Final = filter_reserved_deployments(
        (_deployment([_NIGHT_NY.model_dump()]),),
        None,
        now=datetime(2026, 3, 10, 4, 0, tzinfo=timezone.utc),
    )
    assert result.deployments == ()
    assert result.blocking_window == _NIGHT_NY


def test_listed_team_is_kept():
    deployments: Final = (_deployment([_NIGHT_NY.model_dump()]),)
    result: Final = filter_reserved_deployments(
        deployments, "team-nightly", now=datetime(2026, 3, 10, 4, 0, tzinfo=timezone.utc)
    )
    assert result.deployments == deployments
    assert result.blocking_window is None


def test_deployment_without_windows_kept_for_anyone():
    deployments: Final = (_deployment(),)
    result: Final = filter_reserved_deployments(
        deployments, None, now=datetime(2026, 3, 10, 4, 0, tzinfo=timezone.utc)
    )
    assert result.deployments == deployments
    assert result.blocking_window is None


def test_inactive_window_keeps_deployment_for_unlisted_team():
    deployments: Final = (_deployment([_NIGHT_NY.model_dump()]),)
    result: Final = filter_reserved_deployments(
        deployments, "team-b", now=datetime(2026, 3, 10, 16, 0, tzinfo=timezone.utc)
    )
    assert result.deployments == deployments
    assert result.blocking_window is None


def test_unreserved_deployment_survives_for_other_team():
    reserved: Final = _deployment([_NIGHT_NY.model_dump()])
    open_deployment: Final = _deployment()
    result: Final = filter_reserved_deployments(
        (reserved, open_deployment),
        "team-b",
        now=datetime(2026, 3, 10, 4, 0, tzinfo=timezone.utc),
    )
    assert result.deployments == (open_deployment,)
    assert result.blocking_window == _NIGHT_NY


def test_config_error_unknown_timezone():
    error: Final = access_windows_config_error(
        {"access_windows": [{"start": "22:00", "end": "06:00", "timezone": "Mars/Olympus", "team_ids": ["t"]}]},
        model_name="nightly-model",
    )
    assert error is not None
    assert "nightly-model" in error
    assert "access_windows" in error
    assert "Mars/Olympus" in error


def test_config_error_bad_time():
    error: Final = access_windows_config_error(
        {"access_windows": [{"start": "25:00", "end": "06:00", "timezone": "UTC", "team_ids": ["t"]}]},
        model_name="m",
    )
    assert error is not None
    assert "access_windows" in error


def test_config_error_empty_team_ids():
    error: Final = access_windows_config_error(
        {"access_windows": [{"start": "22:00", "end": "06:00", "timezone": "UTC", "team_ids": []}]},
        model_name="m",
    )
    assert error is not None


def test_config_error_start_equals_end():
    error: Final = access_windows_config_error(
        {"access_windows": [{"start": "22:00", "end": "22:00", "timezone": "UTC", "team_ids": ["t"]}]},
        model_name="m",
    )
    assert error is not None


def test_config_error_none_when_absent():
    assert access_windows_config_error({}, model_name="m") is None
    assert access_windows_config_error({"access_windows": None}, model_name="m") is None


def test_config_error_offset_aware_time():
    error: Final = access_windows_config_error(
        {"access_windows": [{"start": "22:00+05:00", "end": "06:00", "timezone": "UTC", "team_ids": ["t"]}]},
        model_name="m",
    )
    assert error is not None
    assert "UTC offset" in error
