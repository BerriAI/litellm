import pytest

from litellm.proxy.data_manager.config import (
    DATA_MANAGER_JOB_ROLE,
    LITELLM_DATA_MANAGER_ENABLED_ENV,
    data_manager_enabled,
    proxy_skips_spend_log_cleanup,
)


@pytest.mark.parametrize(
    ("environment_value", "expected"),
    ((None, False), ("false", False), ("true", True), ("True", True)),
)
def test_data_manager_enabled_reads_environment_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
    environment_value: str | None,
    expected: bool,
) -> None:
    if environment_value is None:
        monkeypatch.delenv(LITELLM_DATA_MANAGER_ENABLED_ENV, raising=False)
    else:
        monkeypatch.setenv(LITELLM_DATA_MANAGER_ENABLED_ENV, environment_value)

    assert data_manager_enabled() is expected


@pytest.mark.parametrize(
    ("flag_value", "job_role", "expected"),
    (
        (None, None, False),
        ("true", None, True),
        ("true", "collector", True),
        ("true", DATA_MANAGER_JOB_ROLE, False),
        (None, DATA_MANAGER_JOB_ROLE, False),
        ("false", None, False),
    ),
)
def test_proxy_skips_spend_log_cleanup_depends_on_flag_and_role(
    monkeypatch: pytest.MonkeyPatch,
    flag_value: str | None,
    job_role: str | None,
    expected: bool,
) -> None:
    if flag_value is None:
        monkeypatch.delenv(LITELLM_DATA_MANAGER_ENABLED_ENV, raising=False)
    else:
        monkeypatch.setenv(LITELLM_DATA_MANAGER_ENABLED_ENV, flag_value)
    if job_role is None:
        monkeypatch.delenv("LITELLM_JOB_ROLE", raising=False)
    else:
        monkeypatch.setenv("LITELLM_JOB_ROLE", job_role)

    assert proxy_skips_spend_log_cleanup() is expected
