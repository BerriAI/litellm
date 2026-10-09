import pytest

from litellm.proxy.data_manager.config import LITELLM_DATA_MANAGER_ENABLED_ENV, data_manager_enabled


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
