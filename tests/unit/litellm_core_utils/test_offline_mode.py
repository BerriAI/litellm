import pytest

from litellm.litellm_core_utils.offline_mode import is_offline_mode


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("True", True), (" TRUE ", True), ("false", False), ("", False), ("1", False), ("yes", False)],
)
def test_is_offline_mode_reads_true_case_insensitively(monkeypatch, value, expected):
    monkeypatch.setenv("LITELLM_OFFLINE_MODE", value)
    assert is_offline_mode() is expected


def test_is_offline_mode_is_off_when_unset(monkeypatch):
    monkeypatch.delenv("LITELLM_OFFLINE_MODE", raising=False)
    assert is_offline_mode() is False
