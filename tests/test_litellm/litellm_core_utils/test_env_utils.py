import pytest

from litellm.litellm_core_utils.env_utils import get_positive_env_int


class TestGetPositiveEnvInt:
    def test_returns_parsed_positive_value(self, monkeypatch):
        monkeypatch.setenv("MY_LIMIT", "42")
        assert get_positive_env_int("MY_LIMIT", 1000) == 42

    def test_falls_back_to_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MY_LIMIT", raising=False)
        assert get_positive_env_int("MY_LIMIT", 1000) == 1000

    @pytest.mark.parametrize("bad_value", ["0", "-1", "-999", "", "   ", "abc"])
    def test_falls_back_to_default_on_non_positive_or_invalid(self, monkeypatch, bad_value):
        """A misconfigured value must not degrade a bound into e.g. LIMIT 0."""
        monkeypatch.setenv("MY_LIMIT", bad_value)
        assert get_positive_env_int("MY_LIMIT", 1000) == 1000
