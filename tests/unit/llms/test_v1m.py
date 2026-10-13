from typing import Final

from litellm.llms.v1m import V1MConfig, V1MError


def test_v1m_config_initialization() -> None:
    config: Final = V1MConfig()
    assert config is not None


def test_v1m_error_initialization() -> None:
     err: Final = V1MError(status_code=500, message="upstream error")
    assert err.status_code == 500
    assert str(err) == "upstream error"
