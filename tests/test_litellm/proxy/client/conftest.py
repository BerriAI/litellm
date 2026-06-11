import pytest


@pytest.fixture(autouse=True)
def isolate_from_real_cli_token(monkeypatch):
    """Never let tests read the developer's real ~/.litellm/token.json from `lite login`."""
    monkeypatch.setattr(
        "litellm.litellm_core_utils.cli_token_utils.load_cli_token", lambda: None
    )
