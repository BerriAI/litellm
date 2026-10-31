import pytest
from pydantic import ValidationError

from litellm.proxy.control_plane.settings import ControlPlaneSettings
from litellm.types.proxy.control_plane.auth import PASSWORD_MAX_LENGTH, USERNAME_MAX_LENGTH


@pytest.mark.parametrize(
    ("username", "password"),
    [("u" * (USERNAME_MAX_LENGTH + 1), "p"), ("u", "p" * (PASSWORD_MAX_LENGTH + 1))],
)
def test_configured_admin_credentials_share_the_login_request_bounds(username: str, password: str) -> None:
    with pytest.raises(ValidationError):
        ControlPlaneSettings.model_validate(
            {"LITELLM_MASTER_KEY": "sk-cp", "UI_USERNAME": username, "UI_PASSWORD": password}
        )


def test_credentials_at_the_bound_are_accepted() -> None:
    settings = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": "sk-cp",
            "UI_USERNAME": "u" * USERNAME_MAX_LENGTH,
            "UI_PASSWORD": "p" * PASSWORD_MAX_LENGTH,
        }
    )
    assert settings.ui_username is not None and len(settings.ui_username) == USERNAME_MAX_LENGTH
