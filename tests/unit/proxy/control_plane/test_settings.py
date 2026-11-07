from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.control_plane.settings import ControlPlaneSettings
from litellm.types.proxy.control_plane.auth import PASSWORD_MAX_LENGTH, USERNAME_MAX_LENGTH

MASTER_KEY: Final = "01234567890123456789012345678901"


@pytest.mark.parametrize(
    ("username", "password"),
    [("u" * (USERNAME_MAX_LENGTH + 1), "p"), ("u", "p" * (PASSWORD_MAX_LENGTH + 1))],
)
def test_configured_admin_credentials_share_the_login_request_bounds(username: str, password: str) -> None:
    with pytest.raises(ValidationError):
        ControlPlaneSettings.model_validate(
            {"LITELLM_MASTER_KEY": MASTER_KEY, "UI_USERNAME": username, "UI_PASSWORD": password}
        )


def test_password_login_without_default_roles_requires_a_custom_admin_role() -> None:
    with pytest.raises(ValidationError, match="custom admin role"):
        ControlPlaneSettings.model_validate(
            {
                "LITELLM_MASTER_KEY": MASTER_KEY,
                "UI_USERNAME": "admin",
                "UI_PASSWORD": "pw",
                "include_default_roles": False,
            }
        )
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "UI_USERNAME": "admin",
            "UI_PASSWORD": "pw",
            "include_default_roles": False,
            "roles": {"admin": {"scopes": ["dataplanes:read"]}},
        }
    )
    assert not settings.include_default_roles and "admin" in settings.roles


def test_credentials_at_the_bound_are_accepted() -> None:
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "UI_USERNAME": "u" * USERNAME_MAX_LENGTH,
            "UI_PASSWORD": "p" * PASSWORD_MAX_LENGTH,
        }
    )
    assert settings.ui_username is not None and len(settings.ui_username) == USERNAME_MAX_LENGTH
