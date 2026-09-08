from typing import Literal

from typing_extensions import TypedDict


class ReturnedUITokenObject(TypedDict):
    """
    Returned object for UI login
    """

    user_id: str
    key: str
    user_email: str | None
    user_role: str
    login_method: Literal["sso", "username_password"]
    premium_user: bool
    auth_header_name: str
    disabled_non_admin_personal_key_creation: bool
    server_root_path: str  # e.g. `/litellm`


class ParsedOpenIDResult(TypedDict, total=False):
    """
    Parsed OpenID result
    """

    user_email: str | None
    user_id: str | None
    user_role: str | None
