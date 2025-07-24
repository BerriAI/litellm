from typing import Literal, Optional, TypedDict


class ReturnedUITokenObject(TypedDict):
    """
    Returned object for UI login
    """

    user_id: str
    key: str
    user_email: Optional[str]
    user_role: str
    login_method: Literal["sso", "username_password"]
    premium_user: bool
    auth_header_name: str
    disabled_non_admin_personal_key_creation: bool
    server_root_path: str  # e.g. `/litellm`
