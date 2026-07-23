from typing import Literal, Optional

from typing_extensions import TypedDict


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


class UISessionJWTClaims(ReturnedUITokenObject):
    """The signed claim set of the UI session cookie.

    ``exp`` is stamped by the encoder rather than carried on
    :class:`ReturnedUITokenObject`, because the lifetime belongs to the credential and not to the
    payload that endpoints hand back in a response body.
    """

    exp: int


class ParsedOpenIDResult(TypedDict, total=False):
    """
    Parsed OpenID result
    """

    user_email: Optional[str]
    user_id: Optional[str]
    user_role: Optional[str]
