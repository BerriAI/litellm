import time
from datetime import datetime, timedelta, timezone

from fastapi.responses import RedirectResponse


class UISessionHandler:

    @staticmethod
    def build_authenticated_ui_jwt_token(
        user_id: str,
        user_role: str,
        premium_user: bool,
        disabled_non_admin_personal_key_creation: bool,
    ) -> str:
        """
        Build a JWT token for the authenticated UI session

        This token is used to authenticate the user's session when they are redirected to the UI
        """
        import jwt

        from litellm.proxy.proxy_server import general_settings, master_key

        if master_key is None:
            raise ValueError("Master key is not set")

        expiration = datetime.now(timezone.utc) + timedelta(hours=24)
        jwt_token = jwt.encode(  # type: ignore
            {
                "user_id": user_id,
                "user_email": None,
                "user_role": user_role,  # this is the path without sso - we can assume only admins will use this
                "login_method": "username_password",
                "premium_user": premium_user,
                "auth_header_name": general_settings.get(
                    "litellm_key_header_name", "Authorization"
                ),
                "iss": "litellm-proxy",  # Issuer - identifies this as an internal token
                "aud": "litellm-ui",  # Audience - identifies this as a UI token
                "exp": expiration,
                "disabled_non_admin_personal_key_creation": disabled_non_admin_personal_key_creation,
            },
            master_key,
            algorithm="HS256",
        )
        return jwt_token

    @staticmethod
    def is_ui_session_token(token_dict: dict) -> bool:
        """
        Returns True if the token is a UI session token
        """
        return (
            token_dict.get("iss") == "litellm-proxy"
            and token_dict.get("aud") == "litellm-ui"
        )

    @staticmethod
    def generate_authenticated_redirect_response(
        redirect_url: str, jwt_token: str
    ) -> RedirectResponse:
        redirect_response = RedirectResponse(url=redirect_url, status_code=303)
        redirect_response.set_cookie(
            key=UISessionHandler._generate_token_name(),
            value=jwt_token,
            secure=True,
            httponly=True,
            samesite="strict",
        )
        return redirect_response

    @staticmethod
    def _generate_token_name() -> str:
        current_timestamp = int(time.time())
        cookie_name = f"litellm_ui_token_{current_timestamp}"
        return cookie_name
