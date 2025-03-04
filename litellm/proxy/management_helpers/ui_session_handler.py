import time

from fastapi.responses import RedirectResponse


class UISessionHandler:
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
        cookie_name = f"token_{current_timestamp}"
        return cookie_name
