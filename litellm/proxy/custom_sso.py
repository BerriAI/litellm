"""
Example Custom SSO Handler

Use this if you want to run custom code after litellm has retrieved information from your IDP (Identity Provider).

Flow:
- User lands on Admin UI
- LiteLLM redirects user to your SSO provider
- Your SSO provider redirects user back to LiteLLM
- LiteLLM has retrieved user information from your IDP
- Your custom SSO handler is called and returns an object of type SSOUserDefinedValues
- User signed in to UI
"""

from fastapi import Request
from fastapi_sso.sso.base import OpenID

from litellm.proxy._types import LitellmUserRoles, SSOUserDefinedValues
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    new_user,
    user_info,
)
from litellm.proxy.management_endpoints.team_endpoints import add_new_member


async def custom_sso_handler(userIDPInfo: OpenID) -> SSOUserDefinedValues:
    try:
        print("inside custom sso handler")  # noqa
        print(f"userIDPInfo: {userIDPInfo}")  # noqa

        if userIDPInfo.id is None:
            raise ValueError(
                f"No ID found for user. userIDPInfo.id is None {userIDPInfo}"
            )

        # check if user exists in litellm proxy DB
        _user_info = await user_info(user_id=userIDPInfo.id)
        print("_user_info from litellm DB ", _user_info)  # noqa

        return SSOUserDefinedValues(
            models=[],
            user_id=userIDPInfo.id,
            user_email=userIDPInfo.email,
            user_role=LitellmUserRoles.INTERNAL_USER.value,
            max_budget=10,
            budget_duration="1d",
        )
    except Exception as e:
        raise Exception("Failed custom auth")
