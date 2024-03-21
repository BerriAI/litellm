"""
Supports using JWT's for authenticating into the proxy. 

Currently only supports admin. 

JWT token must have 'litellm_proxy_admin' in scope. 
"""

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm
import json
import os
from litellm.caching import DualCache
from litellm.proxy._types import LiteLLMProxyRoles, LiteLLM_UserTable
from litellm.proxy.utils import PrismaClient
from typing import Optional


class HTTPHandler:
    def __init__(self, concurrent_limit=1000):
        # Create a client with a connection pool
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrent_limit,
                max_keepalive_connections=concurrent_limit,
            )
        )

    async def close(self):
        # Close the client when you're done with it
        await self.client.aclose()

    async def get(
        self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None
    ):
        response = await self.client.get(url, params=params, headers=headers)
        return response

    async def post(
        self,
        url: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
    ):
        response = await self.client.post(
            url, data=data, params=params, headers=headers
        )
        return response


class JWTHandler:
    """
    - treat the sub id passed in as the user id
    - return an error if id making request doesn't exist in proxy user table
    - track spend against the user id
    - if role="litellm_proxy_user" -> allow making calls + info. Can not edit budgets
    """

    prisma_client: Optional[PrismaClient]
    user_api_key_cache: DualCache

    def __init__(
        self,
    ) -> None:
        self.http_handler = HTTPHandler()

    def update_environment(
        self, prisma_client: Optional[PrismaClient], user_api_key_cache: DualCache
    ) -> None:
        self.prisma_client = prisma_client
        self.user_api_key_cache = user_api_key_cache

    def is_jwt(self, token: str):
        parts = token.split(".")
        return len(parts) == 3

    def is_admin(self, scopes: list) -> bool:
        if LiteLLMProxyRoles.PROXY_ADMIN.value in scopes:
            return True
        return False

    def get_user_id(self, token: dict, default_value: str) -> str:
        try:
            user_id = token["sub"]
        except KeyError:
            user_id = default_value
        return user_id

    def get_team_id(self, token: dict, default_value: Optional[str]) -> Optional[str]:
        try:
            team_id = token["azp"]
        except KeyError:
            team_id = default_value
        return team_id

    async def get_user_object(self, user_id: str) -> LiteLLM_UserTable:
        """
        - Check if user id in proxy User Table
        - if valid, return LiteLLM_UserTable object with defined limits
        - if not, then raise an error
        """
        if self.prisma_client is None:
            raise Exception(
                "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
            )

        # check if in cache
        cached_user_obj = self.user_api_key_cache.async_get_cache(key=user_id)
        if cached_user_obj is not None:
            if isinstance(cached_user_obj, dict):
                return LiteLLM_UserTable(**cached_user_obj)
            elif isinstance(cached_user_obj, LiteLLM_UserTable):
                return cached_user_obj
        # else, check db
        try:
            response = await self.prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": user_id}
            )

            if response is None:
                raise Exception

            return LiteLLM_UserTable(**response.dict())
        except Exception as e:
            raise Exception(
                f"User doesn't exist in db. User={user_id}. Create user via `/user/new` call."
            )

    def get_scopes(self, token: dict) -> list:
        try:
            # Assuming the scopes are stored in 'scope' claim and are space-separated
            scopes = token["scope"].split()
        except KeyError:
            scopes = []
        return scopes

    async def auth_jwt(self, token: str) -> dict:
        keys_url = os.getenv("JWT_PUBLIC_KEY_URL")

        if keys_url is None:
            raise Exception("Missing JWT Public Key URL from environment.")

        response = await self.http_handler.get(keys_url)

        keys = response.json()["keys"]

        header = jwt.get_unverified_header(token)
        kid = header["kid"]

        for key in keys:
            if key["kid"] == kid:
                jwk = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "n": key["n"],
                    "e": key["e"],
                }
                public_key = RSAAlgorithm.from_jwk(json.dumps(jwk))

                try:
                    # decode the token using the public key
                    payload = jwt.decode(
                        token,
                        public_key,  # type: ignore
                        algorithms=["RS256"],
                        audience="account",
                    )
                    return payload

                except jwt.ExpiredSignatureError:
                    # the token is expired, do something to refresh it
                    raise Exception("Token Expired")
                except Exception as e:
                    raise Exception(f"Validation fails: {str(e)}")

        raise Exception("Invalid JWT Submitted")

    async def close(self):
        await self.http_handler.close()
