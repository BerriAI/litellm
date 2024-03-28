"""
Supports using JWT's for authenticating into the proxy. 

Currently only supports admin. 

JWT token must have 'litellm_proxy_admin' in scope. 
"""

import jwt
import json
import os
from litellm.caching import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_JWTAuth, LiteLLM_UserTable
from litellm.proxy.utils import PrismaClient
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler
from typing import Optional


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
        self,
        prisma_client: Optional[PrismaClient],
        user_api_key_cache: DualCache,
        litellm_jwtauth: LiteLLM_JWTAuth,
    ) -> None:
        self.prisma_client = prisma_client
        self.user_api_key_cache = user_api_key_cache
        self.litellm_jwtauth = litellm_jwtauth

    def is_jwt(self, token: str):
        parts = token.split(".")
        return len(parts) == 3

    def is_admin(self, scopes: list) -> bool:
        if self.litellm_jwtauth.admin_jwt_scope in scopes:
            return True
        return False

    def is_team(self, scopes: list) -> bool:
        if self.litellm_jwtauth.team_jwt_scope in scopes:
            return True
        return False

    def get_end_user_id(self, token: dict, default_value: Optional[str]) -> str:
        try:
            if self.litellm_jwtauth.end_user_id_jwt_field is not None:
                user_id = token[self.litellm_jwtauth.end_user_id_jwt_field]
            else:
                user_id = None
        except KeyError:
            user_id = default_value
        return user_id

    def get_team_id(self, token: dict, default_value: Optional[str]) -> Optional[str]:
        try:
            team_id = token[self.litellm_jwtauth.team_id_jwt_field]
        except KeyError:
            team_id = default_value
        return team_id

    def get_scopes(self, token: dict) -> list:
        try:
            if isinstance(token["scope"], str):
                # Assuming the scopes are stored in 'scope' claim and are space-separated
                scopes = token["scope"].split()
            elif isinstance(token["scope"], list):
                scopes = token["scope"]
            else:
                raise Exception(
                    f"Unmapped scope type - {type(token['scope'])}. Supported types - list, str."
                )
        except KeyError:
            scopes = []
        return scopes

    async def get_public_key(self, kid: Optional[str]) -> dict:
        keys_url = os.getenv("JWT_PUBLIC_KEY_URL")

        if keys_url is None:
            raise Exception("Missing JWT Public Key URL from environment.")

        cached_keys = await self.user_api_key_cache.async_get_cache(
            "litellm_jwt_auth_keys"
        )
        if cached_keys is None:
            response = await self.http_handler.get(keys_url)

            keys = response.json()["keys"]

            await self.user_api_key_cache.async_set_cache(
                key="litellm_jwt_auth_keys",
                value=keys,
                ttl=self.litellm_jwtauth.public_key_ttl,  # cache for 10 mins
            )
        else:
            keys = cached_keys

        public_key: Optional[dict] = None

        if len(keys) == 1:
            public_key = keys[0]
        elif len(keys) > 1:
            for key in keys:
                if kid is not None and key["kid"] == kid:
                    public_key = key

        if public_key is None:
            raise Exception(
                f"No matching public key found. kid={kid}, keys_url={keys_url}, cached_keys={cached_keys}"
            )

        return public_key

    async def auth_jwt(self, token: str) -> dict:
        from jwt.algorithms import RSAAlgorithm

        header = jwt.get_unverified_header(token)

        verbose_proxy_logger.debug("header: %s", header)

        kid = header.get("kid", None)

        public_key = await self.get_public_key(kid=kid)

        if public_key is not None and isinstance(public_key, dict):
            jwk = {}
            if "kty" in public_key:
                jwk["kty"] = public_key["kty"]
            if "kid" in public_key:
                jwk["kid"] = public_key["kid"]
            if "n" in public_key:
                jwk["n"] = public_key["n"]
            if "e" in public_key:
                jwk["e"] = public_key["e"]

            public_key_rsa = RSAAlgorithm.from_jwk(json.dumps(jwk))

            try:
                # decode the token using the public key
                payload = jwt.decode(
                    token,
                    public_key_rsa,  # type: ignore
                    algorithms=["RS256"],
                    options={"verify_aud": False},
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
