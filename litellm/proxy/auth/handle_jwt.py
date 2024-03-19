"""
Supports using JWT's for authenticating into the proxy. 

Currently only supports admin. 

JWT token must have 'litellm_proxy_admin' in scope. 
"""

import httpx
import jwt
import json
from jwt.algorithms import RSAAlgorithm
import os
from litellm.proxy._types import LiteLLMProxyRoles
from typing import Optional


class HTTPHandler:
    def __init__(self):
        self.client = httpx.AsyncClient()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
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

    def __init__(self) -> None:
        self.http_handler = HTTPHandler()

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

    def get_scopes(self, token: dict) -> list:
        try:
            # Assuming the scopes are stored in 'scope' claim and are space-separated
            scopes = token["scope"].split()
        except KeyError:
            scopes = []
        return scopes

    async def auth_jwt(self, token: str) -> dict:
        keys_url = os.getenv("OPENID_PUBLIC_KEY_URL")

        async with self.http_handler as http:
            response = await http.get(keys_url)

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
                        issuer=os.getenv("JWT_ISSUER"),
                    )
                    return payload

                except jwt.ExpiredSignatureError:
                    # the token is expired, do something to refresh it
                    raise Exception("Token Expired")
                except Exception as e:
                    raise Exception(f"Validation fails: {str(e)}")

        raise Exception("Invalid JWT Submitted")
