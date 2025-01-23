"""
Supports using JWT's for authenticating into the proxy. 

Currently only supports admin. 

JWT token must have 'litellm_proxy_admin' in scope. 
"""

import json
import os
from typing import List, Optional, cast

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler
from litellm.proxy._types import JWKKeyValue, JWTKeyItem, LiteLLM_JWTAuth
from litellm.proxy.utils import PrismaClient


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
        self.leeway = 0

    def update_environment(
        self,
        prisma_client: Optional[PrismaClient],
        user_api_key_cache: DualCache,
        litellm_jwtauth: LiteLLM_JWTAuth,
        leeway: int = 0,
    ) -> None:
        self.prisma_client = prisma_client
        self.user_api_key_cache = user_api_key_cache
        self.litellm_jwtauth = litellm_jwtauth
        self.leeway = leeway

    def is_jwt(self, token: str):
        parts = token.split(".")
        return len(parts) == 3

    def is_admin(self, scopes: list) -> bool:
        if self.litellm_jwtauth.admin_jwt_scope in scopes:
            return True
        return False

    def get_team_ids_from_jwt(self, token: dict) -> List[str]:
        if self.litellm_jwtauth.team_ids_jwt_field is not None:
            return token[self.litellm_jwtauth.team_ids_jwt_field]
        return []

    def get_end_user_id(
        self, token: dict, default_value: Optional[str]
    ) -> Optional[str]:
        try:
            if self.litellm_jwtauth.end_user_id_jwt_field is not None:
                user_id = token[self.litellm_jwtauth.end_user_id_jwt_field]
            else:
                user_id = None
        except KeyError:
            user_id = default_value
        return user_id

    def is_required_team_id(self) -> bool:
        """
        Returns:
        - True: if 'team_id_jwt_field' is set
        - False: if not
        """
        if self.litellm_jwtauth.team_id_jwt_field is None:
            return False
        return True

    def is_enforced_email_domain(self) -> bool:
        """
        Returns:
        - True: if 'user_allowed_email_domain' is set
        - False: if 'user_allowed_email_domain' is None
        """

        if self.litellm_jwtauth.user_allowed_email_domain is not None and isinstance(
            self.litellm_jwtauth.user_allowed_email_domain, str
        ):
            return True
        return False

    def get_team_id(self, token: dict, default_value: Optional[str]) -> Optional[str]:
        try:
            if self.litellm_jwtauth.team_id_jwt_field is not None:
                team_id = token[self.litellm_jwtauth.team_id_jwt_field]
            elif self.litellm_jwtauth.team_id_default is not None:
                team_id = self.litellm_jwtauth.team_id_default
            else:
                team_id = None
        except KeyError:
            team_id = default_value
        return team_id

    def is_upsert_user_id(self, valid_user_email: Optional[bool] = None) -> bool:
        """
        Returns:
        - True: if 'user_id_upsert' is set AND valid_user_email is not False
        - False: if not
        """
        if valid_user_email is False:
            return False
        return self.litellm_jwtauth.user_id_upsert

    def get_user_id(self, token: dict, default_value: Optional[str]) -> Optional[str]:
        try:
            if self.litellm_jwtauth.user_id_jwt_field is not None:
                user_id = token[self.litellm_jwtauth.user_id_jwt_field]
            else:
                user_id = default_value
        except KeyError:
            user_id = default_value
        return user_id

    def get_user_email(
        self, token: dict, default_value: Optional[str]
    ) -> Optional[str]:
        try:
            if self.litellm_jwtauth.user_email_jwt_field is not None:
                user_email = token[self.litellm_jwtauth.user_email_jwt_field]
            else:
                user_email = None
        except KeyError:
            user_email = default_value
        return user_email

    def get_org_id(self, token: dict, default_value: Optional[str]) -> Optional[str]:
        try:
            if self.litellm_jwtauth.org_id_jwt_field is not None:
                org_id = token[self.litellm_jwtauth.org_id_jwt_field]
            else:
                org_id = None
        except KeyError:
            org_id = default_value
        return org_id

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

            response_json = response.json()
            if "keys" in response_json:
                keys: JWKKeyValue = response.json()["keys"]
            else:
                keys = response_json

            await self.user_api_key_cache.async_set_cache(
                key="litellm_jwt_auth_keys",
                value=keys,
                ttl=self.litellm_jwtauth.public_key_ttl,  # cache for 10 mins
            )
        else:
            keys = cached_keys

        public_key = self.parse_keys(keys=keys, kid=kid)
        if public_key is None:
            raise Exception(
                f"No matching public key found. kid={kid}, keys_url={keys_url}, cached_keys={cached_keys}, len(keys)={len(keys)}"
            )
        return cast(dict, public_key)

    def parse_keys(self, keys: JWKKeyValue, kid: Optional[str]) -> Optional[JWTKeyItem]:
        public_key: Optional[JWTKeyItem] = None
        if len(keys) == 1:
            if isinstance(keys, dict) and (keys.get("kid", None) == kid or kid is None):
                public_key = keys
            elif isinstance(keys, list) and (
                keys[0].get("kid", None) == kid or kid is None
            ):
                public_key = keys[0]
        elif len(keys) > 1:
            for key in keys:
                if isinstance(key, dict):
                    key_kid = key.get("kid", None)
                else:
                    key_kid = None
                if (
                    kid is not None
                    and isinstance(key, dict)
                    and key_kid is not None
                    and key_kid == kid
                ):
                    public_key = key

        return public_key

    def is_allowed_domain(self, user_email: str) -> bool:
        if self.litellm_jwtauth.user_allowed_email_domain is None:
            return True

        email_domain = user_email.split("@")[-1]  # Extract domain from email
        if email_domain == self.litellm_jwtauth.user_allowed_email_domain:
            return True
        else:
            return False

    async def auth_jwt(self, token: str) -> dict:
        # Supported algos: https://pyjwt.readthedocs.io/en/stable/algorithms.html
        # "Warning: Make sure not to mix symmetric and asymmetric algorithms that interpret
        #   the key in different ways (e.g. HS* and RS*)."
        algorithms = ["RS256", "RS384", "RS512", "PS256", "PS384", "PS512"]

        audience = os.getenv("JWT_AUDIENCE")
        decode_options = None
        if audience is None:
            decode_options = {"verify_aud": False}

        import jwt
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
                    algorithms=algorithms,
                    options=decode_options,
                    audience=audience,
                    leeway=self.leeway,  # allow testing of expired tokens
                )
                return payload

            except jwt.ExpiredSignatureError:
                # the token is expired, do something to refresh it
                raise Exception("Token Expired")
            except Exception as e:
                raise Exception(f"Validation fails: {str(e)}")
        elif public_key is not None and isinstance(public_key, str):
            try:
                cert = x509.load_pem_x509_certificate(
                    public_key.encode(), default_backend()
                )

                # Extract public key
                key = cert.public_key().public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )

                # decode the token using the public key
                payload = jwt.decode(
                    token,
                    key,
                    algorithms=algorithms,
                    audience=audience,
                    options=decode_options,
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
