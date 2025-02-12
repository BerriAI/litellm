"""
Supports using JWT's for authenticating into the proxy. 

Currently only supports admin. 

JWT token must have 'litellm_proxy_admin' in scope. 
"""

import json
import os
from typing import Any, List, Literal, Optional, Set, Tuple, cast

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.dot_notation_indexing import get_nested_value
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler
from litellm.proxy._types import (
    RBAC_ROLES,
    JWKKeyValue,
    JWTAuthBuilderResult,
    JWTKeyItem,
    LiteLLM_EndUserTable,
    LiteLLM_JWTAuth,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ScopeMapping,
    Span,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging

from .auth_checks import (
    _allowed_routes_check,
    allowed_routes_check,
    get_actual_routes,
    get_end_user_object,
    get_org_object,
    get_role_based_models,
    get_role_based_routes,
    get_team_object,
    get_user_object,
)


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

    def _rbac_role_from_role_mapping(self, token: dict) -> Optional[RBAC_ROLES]:
        """
        Returns the RBAC role the token 'belongs' to based on role mappings.

        Args:
            token (dict): The JWT token containing role information

        Returns:
            Optional[RBAC_ROLES]: The mapped internal RBAC role if a mapping exists,
                                None otherwise

        Note:
            The function handles both single string roles and lists of roles from the JWT.
            If multiple mappings match the JWT roles, the first matching mapping is returned.
        """
        if self.litellm_jwtauth.role_mappings is None:
            return None

        jwt_role = self.get_jwt_role(token=token, default_value=None)
        if not jwt_role:
            return None

        jwt_role_set = set(jwt_role)

        for role_mapping in self.litellm_jwtauth.role_mappings:
            # Check if the mapping role matches any of the JWT roles
            if role_mapping.role in jwt_role_set:
                return role_mapping.internal_role

        return None

    def get_rbac_role(self, token: dict) -> Optional[RBAC_ROLES]:
        """
        Returns the RBAC role the token 'belongs' to.

        RBAC roles allowed to make requests:
        - PROXY_ADMIN: can make requests to all routes
        - TEAM: can make requests to routes associated with a team
        - INTERNAL_USER: can make requests to routes associated with a user

        Resolves: https://github.com/BerriAI/litellm/issues/6793

        Returns:
        - PROXY_ADMIN: if token is admin
        - TEAM: if token is associated with a team
        - INTERNAL_USER: if token is associated with a user
        - None: if token is not associated with a team or user
        """
        scopes = self.get_scopes(token=token)
        is_admin = self.is_admin(scopes=scopes)
        user_roles = self.get_user_roles(token=token, default_value=None)

        if is_admin:
            return LitellmUserRoles.PROXY_ADMIN
        elif self.get_team_id(token=token, default_value=None) is not None:
            return LitellmUserRoles.TEAM
        elif self.get_user_id(token=token, default_value=None) is not None:
            return LitellmUserRoles.INTERNAL_USER
        elif user_roles is not None and self.is_allowed_user_role(
            user_roles=user_roles
        ):
            return LitellmUserRoles.INTERNAL_USER
        elif rbac_role := self._rbac_role_from_role_mapping(token=token):
            return rbac_role

        return None

    def is_admin(self, scopes: list) -> bool:
        if self.litellm_jwtauth.admin_jwt_scope in scopes:
            return True
        return False

    def get_team_ids_from_jwt(self, token: dict) -> List[str]:
        if (
            self.litellm_jwtauth.team_ids_jwt_field is not None
            and token.get(self.litellm_jwtauth.team_ids_jwt_field) is not None
        ):
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

    def get_user_roles(
        self, token: dict, default_value: Optional[List[str]]
    ) -> Optional[List[str]]:
        """
        Returns the user role from the token.

        Set via 'user_roles_jwt_field' in the config.
        """
        try:
            if self.litellm_jwtauth.user_roles_jwt_field is not None:
                user_roles = get_nested_value(
                    data=token,
                    key_path=self.litellm_jwtauth.user_roles_jwt_field,
                    default=default_value,
                )
            else:
                user_roles = default_value
        except KeyError:
            user_roles = default_value
        return user_roles

    def get_jwt_role(
        self, token: dict, default_value: Optional[List[str]]
    ) -> Optional[List[str]]:
        """
        Generic implementation of `get_user_roles` that can be used for both user and team roles.

        Returns the jwt role from the token.

        Set via 'roles_jwt_field' in the config.
        """
        try:
            if self.litellm_jwtauth.roles_jwt_field is not None:
                user_roles = get_nested_value(
                    data=token,
                    key_path=self.litellm_jwtauth.roles_jwt_field,
                    default=default_value,
                )
            else:
                user_roles = default_value
        except KeyError:
            user_roles = default_value
        return user_roles

    def is_allowed_user_role(self, user_roles: Optional[List[str]]) -> bool:
        """
        Returns the user role from the token.

        Set via 'user_allowed_roles' in the config.
        """
        if (
            user_roles is not None
            and self.litellm_jwtauth.user_allowed_roles is not None
            and any(
                role in self.litellm_jwtauth.user_allowed_roles for role in user_roles
            )
        ):
            return True
        return False

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

    def get_object_id(self, token: dict, default_value: Optional[str]) -> Optional[str]:
        try:
            if self.litellm_jwtauth.object_id_jwt_field is not None:
                object_id = token[self.litellm_jwtauth.object_id_jwt_field]
            else:
                object_id = default_value
        except KeyError:
            object_id = default_value
        return object_id

    def get_org_id(self, token: dict, default_value: Optional[str]) -> Optional[str]:
        try:
            if self.litellm_jwtauth.org_id_jwt_field is not None:
                org_id = token[self.litellm_jwtauth.org_id_jwt_field]
            else:
                org_id = None
        except KeyError:
            org_id = default_value
        return org_id

    def get_scopes(self, token: dict) -> List[str]:
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


class JWTAuthManager:
    """Manages JWT authentication and authorization operations"""

    @staticmethod
    def can_rbac_role_call_route(
        rbac_role: RBAC_ROLES,
        general_settings: dict,
        route: str,
    ) -> Literal[True]:
        """
        Checks if user is allowed to access the route, based on their role.
        """
        role_based_routes = get_role_based_routes(
            rbac_role=rbac_role, general_settings=general_settings
        )

        if role_based_routes is None or route is None:
            return True

        is_allowed = _allowed_routes_check(
            user_route=route,
            allowed_routes=role_based_routes,
        )

        if not is_allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Role={rbac_role} not allowed to call route={route}. Allowed routes={role_based_routes}",
            )

        return True

    @staticmethod
    def can_rbac_role_call_model(
        rbac_role: RBAC_ROLES,
        general_settings: dict,
        model: Optional[str],
    ) -> Literal[True]:
        """
        Checks if user is allowed to access the model, based on their role.
        """
        role_based_models = get_role_based_models(
            rbac_role=rbac_role, general_settings=general_settings
        )
        if role_based_models is None or model is None:
            return True

        if model not in role_based_models:
            raise HTTPException(
                status_code=403,
                detail=f"Role={rbac_role} not allowed to call model={model}. Allowed models={role_based_models}",
            )

        return True

    @staticmethod
    def check_scope_based_access(
        scope_mappings: List[ScopeMapping],
        scopes: List[str],
        request_data: dict,
        general_settings: dict,
    ) -> None:
        """
        Check if scope allows access to the requested model
        """
        if not scope_mappings:
            return None

        allowed_models = []
        for sm in scope_mappings:
            if sm.scope in scopes and sm.models:
                allowed_models.extend(sm.models)

        requested_model = request_data.get("model")

        if not requested_model:
            return None

        if requested_model not in allowed_models:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "model={} not allowed. Allowed_models={}".format(
                        requested_model, allowed_models
                    )
                },
            )
        return None

    @staticmethod
    async def check_rbac_role(
        jwt_handler: JWTHandler,
        jwt_valid_token: dict,
        general_settings: dict,
        request_data: dict,
        route: str,
        rbac_role: Optional[RBAC_ROLES],
    ) -> None:
        """Validate RBAC role and model access permissions"""
        if jwt_handler.litellm_jwtauth.enforce_rbac is True:
            if rbac_role is None:
                raise HTTPException(
                    status_code=403,
                    detail="Unmatched token passed in. enforce_rbac is set to True. Token must belong to a proxy admin, team, or user.",
                )
            JWTAuthManager.can_rbac_role_call_model(
                rbac_role=rbac_role,
                general_settings=general_settings,
                model=request_data.get("model"),
            )
            JWTAuthManager.can_rbac_role_call_route(
                rbac_role=rbac_role,
                general_settings=general_settings,
                route=route,
            )

    @staticmethod
    async def check_admin_access(
        jwt_handler: JWTHandler,
        scopes: list,
        route: str,
        user_id: Optional[str],
        org_id: Optional[str],
        api_key: str,
    ) -> Optional[JWTAuthBuilderResult]:
        """Check admin status and route access permissions"""
        if not jwt_handler.is_admin(scopes=scopes):
            return None

        is_allowed = allowed_routes_check(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            user_route=route,
            litellm_proxy_roles=jwt_handler.litellm_jwtauth,
        )
        if not is_allowed:
            allowed_routes: List[Any] = jwt_handler.litellm_jwtauth.admin_allowed_routes
            actual_routes = get_actual_routes(allowed_routes=allowed_routes)
            raise Exception(
                f"Admin not allowed to access this route. Route={route}, Allowed Routes={actual_routes}"
            )

        return JWTAuthBuilderResult(
            is_proxy_admin=True,
            team_object=None,
            user_object=None,
            end_user_object=None,
            org_object=None,
            token=api_key,
            team_id=None,
            user_id=user_id,
            end_user_id=None,
            org_id=org_id,
        )

    @staticmethod
    async def find_and_validate_specific_team_id(
        jwt_handler: JWTHandler,
        jwt_valid_token: dict,
        prisma_client: Optional[PrismaClient],
        user_api_key_cache: DualCache,
        parent_otel_span: Optional[Span],
        proxy_logging_obj: ProxyLogging,
    ) -> Tuple[Optional[str], Optional[LiteLLM_TeamTable]]:
        """Find and validate specific team ID"""
        individual_team_id = jwt_handler.get_team_id(
            token=jwt_valid_token, default_value=None
        )

        if not individual_team_id and jwt_handler.is_required_team_id() is True:
            raise Exception(
                f"No team id found in token. Checked team_id field '{jwt_handler.litellm_jwtauth.team_id_jwt_field}'"
            )

        ## VALIDATE TEAM OBJECT ###
        team_object: Optional[LiteLLM_TeamTable] = None
        if individual_team_id:
            team_object = await get_team_object(
                team_id=individual_team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
                team_id_upsert=jwt_handler.litellm_jwtauth.team_id_upsert,
            )

        return individual_team_id, team_object

    @staticmethod
    def get_all_team_ids(jwt_handler: JWTHandler, jwt_valid_token: dict) -> Set[str]:
        """Get combined team IDs from groups and individual team_id"""
        team_ids_from_groups = jwt_handler.get_team_ids_from_jwt(token=jwt_valid_token)

        all_team_ids = set(team_ids_from_groups)

        return all_team_ids

    @staticmethod
    async def find_team_with_model_access(
        team_ids: Set[str],
        requested_model: Optional[str],
        route: str,
        jwt_handler: JWTHandler,
        prisma_client: Optional[PrismaClient],
        user_api_key_cache: DualCache,
        parent_otel_span: Optional[Span],
        proxy_logging_obj: ProxyLogging,
    ) -> Tuple[Optional[str], Optional[LiteLLM_TeamTable]]:
        """Find first team with access to the requested model"""

        if not team_ids:
            if jwt_handler.litellm_jwtauth.enforce_team_based_model_access:
                raise HTTPException(
                    status_code=403,
                    detail="No teams found in token. `enforce_team_based_model_access` is set to True. Token must belong to a team.",
                )
            return None, None

        for team_id in team_ids:
            try:
                team_object = await get_team_object(
                    team_id=team_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )

                if team_object and team_object.models is not None:
                    team_models = team_object.models
                    if isinstance(team_models, list) and (
                        not requested_model
                        or requested_model in team_models
                        or "*" in team_models
                    ):
                        is_allowed = allowed_routes_check(
                            user_role=LitellmUserRoles.TEAM,
                            user_route=route,
                            litellm_proxy_roles=jwt_handler.litellm_jwtauth,
                        )
                        if is_allowed:
                            return team_id, team_object
            except Exception:
                continue

        if requested_model:
            raise HTTPException(
                status_code=403,
                detail=f"No team has access to the requested model: {requested_model}. Checked teams={team_ids}. Check `/models` to see all available models.",
            )

        return None, None

    @staticmethod
    async def get_user_info(
        jwt_handler: JWTHandler,
        jwt_valid_token: dict,
    ) -> Tuple[Optional[str], Optional[str], Optional[bool]]:
        """Get user email and validation status"""
        user_email = jwt_handler.get_user_email(
            token=jwt_valid_token, default_value=None
        )
        valid_user_email = None
        if jwt_handler.is_enforced_email_domain():
            valid_user_email = (
                False
                if user_email is None
                else jwt_handler.is_allowed_domain(user_email=user_email)
            )
        user_id = jwt_handler.get_user_id(
            token=jwt_valid_token, default_value=user_email
        )
        return user_id, user_email, valid_user_email

    @staticmethod
    async def get_objects(
        user_id: Optional[str],
        user_email: Optional[str],
        org_id: Optional[str],
        end_user_id: Optional[str],
        valid_user_email: Optional[bool],
        jwt_handler: JWTHandler,
        prisma_client: Optional[PrismaClient],
        user_api_key_cache: DualCache,
        parent_otel_span: Optional[Span],
        proxy_logging_obj: ProxyLogging,
    ) -> Tuple[
        Optional[LiteLLM_UserTable],
        Optional[LiteLLM_OrganizationTable],
        Optional[LiteLLM_EndUserTable],
    ]:
        """Get user, org, and end user objects"""
        org_object: Optional[LiteLLM_OrganizationTable] = None
        if org_id:
            org_object = (
                await get_org_object(
                    org_id=org_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
                if org_id
                else None
            )

        user_object: Optional[LiteLLM_UserTable] = None
        if user_id:
            user_object = (
                await get_user_object(
                    user_id=user_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    user_id_upsert=jwt_handler.is_upsert_user_id(
                        valid_user_email=valid_user_email
                    ),
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                    user_email=user_email,
                    sso_user_id=user_id,
                )
                if user_id
                else None
            )

        end_user_object: Optional[LiteLLM_EndUserTable] = None
        if end_user_id:
            end_user_object = (
                await get_end_user_object(
                    end_user_id=end_user_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
                if end_user_id
                else None
            )

        return user_object, org_object, end_user_object

    @staticmethod
    def validate_object_id(
        user_id: Optional[str],
        team_id: Optional[str],
        enforce_rbac: bool,
        is_proxy_admin: bool,
    ) -> Literal[True]:
        """If enforce_rbac is true, validate that a valid rbac id is returned for spend tracking"""
        if enforce_rbac and not is_proxy_admin and not user_id and not team_id:
            raise HTTPException(
                status_code=403,
                detail="No user or team id found in token. enforce_rbac is set to True. Token must belong to a proxy admin, team, or user.",
            )
        return True

    @staticmethod
    async def auth_builder(
        api_key: str,
        jwt_handler: JWTHandler,
        request_data: dict,
        general_settings: dict,
        route: str,
        prisma_client: Optional[PrismaClient],
        user_api_key_cache: DualCache,
        parent_otel_span: Optional[Span],
        proxy_logging_obj: ProxyLogging,
    ) -> JWTAuthBuilderResult:
        """Main authentication and authorization builder"""
        jwt_valid_token: dict = await jwt_handler.auth_jwt(token=api_key)

        # Check RBAC
        rbac_role = jwt_handler.get_rbac_role(token=jwt_valid_token)
        await JWTAuthManager.check_rbac_role(
            jwt_handler,
            jwt_valid_token,
            general_settings,
            request_data,
            route,
            rbac_role,
        )

        # Check Scope Based Access
        scopes = jwt_handler.get_scopes(token=jwt_valid_token)
        if (
            jwt_handler.litellm_jwtauth.enforce_scope_based_access
            and jwt_handler.litellm_jwtauth.scope_mappings
        ):
            JWTAuthManager.check_scope_based_access(
                scope_mappings=jwt_handler.litellm_jwtauth.scope_mappings,
                scopes=scopes,
                request_data=request_data,
                general_settings=general_settings,
            )

        object_id = jwt_handler.get_object_id(token=jwt_valid_token, default_value=None)

        # Get basic user info
        scopes = jwt_handler.get_scopes(token=jwt_valid_token)
        user_id, user_email, valid_user_email = await JWTAuthManager.get_user_info(
            jwt_handler, jwt_valid_token
        )

        # Get IDs
        org_id = jwt_handler.get_org_id(token=jwt_valid_token, default_value=None)
        end_user_id = jwt_handler.get_end_user_id(
            token=jwt_valid_token, default_value=None
        )
        team_id: Optional[str] = None
        team_object: Optional[LiteLLM_TeamTable] = None
        object_id = jwt_handler.get_object_id(token=jwt_valid_token, default_value=None)

        if rbac_role and object_id:

            if rbac_role == LitellmUserRoles.TEAM:
                team_id = object_id
            elif rbac_role == LitellmUserRoles.INTERNAL_USER:
                user_id = object_id

        # Check admin access
        admin_result = await JWTAuthManager.check_admin_access(
            jwt_handler, scopes, route, user_id, org_id, api_key
        )
        if admin_result:
            return admin_result

        # Get team with model access
        ## SPECIFIC TEAM ID

        if not team_id:
            team_id, team_object = (
                await JWTAuthManager.find_and_validate_specific_team_id(
                    jwt_handler,
                    jwt_valid_token,
                    prisma_client,
                    user_api_key_cache,
                    parent_otel_span,
                    proxy_logging_obj,
                )
            )

        if not team_object and not team_id:
            ## CHECK USER GROUP ACCESS
            all_team_ids = JWTAuthManager.get_all_team_ids(jwt_handler, jwt_valid_token)
            team_id, team_object = await JWTAuthManager.find_team_with_model_access(
                team_ids=all_team_ids,
                requested_model=request_data.get("model"),
                route=route,
                jwt_handler=jwt_handler,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )

        # Get other objects
        user_object, org_object, end_user_object = await JWTAuthManager.get_objects(
            user_id=user_id,
            user_email=user_email,
            org_id=org_id,
            end_user_id=end_user_id,
            valid_user_email=valid_user_email,
            jwt_handler=jwt_handler,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )

        # Validate that a valid rbac id is returned for spend tracking
        JWTAuthManager.validate_object_id(
            user_id=user_id,
            team_id=team_id,
            enforce_rbac=general_settings.get("enforce_rbac", False),
            is_proxy_admin=False,
        )

        return JWTAuthBuilderResult(
            is_proxy_admin=False,
            team_id=team_id,
            team_object=team_object,
            user_id=user_id,
            user_object=user_object,
            org_id=org_id,
            org_object=org_object,
            end_user_id=end_user_id,
            end_user_object=end_user_object,
            token=api_key,
        )
