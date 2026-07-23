"""
SAML 2.0 SSO for the LiteLLM proxy admin UI.

Supports both SP-initiated and IdP-initiated login via the HTTP-POST binding,
using the OneLogin python3-saml toolkit for signature, audience and time
validation. The IdP is configured from its metadata (``SAML_IDP_METADATA_URL``
or inline ``SAML_IDP_METADATA_XML``); a successful login is mapped to a
``CustomOpenID`` and handed to the shared post-login path used by every other
SSO provider.

python3-saml pulls in the native ``xmlsec``/``libxml2`` libraries, so it is an
optional dependency. When it is not installed the SAML routes return a clear
error instead of breaking proxy startup.
"""

# python3-saml ships no type stubs, so the type checker sees every onelogin call
# as Unknown and the guarded optional import as possibly-unbound. Values crossing
# that boundary are cast() to concrete types at each use site; these directives
# silence only the unavoidable noise from the untyped dependency in this module.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingTypeStubs=false, reportPossiblyUnboundVariable=false
# pyright: reportConstantRedefinition=false

import asyncio
import hashlib
import os
import secrets
import time
from typing import cast
from urllib.parse import parse_qsl

from fastapi import HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.proxy.management_endpoints.types import CustomOpenID, get_litellm_user_role
from litellm.proxy.utils import get_custom_url

try:
    from onelogin.saml2.auth import OneLogin_Saml2_Auth
    from onelogin.saml2.idp_metadata_parser import OneLogin_Saml2_IdPMetadataParser
    from onelogin.saml2.settings import OneLogin_Saml2_Settings
    from onelogin.saml2.xml_utils import OneLogin_Saml2_XML

    SAML_AVAILABLE = True
except ImportError:
    SAML_AVAILABLE = False

SAML_LOGIN_ROUTE = "sso/saml/login"
SAML_CALLBACK_ROUTE = "sso/saml/callback"
SAML_METADATA_ROUTE = "sso/saml/metadata"

_SAML_AUTHN_STATE_COOKIE = "litellm_saml_authn"
_SAML_IDP_SETTINGS_CACHE_PREFIX = "saml_idp_settings"
_SAML_AUTHN_REQUEST_CACHE_PREFIX = "saml_authn_request"
_SAML_CONSUMED_ASSERTION_CACHE_PREFIX = "saml_consumed_assertion"
_SAML_AUTHN_REQUEST_TTL_SECONDS = 600
_SAML_IDP_METADATA_TTL_SECONDS = 3600
_SAML_METADATA_FETCH_TIMEOUT_SECONDS = 10
_SAML_MAX_POST_BYTES = 5 * 1024 * 1024
# The replay guard tracks each assertion's NotOnOrAfter so it spans the full
# validity window; the floor covers IdPs that issue hour-long assertions or omit
# the timestamp, and the cap bounds cache growth.
_SAML_REPLAY_GUARD_DEFAULT_TTL_SECONDS = 3600
_SAML_REPLAY_GUARD_MAX_TTL_SECONDS = 86400

_EMAIL_ATTRIBUTE_CANDIDATES = (
    "urn:oid:0.9.2342.19200300.100.1.3",
    "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
    "email",
    "emailAddress",
    "mail",
    "Email",
)
_FIRST_NAME_ATTRIBUTE_CANDIDATES = (
    "urn:oid:2.5.4.42",
    "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname",
    "givenName",
    "first_name",
    "firstName",
)
_LAST_NAME_ATTRIBUTE_CANDIDATES = (
    "urn:oid:2.5.4.4",
    "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname",
    "sn",
    "surname",
    "last_name",
    "lastName",
)
_ROLE_ATTRIBUTE_CANDIDATES = ("role", "roles", "litellm_role")
_TEAM_IDS_ATTRIBUTE_CANDIDATES = ("teams", "team_ids", "groups")


def _saml_unavailable_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "SAML SSO requires the optional 'python3-saml' dependency, which is "
            "not installed. Re-install litellm with the saml extra: "
            "'pip install litellm[saml]'. The saml extra bundles the native "
            "xmlsec/libxml2 libraries, so no system packages are required."
        ),
    )


class SAMLAuthHandler:
    """SP- and IdP-initiated SAML 2.0 login for the admin UI."""

    @staticmethod
    def _env(name: str, default: str | None = None) -> str | None:
        return os.getenv(name, default)

    @staticmethod
    def is_saml_configured() -> bool:
        return bool(SAMLAuthHandler._env("SAML_IDP_METADATA_URL") or SAMLAuthHandler._env("SAML_IDP_METADATA_XML"))

    @staticmethod
    def _bool_env(name: str, default: bool) -> bool:
        raw = SAMLAuthHandler._env(name)
        if raw is None:
            return default
        return raw.strip().lower() in ("true", "1", "yes", "on")

    @staticmethod
    def _base_url(request: Request) -> str:
        base = get_custom_url(request_base_url=str(request.base_url))
        return base if base.endswith("/") else base + "/"

    @staticmethod
    def _is_https(request: Request) -> bool:
        return SAMLAuthHandler._base_url(request).startswith("https")

    @staticmethod
    def _acs_url(request: Request) -> str:
        return SAMLAuthHandler._base_url(request) + SAML_CALLBACK_ROUTE

    @staticmethod
    def _metadata_url(request: Request) -> str:
        return SAMLAuthHandler._base_url(request) + SAML_METADATA_ROUTE

    @staticmethod
    def _sp_entity_id(request: Request) -> str:
        return SAMLAuthHandler._env("SAML_SP_ENTITY_ID") or SAMLAuthHandler._metadata_url(request)

    @staticmethod
    async def _load_idp_settings(cache: DualCache) -> dict[str, object]:
        metadata_url = SAMLAuthHandler._env("SAML_IDP_METADATA_URL")
        metadata_xml = SAMLAuthHandler._env("SAML_IDP_METADATA_XML")
        source = metadata_url or metadata_xml
        if source is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="SAML SSO is not configured. Set SAML_IDP_METADATA_URL or SAML_IDP_METADATA_XML.",
            )

        cache_key = f"{_SAML_IDP_SETTINGS_CACHE_PREFIX}:{hashlib.sha256(source.encode()).hexdigest()}"
        cached = cache.get_cache(key=cache_key)
        if isinstance(cached, dict):
            return cast(dict[str, object], cached)  # cast-ok: untyped python3-saml

        if metadata_url is not None:
            parsed = await asyncio.to_thread(
                OneLogin_Saml2_IdPMetadataParser.parse_remote,
                metadata_url,
                validate_cert=SAMLAuthHandler._bool_env("SAML_IDP_METADATA_VALIDATE_CERT", True),
                timeout=_SAML_METADATA_FETCH_TIMEOUT_SECONDS,
            )
        else:
            parsed = OneLogin_Saml2_IdPMetadataParser.parse(cast(str, metadata_xml))  # cast-ok: untyped python3-saml

        idp_settings = cast(dict[str, object], parsed)  # cast-ok: untyped python3-saml
        if not idp_settings.get("idp"):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Could not parse an IdP entityID/SSO URL/certificate from the SAML metadata.",
            )
        cache.set_cache(key=cache_key, value=idp_settings, ttl=_SAML_IDP_METADATA_TTL_SECONDS)
        return idp_settings

    @staticmethod
    def _build_settings(request: Request, idp_settings: dict[str, object]) -> dict[str, object]:
        sp_settings: dict[str, object] = {
            "strict": SAMLAuthHandler._bool_env("SAML_STRICT", True),
            "debug": False,
            "sp": {
                "entityId": SAMLAuthHandler._sp_entity_id(request),
                "assertionConsumerService": {
                    "url": SAMLAuthHandler._acs_url(request),
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
                },
                "NameIDFormat": SAMLAuthHandler._env(
                    "SAML_SP_NAME_ID_FORMAT",
                    "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress",
                ),
            },
            "security": {
                "wantAssertionsSigned": SAMLAuthHandler._bool_env("SAML_WANT_ASSERTIONS_SIGNED", True),
                "wantMessagesSigned": SAMLAuthHandler._bool_env("SAML_WANT_MESSAGES_SIGNED", False),
                "authnRequestsSigned": SAMLAuthHandler._bool_env("SAML_AUTHN_REQUESTS_SIGNED", False),
                "wantNameId": True,
                "requestedAuthnContext": False,
                "rejectUnsolicitedResponsesWithInResponseTo": False,
            },
        }
        return OneLogin_Saml2_IdPMetadataParser.merge_settings(sp_settings, idp_settings)

    @staticmethod
    def _prepare_request_data(request: Request, post_data: dict[str, str] | None = None) -> dict[str, object]:
        base = SAMLAuthHandler._base_url(request)
        scheme, _, host_part = base.partition("://")
        host = host_part.split("/", 1)[0]
        return {
            "https": "on" if scheme == "https" else "off",
            "http_host": host,
            "script_name": "/" + SAML_CALLBACK_ROUTE,
            "get_data": dict(request.query_params),
            "post_data": post_data or {},
        }

    @staticmethod
    async def _build_auth(
        request: Request,
        cache: DualCache,
        post_data: dict[str, str] | None = None,
    ) -> "OneLogin_Saml2_Auth":
        if not SAML_AVAILABLE:
            raise _saml_unavailable_error()
        idp_settings = await SAMLAuthHandler._load_idp_settings(cache)
        settings = SAMLAuthHandler._build_settings(request, idp_settings)
        request_data = SAMLAuthHandler._prepare_request_data(request, post_data)
        try:
            return OneLogin_Saml2_Auth(request_data, old_settings=settings)
        except Exception as e:  # noqa: BLE001 - toolkit exposes no common exception base; fail closed
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid SAML configuration: {e}",
            )

    @staticmethod
    async def build_login_redirect(
        request: Request, cache: DualCache, relay_state: str | None = None
    ) -> RedirectResponse:
        auth = await SAMLAuthHandler._build_auth(request, cache)
        redirect_url = cast(str, auth.login(return_to=relay_state))  # cast-ok: untyped python3-saml
        response = RedirectResponse(url=redirect_url, status_code=303)
        request_id = cast(str | None, auth.get_last_request_id())  # cast-ok: untyped python3-saml
        if request_id is not None:
            cache.set_cache(
                key=f"{_SAML_AUTHN_REQUEST_CACHE_PREFIX}:{request_id}",
                value="1",
                ttl=_SAML_AUTHN_REQUEST_TTL_SECONDS,
            )
            secure = SAMLAuthHandler._is_https(request)
            response.set_cookie(
                key=_SAML_AUTHN_STATE_COOKIE,
                value=request_id,
                max_age=_SAML_AUTHN_REQUEST_TTL_SECONDS,
                httponly=True,
                secure=secure,
                samesite="none" if secure else "lax",
            )
        return response

    @staticmethod
    async def build_sp_metadata(request: Request, cache: DualCache) -> str:
        if not SAML_AVAILABLE:
            raise _saml_unavailable_error()
        idp_settings = await SAMLAuthHandler._load_idp_settings(cache)
        settings = SAMLAuthHandler._build_settings(request, idp_settings)
        saml_settings = OneLogin_Saml2_Settings(settings, sp_validation_only=True)
        metadata = cast(str, saml_settings.get_sp_metadata())  # cast-ok: untyped python3-saml
        errors = cast(list[str], saml_settings.validate_metadata(metadata))  # cast-ok: untyped python3-saml
        if errors:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid SP metadata: {', '.join(errors)}",
            )
        return metadata

    @staticmethod
    async def read_acs_post_data(request: Request) -> dict[str, str]:
        """Read the ACS POST form under a hard size cap before any base64/XML decoding.

        Bounds both Content-Length-declared and chunked requests so an unauthenticated
        caller cannot force unbounded buffering while decoding the SAMLResponse."""
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > _SAML_MAX_POST_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="SAML response exceeds the maximum allowed size.",
            )

        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > _SAML_MAX_POST_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="SAML response exceeds the maximum allowed size.",
                )

        return dict(parse_qsl(body.decode("utf-8", "replace")))

    @staticmethod
    async def handle_acs(request: Request, cache: DualCache, post_data: dict[str, str]) -> CustomOpenID:
        auth = await SAMLAuthHandler._build_auth(request, cache, post_data=post_data)
        browser_request_id = request.cookies.get(_SAML_AUTHN_STATE_COOKIE)
        try:
            auth.process_response(request_id=browser_request_id)
        except Exception as e:  # noqa: BLE001 - toolkit exposes no common exception base; fail closed
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Could not process SAML response: {e}",
            )

        errors = cast(list[str], auth.get_errors())  # cast-ok: untyped python3-saml
        if errors or not auth.is_authenticated():
            reason = auth.get_last_error_reason()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"SAML authentication failed: {reason or ', '.join(errors)}",
            )

        await SAMLAuthHandler._enforce_response_binding(auth, cache, browser_request_id)
        return SAMLAuthHandler._result_from_auth(auth)

    @staticmethod
    def _replay_guard_ttl(auth: "OneLogin_Saml2_Auth") -> int:
        not_on_or_after = auth.get_last_assertion_not_on_or_after()
        if not isinstance(not_on_or_after, int):
            return _SAML_REPLAY_GUARD_DEFAULT_TTL_SECONDS
        remaining = not_on_or_after - int(time.time())
        return min(
            max(remaining, _SAML_REPLAY_GUARD_DEFAULT_TTL_SECONDS),
            _SAML_REPLAY_GUARD_MAX_TTL_SECONDS,
        )

    @staticmethod
    def _response_in_response_to(auth: "OneLogin_Saml2_Auth") -> str | None:
        """The request id this response answers, read from the Response element or, when the
        IdP only stamps it on the bearer SubjectConfirmationData, from there. A non-None value
        marks the response as solicited (SP-initiated) and so requiring browser binding."""
        value = cast(str | None, auth.get_last_response_in_response_to())  # cast-ok: untyped python3-saml
        if value:
            return value
        xml = cast(bytes | None, auth.get_last_response_xml())  # cast-ok: untyped python3-saml
        if not xml:
            return None
        root = OneLogin_Saml2_XML.to_etree(xml)
        for node in OneLogin_Saml2_XML.query(root, "//saml:SubjectConfirmationData[@InResponseTo]"):
            irt = cast(str | None, node.get("InResponseTo"))  # cast-ok: untyped python3-saml
            if irt:
                return irt
        return None

    @staticmethod
    async def _enforce_response_binding(
        auth: "OneLogin_Saml2_Auth",
        cache: DualCache,
        browser_request_id: str | None,
    ) -> None:
        in_response_to = SAMLAuthHandler._response_in_response_to(auth)

        if in_response_to is not None:
            authn_key = f"{_SAML_AUTHN_REQUEST_CACHE_PREFIX}:{in_response_to}"
            if cache.get_cache(key=authn_key) is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="SAML response references an unknown or already-used login request.",
                )
            if browser_request_id is None or not secrets.compare_digest(browser_request_id, in_response_to):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="SAML response is not bound to this browser's login request.",
                )
        elif browser_request_id is not None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="SAML response is not bound to this browser's login request.",
            )
        elif not SAMLAuthHandler._bool_env("SAML_ALLOW_UNSOLICITED", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unsolicited (IdP-initiated) SAML responses are disabled.",
            )
        elif cache.redis_cache is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Unsolicited (IdP-initiated) SAML responses require a shared Redis cache "
                    "so the replay guard is enforced across every worker."
                ),
            )

        assertion_id = cast(str | None, auth.get_last_assertion_id())  # cast-ok: untyped python3-saml
        if assertion_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="SAML assertion is missing the required ID attribute.",
            )
        consumed_key = f"{_SAML_CONSUMED_ASSERTION_CACHE_PREFIX}:{assertion_id}"
        consumed_count = await cache.async_increment_cache(
            key=consumed_key, value=1, ttl=SAMLAuthHandler._replay_guard_ttl(auth)
        )
        if consumed_count is not None and consumed_count > 1:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="SAML assertion has already been used (replay detected).",
            )

    @staticmethod
    def _result_from_auth(auth: "OneLogin_Saml2_Auth") -> CustomOpenID:
        attributes = cast(dict[str, list[str]], auth.get_attributes())  # cast-ok: untyped python3-saml
        name_id = cast(str | None, auth.get_nameid())  # cast-ok: untyped python3-saml

        email = SAMLAuthHandler._attribute_value(attributes, "SAML_ATTRIBUTE_EMAIL", _EMAIL_ATTRIBUTE_CANDIDATES)
        if email is None and name_id is not None and "@" in name_id:
            email = name_id

        if email is None and SAMLAuthHandler._env("ALLOWED_EMAIL_DOMAINS") is not None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "SAML assertion did not contain an email address, but ALLOWED_EMAIL_DOMAINS "
                    "restricts sign-in by email domain."
                ),
            )

        user_id = SAMLAuthHandler._attribute_value(attributes, "SAML_ATTRIBUTE_USER_ID", ()) or name_id or email
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="SAML assertion did not contain a usable subject (NameID) or email.",
            )

        first_name = SAMLAuthHandler._attribute_value(
            attributes, "SAML_ATTRIBUTE_FIRST_NAME", _FIRST_NAME_ATTRIBUTE_CANDIDATES
        )
        last_name = SAMLAuthHandler._attribute_value(
            attributes, "SAML_ATTRIBUTE_LAST_NAME", _LAST_NAME_ATTRIBUTE_CANDIDATES
        )
        role_value = SAMLAuthHandler._attribute_value(attributes, "SAML_ATTRIBUTE_ROLE", _ROLE_ATTRIBUTE_CANDIDATES)
        team_ids = SAMLAuthHandler._attribute_values(
            attributes, "SAML_ATTRIBUTE_TEAM_IDS", _TEAM_IDS_ATTRIBUTE_CANDIDATES
        )

        display_name = " ".join(part for part in (first_name, last_name) if part) or email

        verbose_proxy_logger.info(f"SAML login: subject={user_id}, email={email}, attributes={list(attributes.keys())}")

        try:
            return CustomOpenID(
                id=user_id,
                email=email,
                first_name=first_name,
                last_name=last_name,
                display_name=display_name,
                picture=None,
                provider="saml",
                team_ids=team_ids,
                user_role=get_litellm_user_role(role_value) if role_value else None,
            )
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"SAML assertion contained an invalid subject or email: {e}",
            )

    @staticmethod
    def _attribute_value(
        attributes: dict[str, list[str]],
        env_override: str,
        candidates: tuple[str, ...],
    ) -> str | None:
        values = SAMLAuthHandler._attribute_values(attributes, env_override, candidates)
        return values[0] if values else None

    @staticmethod
    def _attribute_values(
        attributes: dict[str, list[str]],
        env_override: str,
        candidates: tuple[str, ...],
    ) -> list[str]:
        override = SAMLAuthHandler._env(env_override)
        keys = (override, *candidates) if override else candidates
        for key in keys:
            values = attributes.get(key)
            if values:
                return [v for v in values if v]
        return []
