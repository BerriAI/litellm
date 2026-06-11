from __future__ import annotations

import secrets
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from saml2 import BINDING_HTTP_POST
from saml2.client import Saml2Client
from saml2.config import SPConfig
from saml2.metadata import entity_descriptor
from scim2_models import Email, Name
from scim2_models import User as ScimUser

from .config import SamlConfig
from .models import AuthMethod, Credential, CredentialRef, SecuritySchemeType
from .resolver import ProvisioningStore

_SINGLE_VALUE_TARGETS = {
    "email",
    "given_name",
    "family_name",
    "user_name",
    "display_name",
}
_MULTI_VALUE_TARGETS = ("groups", "roles")


def _map_attributes(
    ava: Dict[str, Any], attribute_map: Dict[str, str]
) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {}
    for saml_attr, target in attribute_map.items():
        if saml_attr not in ava:
            continue
        value = ava[saml_attr]
        if target in _SINGLE_VALUE_TARGETS:
            scalar = value[0] if isinstance(value, list) and value else value
            mapped.setdefault(target, scalar)
        elif target in _MULTI_VALUE_TARGETS and isinstance(value, list):
            mapped[target] = value
    return mapped


def _formatted_name(mapped: Dict[str, Any]) -> Optional[str]:
    if mapped.get("display_name"):
        return mapped["display_name"]
    parts: List[str] = [
        part for part in (mapped.get("given_name"), mapped.get("family_name")) if part
    ]
    return " ".join(parts) if parts else None


def _user_from_mapped(name_id: str, mapped: Dict[str, Any]) -> ScimUser:
    display = _formatted_name(mapped)
    user = ScimUser(
        external_id=name_id,
        user_name=mapped.get("user_name") or mapped.get("email") or name_id,
        display_name=display,
    )
    if mapped.get("given_name") or mapped.get("family_name"):
        user.name = Name(
            given_name=mapped.get("given_name"),
            family_name=mapped.get("family_name"),
            formatted=display,
        )
    if mapped.get("email"):
        user.emails = [Email(value=mapped["email"], primary=True)]
    return user


def _claims_from_mapped(mapped: Dict[str, Any]) -> Dict[str, Any]:
    claims: Dict[str, Any] = {}
    if mapped.get("email"):
        claims["email"] = mapped["email"]
    if mapped.get("user_name"):
        claims["preferred_username"] = mapped["user_name"]
    display = _formatted_name(mapped)
    if display:
        claims["name"] = display
    for target in _MULTI_VALUE_TARGETS:
        if mapped.get(target):
            claims[target] = mapped[target]
    return claims


def _safe_relay_state(target: Optional[str], default: str) -> str:
    if (
        target
        and target.startswith("/")
        and not target.startswith("//")
        and "://" not in target
        and "\\" not in target
    ):
        return target
    return default


def _metadata_source(idp_metadata: str) -> Dict[str, Any]:
    stripped = idp_metadata.strip()
    if stripped.startswith("<"):
        return {"inline": [idp_metadata]}
    if stripped.startswith("http://") or stripped.startswith("https://"):
        return {"remote": [{"url": idp_metadata}]}
    return {"local": [idp_metadata]}


def _sp_config_dict(config: SamlConfig) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "entityid": config.entity_id,
        "service": {
            "sp": {
                "endpoints": {
                    "assertion_consumer_service": [(config.acs_url, BINDING_HTTP_POST)]
                },
                "allow_unsolicited": config.allow_unsolicited,
                "authn_requests_signed": False,
                "want_assertions_signed": True,
                "want_response_signed": False,
            }
        },
        "metadata": _metadata_source(config.idp_metadata),
        "allow_unknown_attributes": True,
    }
    if config.sp_key_file:
        cfg["key_file"] = config.sp_key_file
    if config.sp_cert_file:
        cfg["cert_file"] = config.sp_cert_file
    if config.xmlsec_binary:
        cfg["xmlsec_binary"] = config.xmlsec_binary
    return cfg


def build_sp_client(config: SamlConfig) -> Saml2Client:
    conf = SPConfig()
    conf.load(_sp_config_dict(config))
    return Saml2Client(config=conf)


class SamlSessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self.outstanding: Dict[str, str] = {}

    def remember_request(self, request_id: str, relay_state: str = "/") -> None:
        self.outstanding[request_id] = relay_state

    def create_session(self, identity: Dict[str, Any]) -> str:
        session_id = secrets.token_urlsafe(32)
        self._sessions[session_id] = identity
        return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)


class SamlAuthenticator:
    scheme = SecuritySchemeType.HTTP

    def __init__(self, config: SamlConfig, session_store: SamlSessionStore) -> None:
        self._config = config
        self._store = session_store

    async def authenticate(self, request: Request) -> Optional[Credential]:
        session_id = request.cookies.get(self._config.session_cookie)
        if not session_id:
            return None
        identity = self._store.get(session_id)
        if identity is None:
            return None
        return Credential(
            scheme=self.scheme,
            method=AuthMethod.SAML,
            subject=identity["name_id"],
            issuer=identity.get("issuer"),
            claims=identity.get("claims", {}),
            credential_ref=CredentialRef(token_id=session_id),
        )

    def challenge(self) -> str:
        return ""


def build_saml_router(config: SamlConfig, session_store: SamlSessionStore) -> APIRouter:
    client = build_sp_client(config)
    router = APIRouter(prefix="/auth/saml", tags=["saml"])

    @router.get("/metadata")
    async def metadata() -> Response:
        return Response(
            content=str(entity_descriptor(client.config)),
            media_type="application/samlmetadata+xml",
        )

    @router.get("/login")
    async def login(request: Request) -> RedirectResponse:
        relay_state = _safe_relay_state(
            request.query_params.get("next"), config.default_redirect_path
        )
        request_id, info = client.prepare_for_authenticate(relay_state=relay_state)
        session_store.remember_request(request_id, relay_state)
        location = dict(info["headers"]).get("Location")
        if not location:
            raise HTTPException(status_code=500, detail="no SAML redirect produced")
        return RedirectResponse(location, status_code=303)

    @router.post("/acs")
    async def assertion_consumer_service(request: Request) -> Response:
        form = await request.form()
        saml_response = form.get("SAMLResponse")
        if not isinstance(saml_response, str):
            raise HTTPException(status_code=400, detail="missing SAMLResponse")
        try:
            authn_response = client.parse_authn_request_response(
                saml_response,
                BINDING_HTTP_POST,
                outstanding=session_store.outstanding or None,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=401, detail="invalid SAML response"
            ) from exc
        if authn_response is None:
            raise HTTPException(status_code=401, detail="invalid SAML response")

        name_id = authn_response.get_subject().text
        ava = authn_response.get_identity() or {}
        mapped = _map_attributes(ava, config.attribute_map)
        user = _user_from_mapped(name_id, mapped)

        store: ProvisioningStore = request.app.state.auth_v2.resolver
        await store.upsert_user(user)

        in_response_to = getattr(authn_response, "in_response_to", None)
        if in_response_to:
            session_store.outstanding.pop(in_response_to, None)
        session_id = session_store.create_session(
            {
                "name_id": name_id,
                "issuer": authn_response.issuer(),
                "claims": _claims_from_mapped(mapped),
            }
        )
        relay_state = form.get("RelayState")
        target = _safe_relay_state(
            relay_state if isinstance(relay_state, str) else None,
            config.default_redirect_path,
        )
        response = RedirectResponse(target, status_code=303)
        response.set_cookie(
            config.session_cookie, session_id, httponly=True, samesite="lax"
        )
        return response

    return router
