from __future__ import annotations

import secrets
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from saml2 import BINDING_HTTP_POST
from saml2.client import Saml2Client
from saml2.config import SPConfig
from saml2.metadata import entity_descriptor
from scim2_models import User as ScimUser

from .config import SamlConfig
from .models import AuthMethod, Credential, CredentialRef, SecuritySchemeType
from .resolver import ProvisioningStore

_SINGLE_VALUE_CLAIM = {
    "email": "email",
    "user_name": "preferred_username",
    "display_name": "name",
}
_MULTI_VALUE_ATTRS = ("groups", "roles")


def _normalize_attributes(
    ava: Dict[str, Any], attribute_map: Dict[str, str]
) -> Dict[str, Any]:
    claims: Dict[str, Any] = {}
    for saml_attr, target in attribute_map.items():
        if saml_attr not in ava or target not in _SINGLE_VALUE_CLAIM:
            continue
        value = ava[saml_attr]
        scalar = value[0] if isinstance(value, list) and value else value
        claims.setdefault(_SINGLE_VALUE_CLAIM[target], scalar)
    for attr in _MULTI_VALUE_ATTRS:
        value = ava.get(attr)
        if isinstance(value, list):
            claims[attr] = value
    return claims


def _user_from_claims(name_id: str, claims: Dict[str, Any]) -> ScimUser:
    return ScimUser(
        external_id=name_id,
        user_name=claims.get("preferred_username") or claims.get("email") or name_id,
        display_name=claims.get("name"),
    )


def _sp_config_dict(config: SamlConfig) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "entityid": config.sp_entity_id,
        "service": {
            "sp": {
                "endpoints": {
                    "assertion_consumer_service": [(config.acs_url, BINDING_HTTP_POST)]
                },
                "allow_unsolicited": config.allow_unsolicited,
                "authn_requests_signed": False,
                "want_assertions_signed": config.want_assertions_signed,
                "want_response_signed": False,
            }
        },
        "allow_unknown_attributes": True,
    }
    if config.idp_metadata_path:
        cfg["metadata"] = {"local": [config.idp_metadata_path]}
    elif config.idp_metadata_inline:
        cfg["metadata"] = {"inline": [config.idp_metadata_inline]}
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
    async def login() -> RedirectResponse:
        request_id, info = client.prepare_for_authenticate()
        session_store.remember_request(request_id)
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
        authn_response = client.parse_authn_request_response(
            saml_response,
            BINDING_HTTP_POST,
            outstanding=session_store.outstanding or None,
        )
        if authn_response is None:
            raise HTTPException(status_code=401, detail="invalid SAML response")

        name_id = authn_response.get_subject().text
        ava = authn_response.get_identity() or {}
        claims = _normalize_attributes(ava, config.attribute_map)
        user = _user_from_claims(name_id, claims)

        store: ProvisioningStore = request.app.state.auth_v2.resolver
        await store.upsert_user(user)

        in_response_to = getattr(authn_response, "in_response_to", None)
        if in_response_to:
            session_store.outstanding.pop(in_response_to, None)
        session_id = session_store.create_session(
            {"name_id": name_id, "issuer": authn_response.issuer(), "claims": claims}
        )
        response = JSONResponse(content=user.model_dump())
        response.set_cookie(
            config.session_cookie, session_id, httponly=True, samesite="lax"
        )
        return response

    return router
