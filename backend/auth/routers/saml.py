from __future__ import annotations

import secrets
from typing import Tuple, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse, Response
from saml2 import BINDING_HTTP_POST
from saml2.client import Saml2Client
from saml2.metadata import entity_descriptor

from litellm.proxy.auth_v2 import errors
from litellm.proxy.auth_v2.models import AuthMethod
from litellm.proxy.auth_v2.authorization import filter_claim_roles
from ..services.redirects import safe_relay_state
from litellm.proxy.auth_v2.resolvers import ProvisioningStore
from litellm.proxy.auth_v2.security import AuthSecurity
from litellm.proxy.auth_v2.sessions.types import SessionState

from ..services.saml import (
    SAMLProtocolStore,
    claims_from_mapped,
    map_attributes,
    user_from_mapped,
)
from .dependencies import get_auth, get_saml_runtime

router = APIRouter(prefix="/auth/saml", tags=["saml"])


@router.get("/metadata")
async def metadata(
    runtime: Tuple[Saml2Client, SAMLProtocolStore] = Depends(get_saml_runtime),
) -> Response:
    client, _ = runtime
    return Response(
        content=str(entity_descriptor(client.config)),
        media_type="application/samlmetadata+xml",
    )


@router.get("/login")
async def login(
    request: Request,
    auth: AuthSecurity = Depends(get_auth),
    runtime: Tuple[Saml2Client, SAMLProtocolStore] = Depends(get_saml_runtime),
) -> RedirectResponse:
    session = auth.config.session
    client, protocol = runtime
    relay_state = safe_relay_state(request.query_params.get("next"), session.default_redirect_path)
    request_id, info = client.prepare_for_authenticate(relay_state=relay_state)
    protocol.remember_request(request_id, relay_state)
    location = dict(info["headers"]).get("Location")
    if not location:
        raise errors.saml_redirect_failed()
    return RedirectResponse(location, status_code=303)


@router.post("/acs")
async def assertion_consumer_service(
    request: Request,
    auth: AuthSecurity = Depends(get_auth),
    runtime: Tuple[Saml2Client, SAMLProtocolStore] = Depends(get_saml_runtime),
) -> Response:
    config = auth.config.saml
    assert config is not None
    session = auth.config.session
    client, protocol = runtime

    form = await request.form()
    saml_response = form.get("SAMLResponse")
    if not isinstance(saml_response, str):
        raise errors.missing_saml_response()
    try:
        authn_response = client.parse_authn_request_response(
            saml_response,
            BINDING_HTTP_POST,
            outstanding=protocol.outstanding_relays() or None,
        )
    except Exception as exc:
        raise errors.invalid_saml_response() from exc
    if authn_response is None:
        raise errors.invalid_saml_response()

    in_response_to = getattr(authn_response, "in_response_to", None)
    bound_relay = protocol.consume_request(in_response_to) if in_response_to else None

    assertion = getattr(authn_response, "assertion", None)
    assertion_id = getattr(assertion, "id", None)
    if assertion_id and not protocol.consume_assertion(assertion_id):
        raise errors.saml_assertion_replay()

    name_id = authn_response.get_subject().text
    ava = authn_response.get_identity() or {}
    mapped = map_attributes(ava, config.attribute_map)
    mapped["roles"] = filter_claim_roles(mapped.get("roles"), config.allowed_roles, config.allow_platform_roles)
    user = user_from_mapped(name_id, mapped)

    store = cast(ProvisioningStore, auth.resolver)
    await store.upsert_user(user)

    session_id = secrets.token_urlsafe(32)
    await auth.session_store.set(
        session_id,
        SessionState(
            method=AuthMethod.SAML.value,
            subject=name_id,
            issuer=authn_response.issuer(),
            claims=claims_from_mapped(mapped),
        ),
    )
    target = safe_relay_state(bound_relay, session.default_redirect_path)
    response = RedirectResponse(target, status_code=303)
    response.set_cookie(
        session.cookie,
        session_id,
        httponly=True,
        samesite="lax",
        secure=session.secure,
    )
    return response
