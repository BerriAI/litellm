from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request

from .models import Credential, SecuritySchemeType


class SamlAuthenticator:
    """Thin SAML SP seam. Full pysaml2 wiring (system libxmlsec1, pinned
    xmlsec/lxml, multi-IdP metadata) is deferred; the ACS maps a SAML assertion's
    NameID and attribute statements into the same scim2_models.User upsert as
    OIDC and SCIM."""

    scheme = SecuritySchemeType.HTTP

    async def authenticate(self, request: Request) -> Optional[Credential]:
        raise NotImplementedError("SAML SP deferred; see 03-design.md cut list")


def build_saml_router() -> APIRouter:
    router = APIRouter(prefix="/auth/saml", tags=["saml"])

    @router.get("/metadata")
    async def metadata() -> None:
        raise NotImplementedError("requires pysaml2 + system libxmlsec1")

    @router.post("/acs")
    async def assertion_consumer_service(request: Request) -> None:
        raise NotImplementedError(
            "parse assertion -> scim2_models.User -> store.upsert_user"
        )

    return router
