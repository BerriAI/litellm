from .oidc import router as oidc_router
from .saml import router as saml_router
from .scim import router as scim_router

__all__ = ["oidc_router", "saml_router", "scim_router"]
