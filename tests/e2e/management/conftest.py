"""Management suite's `client` fixture.

Lifecycle/liveness gate/marker live in the parent conftest. ManagementClient
holds the shared ProxyClient so `resources` / `scoped_key` clean up keys, teams,
users, and orgs this suite creates.
"""

import pytest

from e2e_config import KEYCLOAK_ADMIN_PASSWORD, KEYCLOAK_ADMIN_USER, KEYCLOAK_REALM, KEYCLOAK_URL
from jwt_auth_client import JWTAuthClient, build_jwt_client
from keycloak import KeycloakAdmin, KeycloakEnv
from management_client import ManagementClient, build_client
from proxy_client import ProxyClient
from scim_provisioning_client import SCIMProvisioningClient, build_scim_client
from sso_management_client import SSOManagementClient, build_sso_client


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. mgmt.key.generate.persists",
    )


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> ManagementClient:
    return build_client(proxy)


@pytest.fixture(scope="session")
def sso_client(proxy: ProxyClient) -> SSOManagementClient:
    return build_sso_client(proxy)


@pytest.fixture(scope="session")
def scim_client(proxy: ProxyClient) -> SCIMProvisioningClient:
    return build_scim_client(proxy)


@pytest.fixture(scope="session")
def jwt_client(proxy: ProxyClient) -> JWTAuthClient:
    return build_jwt_client(proxy)


@pytest.fixture(scope="session")
def keycloak_env() -> KeycloakEnv:
    """Provision (idempotently) the realm/clients the JWT-auth suite needs and
    return a handle that mints real tokens. Hard-fails if Keycloak is unreachable -
    a live e2e never skips for missing infrastructure."""
    admin = KeycloakAdmin(
        base_url=KEYCLOAK_URL,
        realm=KEYCLOAK_REALM,
        admin_user=KEYCLOAK_ADMIN_USER,
        admin_password=KEYCLOAK_ADMIN_PASSWORD,
    )
    return admin.provision()
