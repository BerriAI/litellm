"""Live e2e: the SCIM v2 provisioning lifecycle an enterprise IdP (Okta, Entra,
etc.) drives against the gateway - provision a user, watch it become a real
internal user, deactivate it and see the account's access cut, then deprovision
it; plus group-to-team provisioning and the permission boundary on the SCIM token.

SCIM is enterprise-gated (the /scim/v2 router requires a premium license) and
authenticated with a scoped provisioning token: a virtual key whose allowed_routes
is ["/scim/*"], exactly what the IdP is given. Verification read-backs use the
master key via ManagementClient, so each test injects both clients.

Requires a licensed proxy (LITELLM_LICENSE) with STORE_MODEL_IN_DB=True and a DB.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from e2e_config import unique_marker
from e2e_http import Result, UnauthorizedError, UnknownApiError
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import (
    KeyGenerateBody,
    LiteLLMParamsBody,
    SCIMEmail,
    SCIMGroupBody,
    SCIMName,
    SCIMPatchOp,
    SCIMPatchOperation,
    SCIMUserBody,
    SCIMUserResponse,
)
from scim_provisioning_client import SCIMProvisioningClient

pytestmark = pytest.mark.e2e

# SCIM emails are validated as real addresses, so a routable domain is required
# (reserved TLDs such as .test / .example are rejected by the server).
_EMAIL_DOMAIN = "litellm-e2e.com"


def _poll[T](client: ManagementClient, attempt: Callable[[], T | None], failure: str) -> T:
    deadline = time.monotonic() + client.proxy.poll_timeout
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(client.proxy.poll_interval)
    pytest.fail(failure)


def _scim_token(client: ManagementClient, resources: ResourceManager) -> str:
    """A scoped SCIM provisioning token: a virtual key restricted to the /scim/*
    routes, the way an IdP is configured. Auto-deleted on teardown."""
    token = client.proxy.generate_key(KeyGenerateBody(allowed_routes=["/scim/*"]))
    resources.defer(lambda: client.proxy.delete_key(token))
    return token


def _provision_user(
    scim_client: SCIMProvisioningClient, token: str, resources: ResourceManager, marker: str
) -> tuple[SCIMUserResponse, str]:
    email = f"scim-{marker}@{_EMAIL_DOMAIN}"
    user = scim_client.create_user(
        token,
        SCIMUserBody(
            userName=f"scim-user-{marker}",
            emails=[SCIMEmail(value=email)],
            name=SCIMName(givenName="E2E", familyName="Scim"),
        ),
    )
    resources.defer(lambda: scim_client.delete_user(token, user.id))
    return user, email


class TestSCIMUserProvisioning:
    @pytest.mark.covers("mgmt.scim.user_provision.persists")
    def test_provision_creates_a_real_internal_user(
        self, client: ManagementClient, scim_client: SCIMProvisioningClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        token = _scim_token(client, resources)

        user, email = _provision_user(scim_client, token, resources, marker)
        assert user.id == f"scim-user-{marker}", (
            f"SCIM create returned id {user.id!r}; userName should map to the litellm user_id 'scim-user-{marker}'"
        )
        assert user.active is True, "a freshly provisioned SCIM user should be active"

        info = client.user_info(user.id).user_info
        assert info.user_email == email, (
            f"/user/info reports user_email {info.user_email!r} for the SCIM-provisioned user, expected {email!r}"
        )

    @pytest.mark.covers("mgmt.scim.user_deprovision.persists")
    def test_deprovision_removes_the_internal_user(
        self, client: ManagementClient, scim_client: SCIMProvisioningClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        token = _scim_token(client, resources)
        user, _ = _provision_user(scim_client, token, resources, marker)
        assert client.user_count(user.id) == 1, f"SCIM-provisioned user {user.id} absent from /user/list before delete"

        scim_client.delete_user(token, user.id)

        after = scim_client.get_user_result(token, user.id)
        match after:
            case UnknownApiError(status_code=code):
                assert code == 404, f"GET /scim/v2/Users/{user.id} after delete should be 404, got {code}"
            case _:
                pytest.fail(f"GET /scim/v2/Users/{user.id} after delete should be 404, got {after}")

        _ = _poll(
            client,
            lambda: True if client.user_count(user.id) == 0 else None,
            f"SCIM-deleted user {user.id} still present in /user/list after the deadline",
        )


class TestSCIMUserDeactivation:
    @pytest.mark.covers("mgmt.scim.user_deactivate.blocks_key")
    def test_deactivation_blocks_the_users_key_on_chat(
        self, client: ManagementClient, scim_client: SCIMProvisioningClient, resources: ResourceManager
    ) -> None:
        """The cross-feature promise: an IdP setting active=false via SCIM PATCH cuts
        the deactivated user's gateway access. Proven end to end - the user's key
        serves chat before the flip and is rejected at auth after it."""
        marker = unique_marker()
        token = _scim_token(client, resources)
        user, _ = _provision_user(scim_client, token, resources, marker)

        model_name = f"scim-mock-{marker}"
        model_id = client.proxy.create_model(
            model_name, LiteLLMParamsBody(model="openai/gpt-4o-mini", mock_response="scim ok")
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))

        key = client.proxy.generate_key(KeyGenerateBody(user_id=user.id, models=[model_name]))
        resources.defer(lambda: client.proxy.delete_key(key))

        _ = _poll(
            client,
            lambda: True if client.chat_status(key, model_name, f"hi {unique_marker()}").ok else None,
            "the provisioned user's key never served chat before deactivation",
        )

        scim_client.patch_user(
            token, user.id, SCIMPatchOp(Operations=[SCIMPatchOperation(op="replace", path="active", value=False)])
        )

        _ = _poll(
            client,
            lambda: True if client.chat_status(key, model_name, f"hi {unique_marker()}").status_code == 401 else None,
            "the SCIM-deactivated user's key was never rejected (401) on chat after the deadline",
        )


class TestSCIMGroupProvisioning:
    @pytest.mark.covers("mgmt.scim.group_provision.persists")
    def test_group_provision_creates_a_real_team(
        self, client: ManagementClient, scim_client: SCIMProvisioningClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        token = _scim_token(client, resources)
        display_name = f"scim-team-{marker}"

        group = scim_client.create_group(token, SCIMGroupBody(displayName=display_name))
        resources.defer(lambda: scim_client.delete_group(token, group.id))

        info = _poll(
            client,
            lambda: (lambda data: data if data.team_alias == display_name else None)(client.team_info(group.id)),
            f"/team/info never reported team_alias {display_name!r} for the SCIM-provisioned group {group.id}",
        )
        assert info.team_alias == display_name


class TestSCIMAccessControl:
    @pytest.mark.covers("mgmt.scim.user_provision.denied_without_permission")
    def test_token_without_scim_permission_is_denied(
        self, client: ManagementClient, scim_client: SCIMProvisioningClient, resources: ResourceManager
    ) -> None:
        """A key not scoped to /scim/* cannot provision. The IdP's provisioning token
        must carry the SCIM route permission; a general key is refused."""
        marker = unique_marker()
        key = client.proxy.generate_key(KeyGenerateBody(allowed_routes=["llm_api_routes"]))
        resources.defer(lambda: client.proxy.delete_key(key))

        result: Result[SCIMUserResponse] = scim_client.create_user_result(
            key,
            SCIMUserBody(
                userName=f"scim-denied-{marker}",
                emails=[SCIMEmail(value=f"scim-denied-{marker}@{_EMAIL_DOMAIN}")],
            ),
        )
        match result:
            case UnauthorizedError():
                return
            case UnknownApiError(status_code=403):
                return
            case _:
                pytest.fail(f"POST /scim/v2/Users with a non-SCIM key must be denied (401 or 403), got {result}")
