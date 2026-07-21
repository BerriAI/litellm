"""Client for the SCIM v2 provisioning e2e suite: the shared ProxyClient plus the
/scim/v2 Users and Groups operations an identity provider drives when it
provisions, updates, and deprovisions accounts against the gateway.

SCIM calls authenticate with a scoped provisioning token (a virtual key whose
allowed_routes is ["/scim/*"]), which is exactly what the IdP is configured with;
the token is passed per call so a test can also drive the denial path with a key
that lacks the permission. Verification read-backs (/user/info, /team/info) use the
master key through ManagementClient, so the suite injects both clients.
"""

from __future__ import annotations

from dataclasses import dataclass

from e2e_http import NoBody, Result, unwrap
from models import (
    SCIMGroupBody,
    SCIMGroupResponse,
    SCIMPatchOp,
    SCIMUserBody,
    SCIMUserResponse,
)
from proxy_client import ProxyClient


@dataclass(frozen=True, slots=True)
class SCIMProvisioningClient:
    proxy: ProxyClient

    def create_user(self, token: str, body: SCIMUserBody) -> SCIMUserResponse:
        return unwrap(
            self.proxy.transport.post(
                "/scim/v2/Users",
                headers=self.proxy.transport.bearer(token),
                json=body,
                response_type=SCIMUserResponse,
            )
        )

    def create_user_result(self, token: str, body: SCIMUserBody) -> Result[SCIMUserResponse]:
        """The raw outcome, for the access-control path where a key without /scim/*
        permission must be refused."""
        return self.proxy.transport.post(
            "/scim/v2/Users",
            headers=self.proxy.transport.bearer(token),
            json=body,
            response_type=SCIMUserResponse,
        )

    def get_user_result(self, token: str, user_id: str) -> Result[SCIMUserResponse]:
        return self.proxy.transport.get(
            f"/scim/v2/Users/{user_id}",
            headers=self.proxy.transport.bearer(token),
            params=NoBody(),
            response_type=SCIMUserResponse,
        )

    def patch_user(self, token: str, user_id: str, op: SCIMPatchOp) -> None:
        _ = unwrap(
            self.proxy.transport.patch(
                f"/scim/v2/Users/{user_id}",
                headers=self.proxy.transport.bearer(token),
                json=op,
                response_type=SCIMUserResponse,
            )
        )

    def delete_user(self, token: str, user_id: str) -> None:
        """DELETE answers 204 with an empty body (which the typed layer can't parse),
        so the call is fire-and-effect: tests prove deletion by the follow-up 404 and
        the internal user being gone. Best-effort, so teardown re-deletes harmlessly."""
        _ = self.proxy.transport.delete(
            f"/scim/v2/Users/{user_id}",
            headers=self.proxy.transport.bearer(token),
            json=NoBody(),
            response_type=NoBody,
        )

    def create_group(self, token: str, body: SCIMGroupBody) -> SCIMGroupResponse:
        return unwrap(
            self.proxy.transport.post(
                "/scim/v2/Groups",
                headers=self.proxy.transport.bearer(token),
                json=body,
                response_type=SCIMGroupResponse,
            )
        )

    def delete_group(self, token: str, group_id: str) -> None:
        _ = self.proxy.transport.delete(
            f"/scim/v2/Groups/{group_id}",
            headers=self.proxy.transport.bearer(token),
            json=NoBody(),
            response_type=NoBody,
        )


def build_scim_client(proxy: ProxyClient) -> SCIMProvisioningClient:
    return SCIMProvisioningClient(proxy=proxy)
