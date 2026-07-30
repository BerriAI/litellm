"""Live e2e: the master key authenticates and is treated as a proxy admin, and a
key that is not the master key is rejected before reaching the handler.

/user/list is admin-only, so it proves both halves of the master-key contract in
one route: the master key reads it (authenticated + authorized as admin), while a
freshly minted, never-provisioned token is denied 401 by the auth layer. The
invalid case uses a unique, master-key-shaped token so the check exercises the
credential comparison rather than a value that could collide with a real key.
"""

from __future__ import annotations

import pytest

from e2e_config import MASTER_KEY, unique_marker
from e2e_http import UnauthorizedError, unwrap
from other_client import OtherClient

pytestmark = pytest.mark.e2e


class TestMasterKeyAuth:
    @pytest.mark.covers("other.auth.master_key.valid_allows")
    def test_master_key_authenticates_and_grants_admin_route(self, client: OtherClient) -> None:
        listing = unwrap(client.list_users_as(MASTER_KEY))
        assert listing.total >= 0, (
            "master key reached the admin /user/list handler but the response did not "
            f"carry a user count: {listing}"
        )

    @pytest.mark.covers("other.auth.master_key.invalid_denied")
    def test_non_matching_master_key_is_denied(self, client: OtherClient) -> None:
        bogus = f"sk-{unique_marker()}"
        result = client.list_users_as(bogus)
        assert isinstance(result, UnauthorizedError), (
            f"a token that is not the master key must be rejected with 401, got {result}"
        )
