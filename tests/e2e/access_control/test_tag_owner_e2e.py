"""Live e2e: a tag with an owning team is accepted only from that team's keys.

Per-team spend reports group by request tag, so a tag another team can send pollutes
the owning team's numbers. `/tag/new` and `/tag/update` take an optional `team_id`;
a request carrying an owned tag from a key of another team, or from a key with no
team (the master key included), is refused with a 403 that names the tag and the
owner, before any spend is recorded. Tags with no owner keep working for everyone.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest

from access_control_client import AccessControlClient, TAG_OWNER_DENIED_MARKER
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

MODEL: Final = "gemini-2.5-flash"


@dataclass(frozen=True, slots=True)
class OwnedTag:
    tag: str
    owner_team_id: str
    owner_key: str
    other_team_id: str
    other_key: str


@pytest.fixture(scope="module")
def module_resources(client: AccessControlClient) -> Iterator[ResourceManager]:
    manager = ResourceManager(client=client.proxy)
    manager.init()
    yield manager
    manager.teardown()


@pytest.fixture(scope="module")
def owned_tag(client: AccessControlClient, module_resources: ResourceManager) -> OwnedTag:
    marker = unique_marker()
    owner_team_id = client.create_team(f"e2e-tag-owner-{marker}", models=[])
    module_resources.defer(lambda: client.delete_team(owner_team_id))
    other_team_id = client.create_team(f"e2e-tag-other-{marker}", models=[])
    module_resources.defer(lambda: client.delete_team(other_team_id))
    tag = f"e2e-owned-tag-{marker}"
    client.create_owned_tag(tag, team_id=owner_team_id)
    module_resources.defer(lambda: client.delete_tag(tag))
    owner_key = client.team_key(owner_team_id)
    module_resources.defer(lambda: client.delete_key(owner_key))
    other_key = client.team_key(other_team_id)
    module_resources.defer(lambda: client.delete_key(other_key))
    return OwnedTag(tag, owner_team_id, owner_key, other_team_id, other_key)


class TestTagOwner:
    @pytest.mark.covers("other.auth.tag_owner.foreign_team_denied")
    def test_owning_team_key_sends_the_tag(self, client: AccessControlClient, owned_tag: OwnedTag) -> None:
        assert client.tag_owner(owned_tag.tag) == owned_tag.owner_team_id
        result = client.tagged_chat_status(owned_tag.owner_key, MODEL, f"say ok {unique_marker()}", [owned_tag.tag])
        require_successful_call(result)

    @pytest.mark.covers("other.auth.tag_owner.foreign_team_denied")
    def test_other_team_key_is_refused_with_403_naming_tag_and_owner(
        self, client: AccessControlClient, owned_tag: OwnedTag
    ) -> None:
        result = client.tagged_chat_status(owned_tag.other_key, MODEL, f"say ok {unique_marker()}", [owned_tag.tag])
        assert result.status_code == 403, result.body
        assert TAG_OWNER_DENIED_MARKER in result.body
        assert owned_tag.tag in result.body and owned_tag.owner_team_id in result.body

    @pytest.mark.covers("other.auth.tag_owner.foreign_team_denied")
    def test_key_without_a_team_is_refused(
        self, client: AccessControlClient, owned_tag: OwnedTag, scoped_key: str
    ) -> None:
        result = client.tagged_chat_status(scoped_key, MODEL, f"say ok {unique_marker()}", [owned_tag.tag])
        assert result.status_code == 403, result.body
        assert TAG_OWNER_DENIED_MARKER in result.body

    @pytest.mark.covers("other.auth.tag_owner.foreign_team_denied")
    def test_unowned_tag_stays_open_to_every_key(
        self, client: AccessControlClient, owned_tag: OwnedTag, scoped_key: str
    ) -> None:
        result = client.tagged_chat_status(
            scoped_key, MODEL, f"say ok {unique_marker()}", [f"e2e-open-{unique_marker()}"]
        )
        assert result.status_code == 200, result.body
        assert '"choices"' in result.body, result.body

    @pytest.mark.covers("other.auth.tag_owner.foreign_team_denied")
    def test_releasing_the_owner_reopens_the_tag_and_a_team_key_cannot_claim_it(
        self, client: AccessControlClient, owned_tag: OwnedTag, scoped_key: str
    ) -> None:
        claimed = client.update_tag_owner_status(owned_tag.other_key, owned_tag.tag, team_id=owned_tag.other_team_id)
        assert claimed.status_code in (401, 403), claimed.body
        assert client.tag_owner(owned_tag.tag) == owned_tag.owner_team_id

        released = client.update_tag_owner_status(None, owned_tag.tag, team_id=None)
        assert released.ok, released.body
        assert client.tag_owner(owned_tag.tag) is None
        result = client.tagged_chat_status(scoped_key, MODEL, f"say ok {unique_marker()}", [owned_tag.tag])
        require_successful_call(result)
