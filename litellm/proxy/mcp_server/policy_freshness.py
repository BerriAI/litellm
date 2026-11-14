"""Decides, once per MCP listing, whether the caller's cached policy still matches the database.

One query on the writer hashes every row the caller's MCP access depends on. While the hash equals the one
recorded after the last completed listing, the cache is trusted. Anything else reads policy from the database.
"""

import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from litellm._logging import verbose_logger
from litellm.models.mcp_policy import PolicySubject, ReachableObjects
from litellm.proxy.common_utils.strict_cache_refill import release_strict_cache_refill, require_strict_cache_refill
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    get_management_object_ttl,
    object_permission_cache_key,
    team_membership_auth_cache_key,
    team_membership_reservation_cache_key,
    user_object_permission_id_cache_key,
)
from litellm.repositories.mcp_policy_repository import MCPPolicyRepository

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import PrismaClient

_MARK_CAPACITY: Final = 4096
_MARK_TTL_FRACTION: Final = 0.5
_SETTLE_SECONDS: Final = 5.0


def policy_subject_for(
    auth: "UserAPIKeyAuth", *, keyless_user: bool, virtual_key: bool, jwt_teams_from_claim_list: bool
) -> PolicySubject | None:
    """The subject to validate for this caller, or None when its policy has inputs the hash cannot describe."""
    if auth.project_id or auth.agent_id or auth.end_user_id or auth.mcp_toolset_id:
        return None
    if keyless_user:
        return PolicySubject(user_id=auth.user_id, pinned_team_id=auth.team_id)
    if virtual_key:
        return PolicySubject(key_hash=auth.api_key)
    if auth.jwt_claims is not None and auth.user_id and not jwt_teams_from_claim_list:
        return PolicySubject(user_id=auth.user_id, pinned_team_id=auth.team_id, pinned_org_id=auth.org_id)
    return None


def cached_policy_keys(subject: PolicySubject, reachable: ReachableObjects) -> tuple[str, ...]:
    """The cache keys auth reads this subject's policy from; a reload refills the rows that still exist and cannot remove the rest."""
    user_id: Final = reachable.user_id or subject.user_id
    return (
        *((subject.key_hash,) if subject.key_hash is not None else ()),
        *((user_id, user_object_permission_id_cache_key(user_id)) if user_id is not None else ()),
        *(f"team_id:{team_id}" for team_id in reachable.team_ids),
        *(
            team_membership_auth_cache_key(team_id=team_id, user_id=user_id)
            for team_id in reachable.team_ids
            if user_id is not None
        ),
        *(
            team_membership_reservation_cache_key(team_id=team_id, user_id=user_id)
            for team_id in reachable.team_ids
            if user_id is not None
        ),
        *(f"org_id:{org_id}" for org_id in reachable.organization_ids),
        *(f"org_id:{org_id}:with_budget" for org_id in reachable.organization_ids),
        *(f"org_id:{org_id}:with_budget:last_known" for org_id in reachable.organization_ids),
        *((f"project_id:{reachable.project_id}",) if reachable.project_id is not None else ()),
        *(object_permission_cache_key(permission_id) for permission_id in reachable.permission_ids),
        *(f"access_group_id:{group_id}" for group_id in reachable.access_group_ids),
    )


class ValidatedPolicyMarks:
    """Per process, the digest each subject's policy was validated against. Trusted for ``max_age``, and only once settled:
    a refill in flight or from a lagging replica can land after the first eviction, so a second one follows."""

    def __init__(self, capacity: int = _MARK_CAPACITY, clock: Callable[[], float] = time.monotonic) -> None:
        self._capacity: Final = capacity
        self._clock: Final = clock
        self._marks: Final[OrderedDict[str, tuple[str, float, bool]]] = OrderedDict()  # mutable-ok: bounded LRU

    def matches(self, subject: PolicySubject, digest: str, max_age: float) -> bool:
        mark: Final = self._marks.get(subject.mark)
        if mark is None or mark[0] != digest or not mark[2] or self._clock() - mark[1] >= max_age:
            return False
        self._marks.move_to_end(subject.mark)
        return True

    def phase(self, subject: PolicySubject, digest: str) -> Literal["other", "settling", "due", "settled"]:
        """Where this digest stands: not the recorded one, still inside the settle wait, due its second eviction, or settled."""
        mark: Final = self._marks.get(subject.mark)
        if mark is None or mark[0] != digest:
            return "other"
        if mark[2]:
            return "settled"
        return "due" if self._clock() - mark[1] >= _SETTLE_SECONDS else "settling"

    def record(self, subject: PolicySubject, digest: str, settled: bool = True) -> None:
        self._marks[subject.mark] = (digest, self._clock(), settled)
        self._marks.move_to_end(subject.mark)
        while len(self._marks) > self._capacity:
            self._marks.popitem(last=False)


validated_policy_marks: Final = ValidatedPolicyMarks()


@dataclass(slots=True)
class RequestFreshness:
    """One listing: the digest to record once it completes. Live only inside ``validated_listing``,
    because a stateful MCP session's handler task inherits its first request's context."""

    staged: tuple[ValidatedPolicyMarks, PolicySubject, str, "PrismaClient", bool] | None = None
    strict_token: Token[bool] | None = None
    active: bool = True

    def trust_cache(self) -> None:
        if self.strict_token is None:
            self.strict_token = require_strict_cache_refill()

    async def commit(self) -> None:
        """Record the staged digest only if the policy still hashes to it, so a change during the reload is not marked valid."""
        if self.staged is None:
            return
        marks, subject, digest, prisma_client, settled_mark = self.staged
        self.staged = None
        try:
            confirmed: Final = await MCPPolicyRepository(prisma_client).get_fingerprint(subject)
        except Exception as e:  # noqa: BLE001  # an unconfirmed digest is simply not recorded
            verbose_logger.warning("MCP policy fingerprint could not be confirmed after the listing: %s", e)
            return
        if confirmed.digest == digest:
            marks.record(subject, digest, settled=settled_mark)

    def end(self) -> None:
        self.active = False
        if self.strict_token is not None:
            release_strict_cache_refill(self.strict_token)
            self.strict_token = None


_current_listing: Final[ContextVar[RequestFreshness | None]] = ContextVar("mcp_current_listing", default=None)


def current_listing() -> RequestFreshness | None:
    listing: Final = _current_listing.get()
    return listing if listing is not None and listing.active else None


@asynccontextmanager
async def validated_listing() -> AsyncGenerator[RequestFreshness]:
    """One listing's scope. The digest it validated is recorded only when the listing completes."""
    listing: Final = RequestFreshness()
    token: Final = _current_listing.set(listing)
    try:
        yield listing
        await listing.commit()
    finally:
        listing.end()
        _current_listing.reset(token)


async def cached_policy_is_current(
    subject: PolicySubject | None,
    *,
    prisma_client: "PrismaClient | None",
    cache: UserApiKeyCache,
    marks: ValidatedPolicyMarks,
) -> bool:
    """True only when the subject's policy hash equals the one recorded by the last completed listing.

    A changed hash evicts the caller's cached entries, and the digest is staged for the listing to record once it
    completes. Every failure answers False, so the policy is read from the database as it was before this check.
    """
    listing: Final = current_listing()
    if subject is None or listing is None or prisma_client is None:
        return False
    try:
        fingerprint: Final = await MCPPolicyRepository(prisma_client).get_fingerprint(subject)
    except Exception as e:  # noqa: BLE001  # an unreadable fingerprint must read policy from the database, not fail the request
        verbose_logger.warning("MCP policy fingerprint unavailable, reading policy from the database: %s", e)
        return False
    if fingerprint.uses_toolsets:
        return False
    if marks.matches(subject, fingerprint.digest, get_management_object_ttl(cache) * _MARK_TTL_FRACTION):
        listing.trust_cache()
        return True
    phase: Final = marks.phase(subject, fingerprint.digest)
    if phase == "settling":
        return False
    if phase != "settled":
        try:
            await cache.async_delete_cache_keys(cached_policy_keys(subject, fingerprint.reachable))
        except Exception as e:  # noqa: BLE001  # the forced reload still refills rows that exist; only deleted rows rely on this eviction
            verbose_logger.warning("MCP cached policy eviction failed: %s", e)
            return False
    listing.staged = (marks, subject, fingerprint.digest, prisma_client, phase != "other")
    return False
