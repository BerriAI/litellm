"""Live e2e: a team's max_parallel_requests is one in-flight pool shared by every key on the team.

A team is created with max_parallel_requests=1 and two keys. Both keys first make one ordinary
call each, so a fresh key's brief 401 while the auth cache catches up never counts as a refusal.
Then both keys start a long streaming call at the same moment. Each call takes seconds while
admission takes milliseconds, so the two always overlap and exactly one holds the team's slot.
The other must get the team's 429, whichever key loses the race. Once both finish, the slot is
free again and a key on the team is served. The test asserts the recorded state (/team/info
echoes the configured limit) and the enforced behavior on live traffic.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from e2e_http import StreamingResponse, require_successful_call
from e2e_metadata import Domain, Mode, Provider, Subject, meta
from lifecycle import ResourceManager
from models import KeyGenerateBody, TeamNewBody
from quota_client import QuotaClient

pytestmark = pytest.mark.e2e

MODEL = CHEAP_ANTHROPIC_MODEL
HOLD_PROMPT = "Count from 1 to 400, one number per line, and write nothing else."
HOLD_MAX_TOKENS = 1200


def _team_key(client: QuotaClient, resources: ResourceManager, team_id: str) -> str:
    key = client.proxy.generate_key(KeyGenerateBody(models=[MODEL], team_id=team_id))
    resources.defer(lambda: client.proxy.delete_key(key))
    return key


def _served(client: QuotaClient, key: str) -> None:
    """One ordinary call that must succeed, retrying only the 401 a fresh key can briefly get."""
    deadline = time.monotonic() + client.proxy.poll_timeout
    while True:
        outcome = client.chat(key, MODEL, f"reply with one word {unique_marker()}")
        if outcome.ok or outcome.status_code != 401 or time.monotonic() >= deadline:
            require_successful_call(outcome)
            return
        time.sleep(client.proxy.poll_interval)


def _overlapping_streams(client: QuotaClient, keys: tuple[str, str]) -> tuple[StreamingResponse, ...]:
    start = Barrier(len(keys))

    def hold(key: str) -> StreamingResponse:
        start.wait()
        return client.stream_chat(key, MODEL, HOLD_PROMPT, max_tokens=HOLD_MAX_TOKENS)

    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        return tuple(pool.map(hold, keys))


class TestTeamParallelRequestLimit:
    @pytest.mark.covers("quota_management.ratelimit.team_parallel.shared_across_keys")
    @meta(
        Subject(
            domain=Domain.SPEND_BUDGETS,
            providers=(Provider.ANTHROPIC,),
            models=(CHEAP_ANTHROPIC_MODEL,),
            mode=Mode.STREAM,
        )
    )
    def test_team_parallel_limit_is_shared_across_keys(self, client: QuotaClient, resources: ResourceManager) -> None:
        team_id = client.proxy.create_team(
            TeamNewBody(team_alias=f"e2e-team-parallel-{unique_marker()}", models=[MODEL], max_parallel_requests=1)
        )
        resources.defer(lambda: client.proxy.delete_team(team_id))
        recorded = client.team_info(team_id).max_parallel_requests
        assert recorded == 1, f"/team/info reports max_parallel_requests {recorded}, configured 1"
        keys = (_team_key(client, resources, team_id), _team_key(client, resources, team_id))
        for key in keys:
            _served(client, key)

        outcomes = _overlapping_streams(client, keys)

        statuses = sorted(outcome.status_code for outcome in outcomes)
        assert statuses == [200, 429], (
            f"one key must hold the team's only slot and the other be refused, got {statuses}"
        )
        refused = next(outcome for outcome in outcomes if outcome.status_code == 429)
        assert f"Rate limit exceeded for team: {team_id}" in refused.body, (
            f"429 body must name the team scope, got: {refused.body[:300]}"
        )
        assert "Limit type: max_parallel_requests" in refused.body, (
            f"429 body must carry 'Limit type: max_parallel_requests', got: {refused.body[:300]}"
        )
        assert refused.headers.get("rate_limit_type") == "max_parallel_requests", (
            f"429 must carry rate_limit_type=max_parallel_requests, got {refused.headers.get('rate_limit_type')!r}"
        )
        _served(client, keys[1])
