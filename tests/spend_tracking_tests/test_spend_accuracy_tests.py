import pytest
import asyncio
import aiohttp
import time

import litellm
from litellm._uuid import uuid

"""
Tests to run

Basic Tests:
1. Basic Spend Accuracy Test:
    - Make N requests, compute expected total spend locally from each response's usage
    - Poll until batch writer has flushed spend to the DB
    - Expect spend for Key, Team, User, Org (/info endpoints) to equal the computed total

2. Long term spend accuracy test (with 2 bursts of requests)
    - Burst 1: compute expected from responses, verify
    - Burst 2: compute expected from responses, verify total = burst1 + burst2

Additional Test Scenarios:

3. Concurrent Request Accuracy Test:
    - Make 20 concurrent requests
    - Check for race conditions in spend tracking

4. Error Case Test:
    - Make 10 successful requests
    - Make 5 failed requests
    - Verify spend is only counted for successful requests

5. Mixed Request Type Test:
    - Make different types of requests with varying costs
    - Verify accurate total spend calculation
"""

# Upstream model the proxy is configured with (spend_tracking_config.yaml).
# The proxy computes spend using this model's pricing; the local ground-truth
# calculation uses the same pricing table via litellm.cost_per_token.
UPSTREAM_MODEL = "gpt-3.5-turbo"

# Batch writer flush cadence in CI is ~2-7s (PROXY_BATCH_WRITE_AT=2 + up to 5s jitter).
# Poll every 2s for 60s — plenty of headroom for multiple ticks to land.
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 60

TOLERANCE = 1e-10


def _make_test_session() -> aiohttp.ClientSession:
    """
    Session tuned for CI reliability:
    - force_close: avoid aiohttp reusing a TCP connection that the proxy/kernel
      silently closed during the long idle window between setup POSTs and the
      later poll loop (observed failure mode: ConnectionTimeoutError on the
      first /key/info call after 20 chat completions).
    - explicit connect timeout: surface a blocked proxy event loop quickly
      instead of hanging on aiohttp's 5-minute default total timeout.
    """
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(force_close=True),
        timeout=aiohttp.ClientTimeout(total=30, connect=10),
    )


async def create_organization(session, organization_alias: str):
    """Helper function to create a new organization"""
    url = "http://0.0.0.0:4000/organization/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"organization_alias": organization_alias}
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def create_team(session, org_id: str):
    """Helper function to create a new team under an organization"""
    url = "http://0.0.0.0:4000/team/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"organization_id": org_id, "team_alias": f"test-team-{uuid.uuid4()}"}
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def create_user(session, org_id: str):
    """Helper function to create a new user"""
    url = "http://0.0.0.0:4000/user/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"user_name": f"test-user-{uuid.uuid4()}"}
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def generate_key(session, user_id: str, team_id: str):
    """Helper function to generate a key for a specific user and team"""
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"user_id": user_id, "team_id": team_id}
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()


async def chat_completion(session, key: str):
    """Make a chat completion request"""
    from openai import AsyncOpenAI
    from litellm._uuid import uuid

    client = AsyncOpenAI(api_key=key, base_url="http://0.0.0.0:4000/v1")

    response = await client.chat.completions.create(
        model="fake-openai-endpoint",
        messages=[{"role": "user", "content": f"Test message {uuid.uuid4()}"}],
    )
    return response


async def get_spend_info(session, entity_type: str, entity_id: str):
    """Helper function to get spend information for an entity"""
    url = f"http://0.0.0.0:4000/{entity_type}/info"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    if entity_type == "key":
        data = {"key": entity_id}
    else:
        data = {f"{entity_type}_id": entity_id}

    async with session.get(url, headers=headers, params=data) as response:
        return await response.json()


async def get_proxy_readiness(session):
    """Fetch authenticated readiness details. Used both as a fail-fast gate and as a diagnostic on poll timeout."""
    url = "http://0.0.0.0:4000/health/readiness/details"
    headers = {"Authorization": "Bearer sk-1234"}
    async with session.get(url, headers=headers) as response:
        return response.status, await response.json()


async def assert_proxy_healthy(session):
    """Fail fast if the proxy's DB or cache is not reachable — no point running the test."""
    status, body = await get_proxy_readiness(session)
    if status != 200 or body.get("db") != "connected":
        pytest.fail(
            f"Proxy /health/readiness/details unhealthy (status={status}). "
            f"Cannot run spend accuracy test. Response: {body}"
        )
    print(f"Proxy readiness OK: {body}")


def compute_expected_spend(responses) -> float:
    """
    Compute the expected total spend locally from each response's usage tokens,
    using the same pricing table the proxy uses. This is the independent ground
    truth we compare the proxy's reported spend against.
    """
    total = 0.0
    for r in responses:
        usage = r.usage
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=UPSTREAM_MODEL,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        total += prompt_cost + completion_cost
    return total


async def poll_key_spend_until(session, key: str, expected: float) -> float:
    """
    Poll key spend until it matches `expected` within TOLERANCE, or timeout.
    Returns the last observed spend either way; caller decides how to report.
    """
    start = time.time()
    last_spend = 0.0
    while time.time() - start < POLL_TIMEOUT_SECONDS:
        try:
            key_info = await get_spend_info(session, "key", key)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            print(
                f"Transient transport error during spend poll: "
                f"{type(exc).__name__}: {exc}. Retrying... "
                f"({time.time() - start:.1f}s elapsed)"
            )
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            continue
        last_spend = key_info["info"]["spend"]
        if abs(last_spend - expected) < TOLERANCE:
            print(
                f"Key spend reached expected {expected} after {time.time() - start:.1f}s"
            )
            return last_spend
        print(
            f"Key spend {last_spend}, expected {expected}, waiting... "
            f"({time.time() - start:.1f}s elapsed)"
        )
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    return last_spend


async def fail_with_diagnostics(session, stage: str, expected: float, observed: float):
    """Emit a failure with readiness state so CI output points at the real cause."""
    _, readiness = await get_proxy_readiness(session)
    pytest.fail(
        f"{stage}: key spend did not match expected after {POLL_TIMEOUT_SECONDS}s poll. "
        f"expected={expected}, observed={observed}, diff={expected - observed}. "
        f"Proxy readiness: {readiness}"
    )


@pytest.mark.asyncio
async def test_basic_spend_accuracy():
    """
    Test basic spend accuracy across different entities:
    1. Create org, team, user, and key
    2. Make N requests, keeping each response
    3. Compute expected spend locally from response usage (independent ground truth)
    4. Poll until proxy-reported spend matches expected
    5. Verify spend is consistent across key, team, user, and org entities
    """
    NUM_LLM_REQUESTS = 20

    async with _make_test_session() as session:
        await assert_proxy_healthy(session)

        org_response = await create_organization(
            session=session, organization_alias=f"test-org-{uuid.uuid4()}"
        )
        print("org_response: ", org_response)
        org_id = org_response["organization_id"]

        team_response = await create_team(session, org_id)
        print("team_response: ", team_response)
        team_id = team_response["team_id"]

        user_response = await create_user(session, org_id)
        print("user_response: ", user_response)
        user_id = user_response["user_id"]

        key_response = await generate_key(session, user_id, team_id)
        print("key_response: ", key_response)
        key = key_response["key"]

        responses = []
        for i in range(NUM_LLM_REQUESTS):
            response = await chat_completion(session, key)
            responses.append(response)
            print(f"Request {i + 1}/{NUM_LLM_REQUESTS} completed")

        expected_spend = compute_expected_spend(responses)
        assert expected_spend > 0, (
            f"Locally computed expected spend is {expected_spend}. Either cost calc "
            f"is broken or upstream returned zero tokens. "
            f"Usage: {[r.usage.model_dump() for r in responses]}"
        )
        print(f"Expected total spend (local ground truth): {expected_spend}")

        final_spend = await poll_key_spend_until(session, key, expected_spend)
        if abs(final_spend - expected_spend) >= TOLERANCE:
            await fail_with_diagnostics(
                session,
                stage="test_basic_spend_accuracy",
                expected=expected_spend,
                observed=final_spend,
            )

        # Allow a final scheduler tick for team/user/org aggregations to settle
        await asyncio.sleep(5)

        key_info = await get_spend_info(session, "key", key)
        print("key_info: ", key_info)
        team_info = await get_spend_info(session, "team", team_id)
        print("team_info: ", team_info)
        user_info = await get_spend_info(session, "user", user_id)
        print("user_info: ", user_info)
        org_info = await get_spend_info(session, "organization", org_id)
        print("org_info: ", org_info)

        assert (
            abs(key_info["info"]["spend"] - expected_spend) < TOLERANCE
        ), f"Key spend {key_info['info']['spend']} does not match expected {expected_spend}"

        assert (
            abs(user_info["user_info"]["spend"] - expected_spend) < TOLERANCE
        ), f"User spend {user_info['user_info']['spend']} does not match expected {expected_spend}"

        assert (
            abs(team_info["team_info"]["spend"] - expected_spend) < TOLERANCE
        ), f"Team spend {team_info['team_info']['spend']} does not match expected {expected_spend}"

        assert (
            abs(org_info["spend"] - expected_spend) < TOLERANCE
        ), f"Organization spend {org_info['spend']} does not match expected {expected_spend}"


@pytest.mark.asyncio
async def test_long_term_spend_accuracy_with_bursts():
    """
    Test long-term spend accuracy with multiple bursts of requests:
    1. Create org, team, user, and key
    2. Burst 1: make requests, compute expected locally, verify proxy matches
    3. Burst 2: make more requests, verify proxy total == burst1 + burst2
    4. Verify total spend is consistent across all entities
    """
    BURST_1_REQUESTS = 22
    BURST_2_REQUESTS = 12

    async with _make_test_session() as session:
        await assert_proxy_healthy(session)

        org_response = await create_organization(
            session=session, organization_alias=f"test-org-{uuid.uuid4()}"
        )
        print("org_response: ", org_response)
        org_id = org_response["organization_id"]

        team_response = await create_team(session, org_id)
        print("team_response: ", team_response)
        team_id = team_response["team_id"]

        user_response = await create_user(session, org_id)
        print("user_response: ", user_response)
        user_id = user_response["user_id"]

        key_response = await generate_key(session, user_id, team_id)
        print("key_response: ", key_response)
        key = key_response["key"]

        print(f"Starting first burst of {BURST_1_REQUESTS} requests...")
        burst_1_responses = []
        for i in range(BURST_1_REQUESTS):
            response = await chat_completion(session, key)
            burst_1_responses.append(response)
            print(f"Burst 1 - Request {i + 1}/{BURST_1_REQUESTS} completed")

        burst_1_expected = compute_expected_spend(burst_1_responses)
        assert burst_1_expected > 0, (
            f"Burst 1 expected spend is {burst_1_expected}. "
            f"Usage: {[r.usage.model_dump() for r in burst_1_responses]}"
        )
        print(f"Burst 1 expected spend: {burst_1_expected}")

        final_burst_1 = await poll_key_spend_until(session, key, burst_1_expected)
        if abs(final_burst_1 - burst_1_expected) >= TOLERANCE:
            await fail_with_diagnostics(
                session,
                stage="test_long_term_spend_accuracy burst 1",
                expected=burst_1_expected,
                observed=final_burst_1,
            )

        print(f"Starting second burst of {BURST_2_REQUESTS} requests...")
        burst_2_responses = []
        for i in range(BURST_2_REQUESTS):
            response = await chat_completion(session, key)
            burst_2_responses.append(response)
            print(f"Burst 2 - Request {i + 1}/{BURST_2_REQUESTS} completed")

        total_expected = burst_1_expected + compute_expected_spend(burst_2_responses)
        print(f"Total expected spend (burst 1 + burst 2): {total_expected}")

        final_total = await poll_key_spend_until(session, key, total_expected)
        if abs(final_total - total_expected) >= TOLERANCE:
            await fail_with_diagnostics(
                session,
                stage="test_long_term_spend_accuracy total",
                expected=total_expected,
                observed=final_total,
            )

        await asyncio.sleep(5)

        key_info = await get_spend_info(session, "key", key)
        team_info = await get_spend_info(session, "team", team_id)
        user_info = await get_spend_info(session, "user", user_id)
        org_info = await get_spend_info(session, "organization", org_id)

        print(f"Final key spend: {key_info['info']['spend']}")
        print(f"Final team spend: {team_info['team_info']['spend']}")
        print(f"Final user spend: {user_info['user_info']['spend']}")
        print(f"Final org spend: {org_info['spend']}")

        assert (
            abs(key_info["info"]["spend"] - total_expected) < TOLERANCE
        ), f"Key spend {key_info['info']['spend']} does not match expected {total_expected}"

        assert (
            abs(user_info["user_info"]["spend"] - total_expected) < TOLERANCE
        ), f"User spend {user_info['user_info']['spend']} does not match expected {total_expected}"

        assert (
            abs(team_info["team_info"]["spend"] - total_expected) < TOLERANCE
        ), f"Team spend {team_info['team_info']['spend']} does not match expected {total_expected}"

        assert (
            abs(org_info["spend"] - total_expected) < TOLERANCE
        ), f"Organization spend {org_info['spend']} does not match expected {total_expected}"
