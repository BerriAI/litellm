"""Live e2e: retries absorb a failing deployment inside a model group.

Every test registers a model group holding two REAL deployments: one that always
fails in a specific way, and a healthy `gpt-5.5`. Each test first drives the group
with retries switched off until the failing deployment answers, which pins the
failure class the rest of the test is about, then turns retries on and drives it
until the proxy reports a retry, requiring that every response in the meantime was
a real completion. That is the product promise this covers: a caller of a group
that contains a broken deployment never sees the breakage.

Each failure is produced by a real deployment, never a mock:

- 5xx        an api_base nothing listens on, which the gateway surfaces as a 500
- timeout    a 1ms deadline the real backend always exceeds (408)
- 429        a deployment pointed back at this proxy with a virtual key whose
             rpm limit is 0, so the upstream really does rate-limit it. The key is
             polled until it answers 429 before any traffic runs, because a key
             that has not propagated yet answers 401, and a 401 is retryable too
- auth       a deliberately wrong api_key, which the real OpenAI API rejects (401)
- context    a small-context deployment (gpt-3.5-turbo, 16385 tokens) plus a prompt
             that exceeds it, which the real OpenAI API rejects

The 5xx case rides the flat `num_retries`; the others ride the router's
`model_group_retry_policy`, the per-exception-class retry budget. For the
context-window case that policy is what makes the failure retryable at all, since
the provider's 400 is otherwise terminal.

Cooldowns are disabled on the failing deployment (`cooldown_time: 0`) so the
behavior under test is the retry loop alone. The output budget is roomy for the
same reason the assertions do not look at message content: gpt-5.5 spends output
tokens on reasoning, and a tight cap makes the provider itself reject the call.

The deployment is re-picked at random on every attempt, so both bounds here are
sized to make a correct build failing by chance negligible rather than merely
unlikely. With two deployments each pick is even, so across `MAX_REQUESTS` = 24
requests the odds of never once landing on the failing deployment are 2**-24,
about 6e-8, and with `RETRY_BUDGET` = 24 retries the odds of a single request
exhausting them all on the failing deployment are 2**-25. Biasing selection toward
the failing deployment would trade one of those risks for the other, since the
same bias that makes the first attempt fail also makes the rescue less likely,
so the bounds are raised instead. Both bounds are ceilings: a typical run needs
two requests and two attempts.
"""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from complexity_router_client import ComplexityRouterClient
from e2e_config import CHEAP_OPENAI_MODEL, PROXY_BASE_URL, unique_marker
from e2e_http import NoBody, StreamingResponse, Success, unwrap
from lifecycle import ResourceManager
from models import (
    ChatBody,
    ChatMessage,
    ChatResponse,
    KeyGenerateBody,
    LiteLLMParamsBody,
    ModelNewResponse,
    ModelsListResponse,
)
from proxy_client import ProxyClient
from reliability_support import REAL_KEY, REAL_MODEL

pytestmark = pytest.mark.e2e

UNREACHABLE_BASE = "http://127.0.0.1:9/v1"
SMALL_CONTEXT_MODEL = "openai/gpt-3.5-turbo"
OVERSIZED_PROMPT = "word " * 20000
RETRY_BUDGET = 24
MAX_REQUESTS = 24
MAX_TOKENS = 256
KEY_POLL_INTERVAL = 1.0


class RetryDeploymentParams(LiteLLMParamsBody):
    """Deployment params plus the router's per-deployment `cooldown_time`.

    A zero cooldown keeps the router from taking the failing deployment out of
    rotation mid-test, so what the assertions see is the retry loop and nothing
    else."""

    cooldown_time: float = 0.0


class RetryPolicyBody(BaseModel):
    """The router's per-exception retry counts, one field per exception class the
    tests exercise. Aliased to the router's wire names."""

    authentication: int | None = Field(default=None, serialization_alias="AuthenticationErrorRetries")
    timeout: int | None = Field(default=None, serialization_alias="TimeoutErrorRetries")
    rate_limit: int | None = Field(default=None, serialization_alias="RateLimitErrorRetries")
    bad_request: int | None = Field(default=None, serialization_alias="BadRequestErrorRetries")


class RetryRouterSettings(BaseModel):
    """The `router_settings_override` slice these tests drive: a flat retry count,
    or a per-model-group retry policy keyed by exception class."""

    num_retries: int | None = None
    model_group_retry_policy: dict[str, RetryPolicyBody] | None = None


class RetryChatBody(ChatBody):
    """A /chat/completions body carrying the retry-only router override."""

    router_settings_override: RetryRouterSettings


class RetryModelNewBody(BaseModel):
    """POST /model/new body whose litellm_params carry `cooldown_time`."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    litellm_params: RetryDeploymentParams


NO_RETRIES = RetryRouterSettings(num_retries=0)


def _await_servable(proxy: ProxyClient, model_name: str) -> None:
    deadline = time.monotonic() + proxy.poll_timeout
    while time.monotonic() < deadline:
        listed = proxy.transport.get(
            "/v1/models",
            headers=proxy.transport.master,
            params=NoBody(),
            response_type=ModelsListResponse,
        )
        if isinstance(listed, Success) and any(entry.id == model_name for entry in listed.data.data):
            return
        time.sleep(proxy.poll_interval)
    raise AssertionError(f"model {model_name!r} never became servable on the data plane")


def _register(proxy: ProxyClient, model_name: str, params: RetryDeploymentParams) -> str:
    model_id = unwrap(
        proxy.transport.post(
            "/model/new",
            headers=proxy.transport.master,
            json=RetryModelNewBody(model_name=model_name, litellm_params=params),
            response_type=ModelNewResponse,
        )
    ).model_id
    _await_servable(proxy, model_name)
    return model_id


def _group_with_failing_deployment(
    proxy: ProxyClient, resources: ResourceManager, label: str, failing: RetryDeploymentParams
) -> tuple[str, str]:
    """Register a model group holding `failing` alongside a healthy gpt-5.5, and
    return the group name and the healthy deployment's model_id."""
    group = f"reliability-retry-{label}-{unique_marker()}"
    failing_id = _register(proxy, group, failing)
    resources.defer(lambda: proxy.delete_model(failing_id))
    healthy_id = _register(proxy, group, RetryDeploymentParams(model=REAL_MODEL, api_key=REAL_KEY))
    resources.defer(lambda: proxy.delete_model(healthy_id))
    return group, healthy_id


def _chat(
    proxy: ProxyClient, key: str, group: str, settings: RetryRouterSettings, content: str
) -> StreamingResponse:
    return proxy.transport.send(
        "/chat/completions",
        headers=proxy.transport.bearer(key),
        json=RetryChatBody(
            model=group,
            messages=[ChatMessage(role="user", content=content)],
            max_tokens=MAX_TOKENS,
            router_settings_override=settings,
        ),
    )


def _assert_real_completion(resp: StreamingResponse) -> None:
    """The body is a completion the provider actually produced: a chat-completion
    id, a choice, and usage the gateway billed. Content is not asserted because a
    reasoning model can spend a small max_tokens budget without emitting text."""
    try:
        parsed = ChatResponse.model_validate_json(resp.body)
    except ValidationError as exc:
        raise AssertionError(f"expected a chat completion body, got: {resp.body[:300]}") from exc
    assert parsed.id and parsed.choices, f"expected a completion with choices, got: {resp.body[:300]}"
    assert parsed.usage is not None and (parsed.usage.prompt_tokens or 0) > 0, (
        f"expected billed prompt tokens from a real provider call, got: {resp.body[:300]}"
    )


def _await_rate_limited_key(proxy: ProxyClient, key: str) -> None:
    """Block until `key` is live on the data plane and really answers 429.

    A key created a moment ago can answer 401 until it propagates, and a 401 is
    retryable as well, so a test that started driving traffic straight away could
    watch a retry absorb an auth failure and call it a rate limit."""
    deadline = time.monotonic() + proxy.poll_timeout
    last = ""
    while time.monotonic() < deadline:
        resp = proxy.transport.send(
            "/chat/completions",
            headers=proxy.transport.bearer(key),
            json=ChatBody(
                model=CHEAP_OPENAI_MODEL,
                messages=[ChatMessage(role="user", content=f"say hi {unique_marker()}")],
                max_tokens=MAX_TOKENS,
            ),
        )
        if resp.status_code == 429 and "rate limit" in resp.body.lower():
            return
        last = f"{resp.status_code}: {resp.body[:300]}"
        time.sleep(KEY_POLL_INTERVAL)
    raise AssertionError(f"the rpm-limited key never answered a rate limit within {proxy.poll_timeout}s; last was {last}")


def _observe_raw_failure(
    proxy: ProxyClient, key: str, group: str, expected_status: int, expected_error: str, prompt: str
) -> None:
    """Drive the group with retries off until the failing deployment answers, and
    pin what it answered.

    This is what ties a test to its registry cell: without it an observed retry
    only proves that something in the group failed, and any retryable failure
    would satisfy the assertion equally well."""
    for _ in range(MAX_REQUESTS):
        resp = _chat(proxy, key, group, NO_RETRIES, f"{prompt} {unique_marker()}")
        if resp.status_code == 200:
            _assert_real_completion(resp)
            continue
        assert resp.status_code == expected_status, (
            f"the failing deployment in {group!r} should answer {expected_status} with retries off, "
            f"got {resp.status_code}: {resp.body[:300]}"
        )
        assert expected_error in resp.body, (
            f"expected {expected_error!r} in the failure {group!r} absorbs, got: {resp.body[:300]}"
        )
        return
    raise AssertionError(
        f"{MAX_REQUESTS} requests to {group!r} with retries off all succeeded; the failing deployment "
        f"was never selected, so the failure class under test was never established"
    )


def _drive_until_rescued_by_retry(
    proxy: ProxyClient,
    key: str,
    group: str,
    settings: RetryRouterSettings,
    healthy_id: str,
    prompt: str = "say hi",
) -> None:
    """Send real traffic at `group` until the gateway reports it retried, failing
    if any response along the way was not a real completion.

    The router picks a deployment at random per attempt, so which request first
    lands on the failing deployment is not fixed; what is fixed is that no caller
    ever sees the failure, and that the request the retry rescued was served by
    the healthy deployment."""
    for _ in range(MAX_REQUESTS):
        resp = _chat(proxy, key, group, settings, f"{prompt} {unique_marker()}")
        assert resp.status_code == 200, (
            f"a group holding a failing deployment must still answer 200, got {resp.status_code}: {resp.body[:300]}"
        )
        _assert_real_completion(resp)
        attempted = resp.headers.get("x-litellm-attempted-retries")
        assert attempted is not None, "response is missing the x-litellm-attempted-retries header"
        if int(attempted) >= 1:
            assert resp.headers.get("x-litellm-model-id") == healthy_id, (
                f"the retry should have landed on the healthy deployment {healthy_id!r}, "
                f"got {resp.headers.get('x-litellm-model-id')!r}"
            )
            return
    raise AssertionError(
        f"{MAX_REQUESTS} requests to {group!r} never reported a retry; the failing deployment "
        f"was never selected, or retries are not running"
    )


def _assert_retry_absorbs(
    proxy: ProxyClient,
    key: str,
    group: str,
    healthy_id: str,
    settings: RetryRouterSettings,
    expected_status: int,
    expected_error: str,
    prompt: str = "say hi",
) -> None:
    """Establish which failure the group produces, then show that retries hide it."""
    _observe_raw_failure(proxy, key, group, expected_status, expected_error, prompt)
    _drive_until_rescued_by_retry(proxy, key, group, settings, healthy_id, prompt)


class TestReliabilityRetries:
    @pytest.mark.covers("reliability.retry.5xx.succeeds_within_retries")
    def test_5xx_succeeds_within_retries(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group, healthy_id = _group_with_failing_deployment(
            client.proxy,
            resources,
            "5xx",
            RetryDeploymentParams(model=REAL_MODEL, api_key=REAL_KEY, api_base=UNREACHABLE_BASE),
        )
        _assert_retry_absorbs(
            client.proxy,
            scoped_key,
            group,
            healthy_id,
            RetryRouterSettings(num_retries=RETRY_BUDGET),
            expected_status=500,
            expected_error="InternalServerError",
        )

    @pytest.mark.covers("reliability.retry.timeout.succeeds_within_retries")
    def test_timeout_succeeds_within_retries(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group, healthy_id = _group_with_failing_deployment(
            client.proxy,
            resources,
            "timeout",
            RetryDeploymentParams(model=REAL_MODEL, api_key=REAL_KEY, timeout=0.001),
        )
        _assert_retry_absorbs(
            client.proxy,
            scoped_key,
            group,
            healthy_id,
            RetryRouterSettings(model_group_retry_policy={group: RetryPolicyBody(timeout=RETRY_BUDGET)}),
            expected_status=408,
            expected_error="Timeout",
        )

    @pytest.mark.covers("reliability.retry.429.succeeds_within_retries")
    def test_429_succeeds_within_retries(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        throttled_key = client.proxy.generate_key(
            KeyGenerateBody(models=[CHEAP_OPENAI_MODEL], rpm_limit=0, user_id="e2e-reliability-retry")
        )
        resources.defer(lambda: client.proxy.delete_key(throttled_key))
        _await_rate_limited_key(client.proxy, throttled_key)
        group, healthy_id = _group_with_failing_deployment(
            client.proxy,
            resources,
            "429",
            RetryDeploymentParams(model=REAL_MODEL, api_key=throttled_key, api_base=f"{PROXY_BASE_URL}/v1"),
        )
        _assert_retry_absorbs(
            client.proxy,
            scoped_key,
            group,
            healthy_id,
            RetryRouterSettings(model_group_retry_policy={group: RetryPolicyBody(rate_limit=RETRY_BUDGET)}),
            expected_status=429,
            expected_error="RateLimitError",
        )

    @pytest.mark.covers("reliability.retry.auth.succeeds_within_retries")
    def test_auth_succeeds_within_retries(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group, healthy_id = _group_with_failing_deployment(
            client.proxy,
            resources,
            "auth",
            RetryDeploymentParams(model=REAL_MODEL, api_key=f"sk-invalid-{unique_marker()}"),
        )
        _assert_retry_absorbs(
            client.proxy,
            scoped_key,
            group,
            healthy_id,
            RetryRouterSettings(model_group_retry_policy={group: RetryPolicyBody(authentication=RETRY_BUDGET)}),
            expected_status=401,
            expected_error="AuthenticationError",
        )

    @pytest.mark.covers("reliability.retry.context_window.succeeds_within_retries")
    def test_context_window_succeeds_within_retries(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group, healthy_id = _group_with_failing_deployment(
            client.proxy,
            resources,
            "ctx",
            RetryDeploymentParams(model=SMALL_CONTEXT_MODEL, api_key=REAL_KEY),
        )
        _assert_retry_absorbs(
            client.proxy,
            scoped_key,
            group,
            healthy_id,
            RetryRouterSettings(model_group_retry_policy={group: RetryPolicyBody(bad_request=RETRY_BUDGET)}),
            expected_status=400,
            expected_error="ContextWindowExceededError",
            prompt=OVERSIZED_PROMPT,
        )
