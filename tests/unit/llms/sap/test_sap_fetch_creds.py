import json
import queue
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Final, Literal

import httpx
import pytest

import litellm
import litellm.llms.sap.credentials as sap_credentials
from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig
from litellm.llms.sap.embed.transformation import GenAIHubEmbeddingConfig
from tests.unit.llms.sap.conftest import MockAICore

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

mock_sap_service_key_dict = {
    "serviceurls": {"AI_API_URL": "https://testurl.hana.ondemand.com/"},
    "clientid": "mockclientid",
    "clientsecret": "mockclientsecret",
    "url": "https://test.sap.hana.ondemand.com/",
}

mock_wrapped_sap_service_key_dict = {
    "credentials": {
        "serviceurls": {"AI_API_URL": "https://testurl.hana.ondemand.com/"},
        "clientid": "mockclientid",
        "clientsecret": "mockclientsecret",
        "url": "https://test.sap.hana.ondemand.com/",
    }
}

expected_creds = {
    "client_id": "mockclientid",
    "client_secret": "mockclientsecret",
    "auth_url": "https://test.sap.hana.ondemand.com/oauth/token",
    "base_url": "https://testurl.hana.ondemand.com/v2",
    "resource_group": "default",
}

mock_sap_vcap_service_key_dict = {
    "aicore": [
        {
            "label": "aicore",
            "name": "aicore-instance",
            "instance_guid": "53ad5b47-a49a-4fec-9f0b-cd921c00b828",
            "credentials": {
                "serviceurls": {"AI_API_URL": "vcap-api-url"},
                "url": "vcap-auth-url",
                "clientid": "vcap-clientid",
                "clientsecret": "vcap-clientsecret",
            },
        }
    ]
}


def _prep_env(monkeypatch):
    for var in (
        "AICORE_CLIENT_ID",
        "AICORE_CLIENT_SECRET",
        "AICORE_AUTH_URL",
        "AICORE_RESOURCE_GROUP",
        "AICORE_BASE_URL",
        "AICORE_CERT_URL",
        "AICORE_SERVICE_KEY",
        "VCAP_SERVICES",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AICORE_HOME", "notexist")
    monkeypatch.setattr("litellm.sap_service_key", None)


def test_sap_fetch_creds_from_env_service_key(monkeypatch):
    _prep_env(monkeypatch)
    monkeypatch.setenv("AICORE_SERVICE_KEY", json.dumps(mock_sap_service_key_dict))
    creds = sap_credentials.fetch_credentials()
    assert creds == expected_creds


def test_sap_fetch_creds_from_env_wrapped_service_key(monkeypatch):
    _prep_env(monkeypatch)
    monkeypatch.setenv(
        "AICORE_SERVICE_KEY", json.dumps(mock_wrapped_sap_service_key_dict)
    )
    creds = sap_credentials.fetch_credentials()
    assert creds == expected_creds


def test_sap_fetch_creds_from_arg_service_key(monkeypatch):
    _prep_env(monkeypatch)
    creds = sap_credentials.fetch_credentials(
        service_key=json.dumps(mock_sap_service_key_dict)
    )
    assert creds == expected_creds


def test_fetch_creds_from_env_vcap_service(monkeypatch):
    _prep_env(monkeypatch)
    monkeypatch.setenv("VCAP_SERVICES", json.dumps(mock_sap_vcap_service_key_dict))
    creds = sap_credentials.fetch_credentials()
    assert creds["client_id"] == "vcap-clientid"
    assert creds["client_secret"] == "vcap-clientsecret"
    assert creds["auth_url"] == "vcap-auth-url/oauth/token"
    assert creds["base_url"] == "vcap-api-url/v2"
    assert creds["resource_group"] == "default"


def test_fetch_creds_from_env(monkeypatch):
    _prep_env(monkeypatch)
    monkeypatch.setenv("AICORE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("AICORE_CLIENT_SECRET", "env-client-secret")
    monkeypatch.setenv("AICORE_AUTH_URL", "env-auth-url")
    monkeypatch.setenv("AICORE_BASE_URL", "env-base-url")
    monkeypatch.setenv("AICORE_RESOURCE_GROUP", "env-resource-group")

    creds = sap_credentials.fetch_credentials()

    assert creds["client_id"] == "env-client-id"
    assert creds["client_secret"] == "env-client-secret"
    assert creds["auth_url"] == "env-auth-url/oauth/token"
    assert creds["base_url"] == "env-base-url/v2"
    assert creds["resource_group"] == "env-resource-group"


def test_creds_priority_order(monkeypatch):
    _prep_env(monkeypatch)
    monkeypatch.setenv("AICORE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("AICORE_CLIENT_SECRET", "env-client-secret")
    monkeypatch.setenv("AICORE_AUTH_URL", "env-auth-url")
    monkeypatch.setenv("AICORE_BASE_URL", "env-base-url")
    monkeypatch.setenv("AICORE_RESOURCE_GROUP", "env-resource-group")
    creds = sap_credentials.fetch_credentials(
        service_key=json.dumps(mock_sap_service_key_dict)
    )
    assert creds["client_id"] == "mockclientid"
    assert creds["resource_group"] == "env-resource-group"


def test_no_credentials_configured(monkeypatch):
    _prep_env(monkeypatch)
    with pytest.raises(ValueError, match="No credentials found in any source"):
        sap_credentials.fetch_credentials()


def test_partial_credentials_missing_auth_url(monkeypatch):
    _prep_env(monkeypatch)

    # Set only client_id and base_url, missing auth_url
    monkeypatch.setenv("AICORE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AICORE_BASE_URL", "test-base-url")

    # fetch_credentials should succeed (it returns whatever it finds)
    creds = sap_credentials.fetch_credentials()
    creds.pop("resource_group")

    with pytest.raises(ValueError, match="SAP AI Core credentials not found"):
        sap_credentials.validate_credentials(**creds)


def test_credentials_without_authentication_mode(monkeypatch):
    _prep_env(monkeypatch)

    # Set all required fields but no authentication mode (no client_secret, no certs)
    monkeypatch.setenv("AICORE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AICORE_AUTH_URL", "test-auth-url")
    monkeypatch.setenv("AICORE_BASE_URL", "test-base-url")

    creds = sap_credentials.fetch_credentials()
    creds.pop("resource_group")

    # validate_credentials should raise because no authentication mode is provided
    with pytest.raises(ValueError, match="SAP AI Core credentials are incomplete"):
        sap_credentials.validate_credentials(**creds)


@pytest.mark.parametrize("vcap_services", [["aicore"], "aicore"])
def test_vcap_services_that_is_not_an_object_is_rejected(monkeypatch, vcap_services):
    _prep_env(monkeypatch)
    monkeypatch.setenv("VCAP_SERVICES", json.dumps(vcap_services))

    with pytest.raises(
        AttributeError,
        match=f"^'{type(vcap_services).__name__}' object has no attribute 'values'$",
    ):
        sap_credentials.fetch_credentials()


@pytest.mark.parametrize("expires_in", [43200, 3000], ids=["12-hour-token", "50-minute-token"])
def test_token_is_shared_across_token_creators_while_fresh(aicore: MockAICore, expires_in: int) -> None:
    aicore.token.respond(json={"access_token": "mock-token", "expires_in": expires_in})

    def clock() -> datetime:
        return _T0

    tokens: Final = tuple(sap_credentials.get_token_creator(clock=clock)[0]() for _ in range(3))

    assert tokens == ("Bearer mock-token",) * 3
    assert aicore.token.call_count == 1


def test_reused_token_is_replaced_once_time_moves_into_refresh_buffer(aicore: MockAICore) -> None:
    aicore.token.side_effect = [
        httpx.Response(200, json={"access_token": "first", "expires_in": 43200}),
        httpx.Response(200, json={"access_token": "second", "expires_in": 43200}),
    ]
    call_times: Final = iter((_T0, _T0 + timedelta(hours=10), _T0 + timedelta(hours=11, minutes=30)))

    def clock() -> datetime:
        return next(call_times)

    tokens: Final = tuple(sap_credentials.get_token_creator(clock=clock)[0]() for _ in range(3))

    assert tokens == ("Bearer first", "Bearer first", "Bearer second")
    assert aicore.token.call_count == 2


def test_each_service_key_keeps_its_own_token(aicore: MockAICore) -> None:
    aicore.token.side_effect = [
        httpx.Response(200, json={"access_token": "first", "expires_in": 43200}),
        httpx.Response(200, json={"access_token": "second", "expires_in": 43200}),
    ]
    service_key: Final = json.loads(litellm.sap_service_key)
    first_creator: Final = sap_credentials.get_token_creator(service_key)[0]
    second_creator: Final = sap_credentials.get_token_creator({**service_key, "clientid": "other-client"})[0]

    tokens: Final = (first_creator(), second_creator(), first_creator(), second_creator())

    assert tokens == ("Bearer first", "Bearer second", "Bearer first", "Bearer second")
    assert aicore.token.call_count == 2


def test_short_lived_token_is_replaced_halfway_through_its_lifetime(aicore: MockAICore) -> None:
    aicore.token.side_effect = [
        httpx.Response(200, json={"access_token": "first", "expires_in": 3000}),
        httpx.Response(200, json={"access_token": "second", "expires_in": 3000}),
    ]
    call_times: Final = iter((_T0, _T0 + timedelta(minutes=25), _T0 + timedelta(minutes=26)))

    def clock() -> datetime:
        return next(call_times)

    tokens: Final = tuple(sap_credentials.get_token_creator(clock=clock)[0]() for _ in range(3))

    assert tokens == ("Bearer first", "Bearer first", "Bearer second")
    assert aicore.token.call_count == 2


class _Clock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def __call__(self) -> datetime:
        return self._now

    def move_to(self, now: datetime) -> None:
        self._now = now


def test_a_failed_early_refresh_reuses_the_cached_token_until_it_expires(aicore: MockAICore) -> None:
    aicore.token.side_effect = [
        httpx.Response(200, json={"access_token": "first", "expires_in": 43200}),
        httpx.Response(503, text="Service Unavailable"),
        httpx.Response(503, text="Service Unavailable"),
    ]
    clock: Final = _Clock(_T0)
    token_creator: Final = sap_credentials.get_token_creator(clock=clock)[0]

    before_outage: Final = token_creator()
    clock.move_to(_T0 + timedelta(hours=11, minutes=30))
    first_read_in_outage: Final = token_creator()
    clock.move_to(_T0 + timedelta(hours=11, minutes=31))
    next_read_in_outage: Final = token_creator()
    clock.move_to(_T0 + timedelta(hours=12, seconds=1))

    with pytest.raises(RuntimeError, match="Service Unavailable"):
        token_creator()
    assert (before_outage, first_read_in_outage, next_read_in_outage) == ("Bearer first",) * 3
    assert aicore.token.call_count == 3


@pytest.mark.parametrize("while_refreshing", ["rejected", "expired"])
def test_a_failed_refresh_never_returns_a_token_that_stopped_working_while_it_ran(
    while_refreshing: Literal["rejected", "expired"],
) -> None:
    clock: Final = _Clock(_T0)
    first_fetch: Final = iter((("Bearer first", timedelta(hours=12)),))

    def refresh_that_fails() -> tuple[str, timedelta]:
        token: Final = next(first_fetch, None)
        if token is not None:
            return token
        if while_refreshing == "rejected":
            another_request_gets_401: Final = threading.Thread(target=token_cache.invalidate, args=("Bearer first",))
            another_request_gets_401.start()
            another_request_gets_401.join(timeout=5)
        else:
            clock.move_to(_T0 + timedelta(hours=12))
        raise RuntimeError("Token request failed: Service Unavailable")

    token_cache: Final = sap_credentials.OAuthTokenCache(
        refresh_that_fails, timedelta(minutes=60), clock, threading.Lock(), threading.Lock()
    )
    assert token_cache() == "Bearer first"
    clock.move_to(_T0 + timedelta(hours=11, minutes=30))

    with pytest.raises(RuntimeError, match="Service Unavailable"):
        token_cache()


def test_a_rejected_token_is_dropped_without_waiting_for_a_token_fetch_in_progress() -> None:
    fetch_started: Final = threading.Event()
    rejection_handled: Final = threading.Event()

    def fetch_that_finishes_after_the_rejection() -> tuple[str, timedelta]:
        fetch_started.set()
        assert rejection_handled.wait(timeout=5)
        return "Bearer fresh", timedelta(hours=12)

    token_cache: Final = sap_credentials.OAuthTokenCache(
        fetch_that_finishes_after_the_rejection,
        timedelta(minutes=60),
        lambda: _T0,
        threading.Lock(),
        threading.Lock(),
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        fetching: Final = pool.submit(token_cache)
        assert fetch_started.wait(timeout=5)
        token_cache.invalidate("Bearer revoked")
        rejection_handled.set()
        fresh_token: Final = fetching.result()

    assert fresh_token == "Bearer fresh"
    assert token_cache() == "Bearer fresh"


class _LockThatFlagsContention:
    def __init__(self, contended: threading.Event) -> None:
        self._contended: Final = contended
        self._lock: Final = threading.Lock()

    def __enter__(self) -> bool:
        if self._lock.acquire(blocking=False):
            return True
        self._contended.set()
        return self._lock.acquire()

    def __exit__(self, *_: object) -> None:
        self._lock.release()


def test_a_rejection_that_arrives_while_a_failed_refresh_falls_back_is_never_undone() -> None:
    contended: Final = threading.Event()
    refresh_failed: Final = threading.Event()
    rejection_sent: Final = threading.Event()
    rejections: Final = queue.SimpleQueue[threading.Thread]()
    first_fetch: Final = iter((("Bearer first", timedelta(hours=12)),))
    call_times: Final = iter((_T0,))

    def refresh_that_fails() -> tuple[str, timedelta]:
        token: Final = next(first_fetch, None)
        if token is not None:
            return token
        refresh_failed.set()
        raise RuntimeError("Token request failed: Service Unavailable")

    def clock_that_lets_a_401_arrive_once_the_refresh_failed() -> datetime:
        if refresh_failed.is_set() and not rejection_sent.is_set():
            rejection_sent.set()
            another_request_gets_401: Final = threading.Thread(target=token_cache.invalidate, args=("Bearer first",))
            rejections.put(another_request_gets_401)
            another_request_gets_401.start()
            contended.wait(timeout=5)
        return next(call_times, _T0 + timedelta(hours=11, minutes=30))

    token_cache: Final = sap_credentials.OAuthTokenCache(
        refresh_that_fails,
        timedelta(minutes=60),
        clock_that_lets_a_401_arrive_once_the_refresh_failed,
        threading.Lock(),
        _LockThatFlagsContention(contended),
    )
    assert token_cache() == "Bearer first"
    token_cache()
    rejections.get().join(timeout=5)

    with pytest.raises(RuntimeError, match="Service Unavailable"):
        token_cache()


@pytest.mark.parametrize(
    "make_config", [GenAIHubOrchestrationConfig, GenAIHubEmbeddingConfig], ids=["chat", "embedding"]
)
def test_a_late_rejection_of_a_replaced_token_keeps_the_new_token(
    aicore: MockAICore, make_config: Callable[[], GenAIHubOrchestrationConfig | GenAIHubEmbeddingConfig]
) -> None:
    aicore.token.side_effect = [
        httpx.Response(200, json={"access_token": "first", "expires_in": 0}),
        httpx.Response(200, json={"access_token": "second", "expires_in": 43200}),
    ]
    slow_request: Final = make_config()
    slow_request.sign_request({}, {}, {}, aicore.deployment_url)
    make_config().sign_request({}, {}, {}, aicore.deployment_url)

    slow_request.get_error_class("Unauthorized", 401, {})

    assert make_config().sign_request({}, {}, {}, aicore.deployment_url)[0]["Authorization"] == "Bearer second"
    assert aicore.token.call_count == 2


class _LockThatCountsArrivals:
    def __init__(self, arrived: threading.Semaphore) -> None:
        self._arrived: Final = arrived
        self._lock: Final = threading.Lock()

    def __enter__(self) -> bool:
        self._arrived.release()
        return self._lock.acquire()

    def __exit__(self, *_: object) -> None:
        self._lock.release()


def _results_of_concurrent_requests(
    token_cache: sap_credentials.OAuthTokenCache, count: int
) -> tuple[str | RuntimeError, ...]:
    def request_token(_: int) -> str | RuntimeError:
        try:
            return token_cache()
        except RuntimeError as err:
            return err

    with ThreadPoolExecutor(max_workers=count) as pool:
        return tuple(pool.map(request_token, range(count)))


def test_requests_waiting_on_a_failing_token_fetch_share_its_error() -> None:
    concurrent_requests: Final = 4
    arrived: Final = threading.Semaphore(0)
    fetches: Final = queue.SimpleQueue[None]()

    def fetch_fails_once_every_request_is_waiting() -> tuple[str, timedelta]:
        fetches.put(None)
        assert all(arrived.acquire(timeout=5) for _ in range(concurrent_requests))
        raise RuntimeError("Token request failed: Service Unavailable")

    token_cache: Final = sap_credentials.OAuthTokenCache(
        fetch_fails_once_every_request_is_waiting,
        timedelta(minutes=60),
        lambda: _T0,
        _LockThatCountsArrivals(arrived),
        threading.Lock(),
    )

    results: Final = _results_of_concurrent_requests(token_cache, concurrent_requests)

    assert tuple(map(str, results)) == ("Token request failed: Service Unavailable",) * concurrent_requests
    assert len({id(error) for error in results}) == concurrent_requests
    assert fetches.qsize() == 1


def test_requests_waiting_on_a_failing_refresh_reuse_the_unexpired_cached_token() -> None:
    concurrent_requests: Final = 4
    arrived: Final = threading.Semaphore(0)
    fetches: Final = queue.SimpleQueue[None]()
    first_fetch: Final = iter((("Bearer first", timedelta(hours=12)),))
    call_times: Final = iter((_T0,))

    def refresh_fails_once_every_request_is_waiting() -> tuple[str, timedelta]:
        fetches.put(None)
        token: Final = next(first_fetch, None)
        if token is not None:
            return token
        assert all(arrived.acquire(timeout=5) for _ in range(concurrent_requests))
        raise RuntimeError("Token request failed: Service Unavailable")

    token_cache: Final = sap_credentials.OAuthTokenCache(
        refresh_fails_once_every_request_is_waiting,
        timedelta(minutes=60),
        lambda: next(call_times, _T0 + timedelta(hours=11, minutes=30)),
        _LockThatCountsArrivals(arrived),
        threading.Lock(),
    )
    assert token_cache() == "Bearer first"
    assert arrived.acquire(timeout=5)

    results: Final = _results_of_concurrent_requests(token_cache, concurrent_requests)

    assert results == ("Bearer first",) * concurrent_requests
    assert fetches.qsize() == 2
