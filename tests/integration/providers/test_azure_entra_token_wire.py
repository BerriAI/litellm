import base64
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from queue import Empty, SimpleQueue
from types import MappingProxyType
from typing import Final, TypeVar
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.forward_proxy import tunnelling_forward_proxy
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.server import ServerConnection, serve

pytestmark: Final = [
    pytest.mark.timeout(2 * graceful_stop_seconds() + 120),
    pytest.mark.xdist_group("azure_entra_token_wire"),
    pytest.mark.shares_worker_state,
]

T = TypeVar("T")

_AUTHORITY: Final = "login.microsoftonline.com"
_TENANT: Final = "scripted-tenant"
_SERVICE_CLIENT: Final = "scripted-service-client"
_PUBLIC_CLIENT: Final = "scripted-public-client"
_DEPLOYMENT: Final = "gpt-5.4-nano"
_REALTIME_DEPLOYMENT: Final = "gpt-realtime-2.1"
_LEGACY_API_VERSION: Final = "2024-10-21"
_TWIN_API_VERSION: Final = "2025-03-01-preview"
_REALTIME_API_VERSION: Final = "2025-04-01-preview"
_BING_MODEL: Final = "gpt-4.1"
_BING_SCOPE: Final = "https://ai.azure.com/.default"
_LONG_LIVED: Final = 3600
_NEAR_EXPIRY: Final = 60
_PROBE_SECONDS: Final = 2.0
_TOKEN_PATH: Final = "/oauth2/v2.0/token"
_AZURE_ENVIRONMENT: Final = (
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_AD_TOKEN",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
    "AZURE_USERNAME",
    "AZURE_PASSWORD",
    "AZURE_SCOPE",
    "AZURE_AUTHORITY_HOST",
    "AZURE_FEDERATED_TOKEN_FILE",
    "AZURE_CREDENTIAL",
    "AZURE_CERTIFICATE_PATH",
    "AZURE_CERTIFICATE_PASSWORD",
    "BING_GROUNDING_TOKEN",
    "BING_GROUNDING_PROJECT_ENDPOINT",
    "BING_GROUNDING_MODEL",
)


@dataclass(frozen=True, slots=True)
class Hold:
    expires_in: int = _LONG_LIVED
    arrived: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True, slots=True)
class Authority:
    wire: Wire
    holds: SimpleQueue[Hold]

    def token_forms(self) -> tuple[Mapping[str, str], ...]:
        posted: Final = tuple(request for request in self.wire.drain() if request.method == "POST")
        return tuple(_form(request) for request in posted if urlsplit(request.target).path.endswith(_TOKEN_PATH))


@dataclass(frozen=True, slots=True)
class Entra:
    authority: Authority
    environment: Mapping[str, str]
    certificate: tuple[Path, Path]


@dataclass(frozen=True, slots=True)
class RealtimePeer:
    url: str
    handshakes: SimpleQueue[str]

    def authorizations(self) -> tuple[str, ...]:
        return tuple(self.handshakes.get_nowait() for _ in range(self.handshakes.qsize()))


def _form(request: Request) -> Mapping[str, str]:
    return MappingProxyType({name: values[-1] for name, values in parse_qs(request.body.decode()).items()})


def _encoded(claims: Mapping[str, str | int]) -> str:
    return base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")


def _issuer(tenant: str) -> str:
    return f"https://{_AUTHORITY}/{tenant}/v2.0"


def _openid_configuration(tenant: str) -> bytes:
    endpoint: Final = f"https://{_AUTHORITY}/{tenant}/oauth2/v2.0"
    return json.dumps(
        {
            "issuer": _issuer(tenant),
            "authorization_endpoint": f"{endpoint}/authorize",
            "token_endpoint": f"{endpoint}/token",
            "device_authorization_endpoint": f"{endpoint}/devicecode",
        }
    ).encode()


def _ropc_token(username: str, password: str) -> str:
    return f"ropc.{username}.{password}"


def _secret_token(scope: str) -> str:
    return f"secret.{scope}"


def _exchanged_token(assertion: str) -> str:
    return f"exchanged.{assertion}"


def _account_claims(tenant: str, form: Mapping[str, str]) -> Mapping[str, str]:
    now: Final = int(time.time())
    id_token: Final = (
        _encoded({"alg": "none", "typ": "JWT"})
        + "."
        + _encoded(
            {
                "iss": _issuer(tenant),
                "aud": form["client_id"],
                "iat": now,
                "nbf": now,
                "exp": now + _LONG_LIVED,
                "sub": "scripted-subject",
                "oid": "scripted-object",
                "tid": _TENANT,
                "preferred_username": form["username"],
            }
        )
        + "."
    )
    return MappingProxyType(
        {
            "id_token": id_token,
            "client_info": _encoded({"uid": "scripted-object", "utid": _TENANT}),
        }
    )


def _token_reply(tenant: str, form: Mapping[str, str], expires_in: int) -> bytes:
    lifetime: Final = {"token_type": "Bearer", "expires_in": expires_in, "ext_expires_in": expires_in}
    match form.get("grant_type"), form.get("client_assertion"):
        case "password", _:
            return json.dumps(
                {
                    **lifetime,
                    "access_token": _ropc_token(form["username"], form["password"]),
                    "scope": form["scope"],
                    **_account_claims(tenant, form),
                }
            ).encode()
        case "client_credentials", None:
            return json.dumps({**lifetime, "access_token": _secret_token(form["scope"])}).encode()
        case "client_credentials", assertion:
            return json.dumps({**lifetime, "access_token": _exchanged_token(assertion)}).encode()
        case grant:
            raise AssertionError(f"unscripted grant {grant}")


def _next_hold(holds: SimpleQueue[Hold]) -> Hold | None:
    try:
        return holds.get_nowait()
    except Empty:
        return None


def _scripted_authority(holds: SimpleQueue[Hold]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        tenant: Final = path.strip("/").split("/")[0]
        if path.endswith("/.well-known/openid-configuration"):
            return Reply(body=_openid_configuration(tenant))
        if not path.endswith(_TOKEN_PATH):
            return Reply(status=404)
        hold: Final = _next_hold(holds)
        body: Final = _token_reply(tenant, _form(request), _LONG_LIVED if hold is None else hold.expires_in)
        if hold is None:
            return Reply(body=body)
        hold.arrived.set()
        return Reply(chunks=(body,), gate_before_first=hold.release)

    return respond


def _azure_echo(request: Request) -> Reply:
    authorization: Final = request.headers.get("authorization", "")
    if urlsplit(request.target).path.endswith("/responses"):
        return responses_reply("resp_entra", _DEPLOYMENT, authorization, stream=False)
    return chat_reply("chatcmpl-entra", _DEPLOYMENT, authorization, stream=False)


def _foundry(_: Request) -> Reply:
    return Reply(body=json.dumps({"id": "resp_bing", "object": "response", "output": []}).encode())


def _port(url: str) -> int:
    port: Final = urlsplit(url).port
    assert port is not None, url
    return port


@pytest.fixture(scope="module")
def entra(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Entra]:
    certificate: Final = write_self_signed_cert(tmp_path_factory.mktemp("azure-entra-authority"), (_AUTHORITY,))
    holds: Final[SimpleQueue[Hold]] = SimpleQueue()
    with (
        wire_server(_scripted_authority(holds), tls=server_context(*certificate)) as wire,
        tunnelling_forward_proxy(_port(wire.url)) as tunnel,
    ):
        environment: Final = MappingProxyType(
            {
                "HTTPS_PROXY": tunnel.url,
                "https_proxy": tunnel.url,
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
                "SSL_CERT_FILE": str(certificate[0]),
                "REQUESTS_CA_BUNDLE": str(certificate[0]),
            }
        )
        yield Entra(Authority(wire, holds), environment, certificate)


@pytest.fixture(scope="module")
def azure() -> Iterator[Wire]:
    with wire_server(_azure_echo) as wire:
        yield wire


@pytest.fixture(scope="module")
def foundry() -> Iterator[Wire]:
    with wire_server(_foundry) as wire:
        yield wire


@pytest.fixture(scope="module")
def realtime(entra: Entra) -> Iterator[RealtimePeer]:
    handshakes: Final[SimpleQueue[str]] = SimpleQueue()

    def session(connection: ServerConnection) -> None:
        assert connection.request is not None
        handshakes.put(connection.request.headers.get("Authorization", ""))
        connection.send(json.dumps({"type": "session.created", "event_id": "event_entra", "session": {}}))
        with suppress(ConnectionClosed):
            connection.recv()

    with serve(session, "127.0.0.1", 0, ssl=server_context(*entra.certificate)) as server:
        thread: Final = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield RealtimePeer(f"https://127.0.0.1:{server.socket.getsockname()[1]}", handshakes)
        finally:
            server.shutdown()
            thread.join(timeout=6)


@pytest.fixture(scope="module")
def user_gateway(tmp_path_factory: pytest.TempPathFactory, entra: Entra) -> Iterator[Gateway]:
    with (
        gateway_from_environment() as parent,
        owned_proxy(
            parent,
            tmp_path_factory.mktemp("azure-entra-user-proxy"),
            entra.environment,
            remove_environment=_AZURE_ENVIRONMENT,
        ) as owned,
    ):
        yield owned


@pytest.fixture(scope="module")
def service_gateway(tmp_path_factory: pytest.TempPathFactory, entra: Entra, foundry: Wire) -> Iterator[Gateway]:
    overrides: Final = {
        **entra.environment,
        "AZURE_CLIENT_ID": _SERVICE_CLIENT,
        "AZURE_CLIENT_SECRET": "scripted-service-secret",
        "AZURE_TENANT_ID": _TENANT,
        "BING_GROUNDING_PROJECT_ENDPOINT": foundry.url,
        "BING_GROUNDING_MODEL": _BING_MODEL,
    }
    with (
        gateway_from_environment() as parent,
        owned_proxy(
            parent,
            tmp_path_factory.mktemp("azure-entra-service-proxy"),
            overrides,
            remove_environment=tuple(name for name in _AZURE_ENVIRONMENT if name not in overrides),
        ) as owned,
    ):
        yield owned


@pytest.fixture(autouse=True)
def _drained_peers(entra: Entra, azure: Wire, foundry: Wire) -> None:
    entra.authority.wire.drain()
    azure.drain()
    foundry.drain()


def _username(purpose: str) -> str:
    return f"{purpose}-{uuid.uuid4().hex[:12]}@scripted.example"


def _audience(purpose: str) -> str:
    return f"api://{purpose}-{uuid.uuid4().hex[:12]}/.default"


def _user_deployment(scenario: Scenario, api_base: str, username: str, password: str, api_version: str) -> str:
    return scenario.model(
        model=f"azure/{_DEPLOYMENT}",
        api_base=api_base,
        api_key=None,
        api_version=api_version,
        azure_username=username,
        azure_password=password,
        client_id=_PUBLIC_CLIENT,
    )


def _oidc_deployment(scenario: Scenario, azure: Wire, audience: str, api_version: str) -> str:
    return scenario.model(
        model=f"azure/{_DEPLOYMENT}",
        api_base=azure.url,
        api_key=None,
        api_version=api_version,
        azure_ad_token=f"oidc/azure/{audience}",
    )


def _chat_authorization(gateway: Gateway, model: str) -> str:
    response: Final = gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "entra"}]}
    )
    assert response.status_code == 200, response.text
    choice: Final = object_value(list_value(JSON_OBJECT.validate_json(response.content)["choices"])[0])
    return string_value(object_value(choice["message"])["content"])


def _responses_authorization(gateway: Gateway, model: str) -> str:
    response: Final = gateway.request("POST", "/v1/responses", {"model": model, "input": "entra"})
    assert response.status_code == 200, response.text
    message: Final = object_value(list_value(JSON_OBJECT.validate_json(response.content)["output"])[0])
    return string_value(object_value(list_value(message["content"])[0])["text"])


def _realtime_first_event(gateway: Gateway, model: str) -> str:
    url: Final = f"ws://127.0.0.1:{_port(str(gateway.client.base_url))}/v1/realtime?model={model}"
    with connect(
        url, additional_headers={"Authorization": f"Bearer {gateway.key}"}, open_timeout=30, proxy=None
    ) as socket:
        return string_value(JSON_OBJECT.validate_json(socket.recv(timeout=60))["type"])


def _bing_tool(gateway: Gateway, scenario: Scenario) -> str:
    name: Final = f"bing-{uuid.uuid4().hex[:12]}"
    created: Final = gateway.post(
        "/search_tools",
        {"search_tool": {"search_tool_name": name, "litellm_params": {"search_provider": "bing_grounding"}}},
    )
    scenario.cleanups.callback(gateway.request, "DELETE", f"/search_tools/{created['search_tool_id']}")
    return name


def _search(gateway: Gateway, tool: str) -> str:
    response: Final = gateway.request("POST", f"/v1/search/{tool}", {"query": "entra"})
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["object"])


def _liveliness(gateway: Gateway) -> int | None:
    try:
        return gateway.client.get("/health/liveliness", timeout=_PROBE_SECONDS).status_code
    except httpx.TimeoutException:
        return None


def _probe_while_held(gateway: Gateway, hold: Hold, pending: Callable[[], bool]) -> int | None:
    try:
        eventually(lambda: hold.arrived.is_set() or pending(), bool, seconds=30)
        assert hold.arrived.is_set(), "the request finished without minting an Entra token"
        return _liveliness(gateway)
    finally:
        hold.release.set()


def _release(queue: SimpleQueue[Hold], holds: tuple[Hold, ...]) -> None:
    unclaimed: Final = tuple(iter(partial(_next_hold, queue), None))
    for hold in (*holds, *unclaimed):
        hold.release.set()


def _probed_during_mints(
    gateway: Gateway, entra: Entra, send: Callable[[], T], holds: tuple[Hold, ...]
) -> tuple[tuple[int | None, ...], T]:
    for hold in holds:
        entra.authority.holds.put(hold)
    with ThreadPoolExecutor(max_workers=1) as pool:
        sent: Final = pool.submit(send)
        try:
            probes: Final = tuple(_probe_while_held(gateway, hold, sent.done) for hold in holds)
        finally:
            _release(entra.authority.holds, holds)
        return probes, sent.result(timeout=60)


def test_username_password_deployments_share_one_entra_token_until_the_password_changes(
    user_gateway: Gateway, entra: Entra, azure: Wire
) -> None:
    username: Final = _username("reuse")
    with user_gateway.scenario() as scenario:
        original: Final = _user_deployment(scenario, azure.url, username, "first-password", _LEGACY_API_VERSION)
        twin: Final = _user_deployment(scenario, azure.url, username, "first-password", _TWIN_API_VERSION)
        rotated: Final = _user_deployment(scenario, azure.url, username, "rotated-password", _LEGACY_API_VERSION)
        answers: Final = tuple(
            _chat_authorization(user_gateway, model) for model in (original, twin, original, rotated)
        )
        minted: Final = tuple(form["password"] for form in entra.authority.token_forms() if "password" in form)

    first: Final = f"Bearer {_ropc_token(username, 'first-password')}"
    assert answers == (first, first, first, f"Bearer {_ropc_token(username, 'rotated-password')}")
    assert minted == ("first-password", "rotated-password")


def test_a_cold_burst_across_deployments_sharing_a_user_mints_one_entra_token(
    user_gateway: Gateway, entra: Entra, azure: Wire
) -> None:
    username: Final = _username("burst")
    holds: Final = (Hold(), Hold())
    for hold in holds:
        entra.authority.holds.put(hold)
    with user_gateway.scenario() as scenario:
        models: Final = tuple(
            _user_deployment(scenario, azure.url, username, "burst-password", api_version)
            for api_version in (_LEGACY_API_VERSION, _TWIN_API_VERSION)
        )
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            sent: Final = tuple(pool.submit(_chat_authorization, user_gateway, model) for model in models)
            try:
                eventually(holds[0].arrived.is_set, bool, seconds=30)
                second_mint_started: Final = holds[1].arrived.wait(_PROBE_SECONDS)
            finally:
                _release(entra.authority.holds, holds)
            answers: Final = tuple(future.result(timeout=60) for future in sent)
        minted: Final = tuple(form["password"] for form in entra.authority.token_forms() if "password" in form)

    token: Final = f"Bearer {_ropc_token(username, 'burst-password')}"
    assert not second_mint_started
    assert answers == (token, token)
    assert minted == ("burst-password",)


def test_a_slow_entra_mint_for_legacy_chat_leaves_liveliness_answering(
    user_gateway: Gateway, entra: Entra, azure: Wire
) -> None:
    username: Final = _username("chat")
    with user_gateway.scenario() as scenario:
        model: Final = _user_deployment(scenario, azure.url, username, "chat-password", _LEGACY_API_VERSION)
        probes, answer = _probed_during_mints(
            user_gateway, entra, lambda: _chat_authorization(user_gateway, model), (Hold(),)
        )

    assert probes == (200,)
    assert answer == f"Bearer {_ropc_token(username, 'chat-password')}"


def test_a_slow_entra_mint_for_responses_leaves_liveliness_answering(
    user_gateway: Gateway, entra: Entra, azure: Wire
) -> None:
    username: Final = _username("responses")
    with user_gateway.scenario() as scenario:
        model: Final = _user_deployment(scenario, azure.url, username, "responses-password", _LEGACY_API_VERSION)
        probes, answer = _probed_during_mints(
            user_gateway, entra, lambda: _responses_authorization(user_gateway, model), (Hold(),)
        )

    assert probes == (200,)
    assert answer == f"Bearer {_ropc_token(username, 'responses-password')}"


def test_a_slow_entra_mint_for_a_realtime_session_leaves_liveliness_answering(
    user_gateway: Gateway, entra: Entra, realtime: RealtimePeer
) -> None:
    username: Final = _username("realtime")
    with user_gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure/{_REALTIME_DEPLOYMENT}",
            api_base=realtime.url,
            api_key=None,
            api_version=_REALTIME_API_VERSION,
            azure_username=username,
            azure_password="realtime-password",
            client_id=_PUBLIC_CLIENT,
        )
        probes, first_event = _probed_during_mints(
            user_gateway, entra, lambda: _realtime_first_event(user_gateway, model), (Hold(),)
        )

    assert probes == (200,)
    assert first_event == "session.created"
    assert realtime.authorizations() == (f"Bearer {_ropc_token(username, 'realtime-password')}",)


def test_the_first_oidc_azure_chat_exchanges_its_token_off_the_event_loop(
    service_gateway: Gateway, entra: Entra, azure: Wire
) -> None:
    audience: Final = _audience("first-chat")
    with service_gateway.scenario() as scenario:
        model: Final = _oidc_deployment(scenario, azure, audience, _LEGACY_API_VERSION)
        probes, answer = _probed_during_mints(
            service_gateway, entra, lambda: _chat_authorization(service_gateway, model), (Hold(),)
        )

    assert probes == (200,)
    assert answer == f"Bearer {_exchanged_token(_secret_token(audience))}"


def test_oidc_azure_deployments_share_one_entra_mint_per_audience(
    service_gateway: Gateway, entra: Entra, azure: Wire
) -> None:
    shared: Final = _audience("shared")
    other: Final = _audience("other")
    with service_gateway.scenario() as scenario:
        models: Final = (
            _oidc_deployment(scenario, azure, shared, _LEGACY_API_VERSION),
            _oidc_deployment(scenario, azure, shared, _TWIN_API_VERSION),
            _oidc_deployment(scenario, azure, other, _LEGACY_API_VERSION),
        )
        answers: Final = tuple(_chat_authorization(service_gateway, model) for model in models)
        minted: Final = tuple(form["scope"] for form in entra.authority.token_forms() if "client_secret" in form)

    assert answers == tuple(
        f"Bearer {_exchanged_token(_secret_token(audience))}" for audience in (shared, shared, other)
    )
    assert minted == (shared, other)


def test_an_oidc_token_beside_a_client_secret_is_the_credential_sent_on_every_api_version(
    service_gateway: Gateway, azure: Wire
) -> None:
    audience: Final = _audience("precedence")
    with service_gateway.scenario() as scenario:
        models: Final = tuple(
            scenario.model(
                model=f"azure/{_DEPLOYMENT}",
                api_base=azure.url,
                api_key=None,
                api_version=api_version,
                azure_ad_token=f"oidc/azure/{audience}",
                tenant_id=_TENANT,
                client_id=_SERVICE_CLIENT,
                client_secret="scripted-service-secret",
            )
            for api_version in (_LEGACY_API_VERSION, "v1")
        )
        answers: Final = tuple(_chat_authorization(service_gateway, model) for model in models)

    oidc: Final = f"Bearer {_exchanged_token(_secret_token(audience))}"
    assert answers == (oidc, oidc)


def test_bing_grounding_mints_off_the_event_loop_and_reuses_its_entra_token_across_searches(
    service_gateway: Gateway, entra: Entra, foundry: Wire
) -> None:
    with service_gateway.scenario() as scenario:
        tool: Final = _bing_tool(service_gateway, scenario)
        probes, first = _probed_during_mints(
            service_gateway, entra, lambda: _search(service_gateway, tool), (Hold(_NEAR_EXPIRY), Hold())
        )
        later: Final = tuple(_search(service_gateway, tool) for _ in range(2))
        minted: Final = tuple(form["scope"] for form in entra.authority.token_forms() if "client_secret" in form)
        authorizations: Final = tuple(request.headers.get("authorization") for request in foundry.drain())

    assert (first, *later) == ("search", "search", "search")
    assert minted == (_BING_SCOPE, _BING_SCOPE)
    assert authorizations == (f"Bearer {_secret_token(_BING_SCOPE)}",) * 3
    assert probes == (200, 200)
