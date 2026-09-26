"""The owned proxy every canary scenario runs against, with its provider and sink doubles.

``canary_rig(root)`` starts a provider double and a ``generic_api`` sink double, writes an owned
config derived from ``tests/integration/proxy_config.yaml`` and starts an owned proxy on it:

- ``store_prompts_in_spend_logs`` is on, so the stored request body exists for every sweep;
- Redis response-cache entries live 600 s, longer than any scenario's sweeps;
- spend logs flush every second (``proxy_batch_write_at``) and callbacks flush every second
  (``DEFAULT_FLUSH_INTERVAL_SECONDS``), so ``eventually`` converges quickly;
- provider-default routes (file, batch, container lists with no deployment) resolve to the
  provider double through ``OPENAI_BASE_URL``, and the remote catalogs (cost map, blog posts,
  beta headers, autorouter presets, policy templates) are read from the package, so a route
  sweep never leaves the machine;
- ``HTTP(S)_PROXY`` points at an egress trap that answers every connection with 403 and
  records its first line; the rig fails on exit if the proxy tried to reach any non-loopback
  host (``Rig.egress()`` lists the attempts so far);
- the config ``model_list`` declares ``CONFIG_MODEL`` whose ``api_key`` is a fresh slot B1
  canary, reaching the provider double at ``<provider>/v1``.

API:

- ``canary_rig(root, *, configure=None, environment=None, upstream=None, sink_token=SINK_TOKEN)
  -> Iterator[Rig]``. ``configure(config, provider_url)`` may edit the parsed config before it
  is written (add deployments, settings, callbacks); ``environment`` adds or overrides proxy
  environment variables; ``upstream`` replaces ``chat_upstream`` as the provider double's
  handler. ``sink_token`` is the bearer the ``generic_api`` double requires and
  ``GENERIC_LOGGER_HEADERS`` sends; pass a ``Canary`` (slot G1 style) to plant a sink credential,
  and ``Rig.own_headers`` then allows that one header to carry it (pass it to ``sweep_all``).
- ``Rig.proxy``: the owned proxy ``Gateway`` (master key). ``Rig.canaries``: config-held
  canaries by slot id. ``Rig.provider`` and ``Rig.sinks[name]``: ``Recorder`` objects whose
  ``requests()`` returns every request received so far (the underlying queue is drained into a
  list, so repeated polls keep earlier requests).
- ``chat_upstream(request)``: an OpenAI chat double that echoes the last user message and
  answers HTTP 400 when the message contains ``PROVIDER_4XX``.
- ``SINK_TOKEN``: the default static bearer the ``generic_api`` sink authenticates with.
- ``team_caller(scenario) -> Caller``: a team, an ``internal_user`` on it and a virtual key for
  that user on that team (allowed ``CONFIG_MODEL``). Scenarios send traffic with ``Caller.key``
  and pass ``Caller.callers(rig)`` to ``sweep_all`` so S2 reads every route as the admin and as
  the internal user.
- ``settle(rig, request_id, marker)``: wait (bounded) until the spend row for ``request_id``
  is written and every sink double has received an event carrying ``marker``, so the sweeps
  that follow read the finished state instead of racing the asynchronous writers.
"""

from __future__ import annotations

import json
import socket
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.security._canary import Canary, canary

STOCK_CONFIG: Final = Path("tests/integration/proxy_config.yaml")
CONFIG_MODEL: Final = "canary-config-deployment"
PROVIDER_4XX: Final = "canary-provider-4xx"
SINK_TOKEN: Final = "synthetic-canary-sink-token"
GENERIC_SINK: Final = "generic_api"
LOCAL_CATALOGS: Final = MappingProxyType(
    {
        name: "True"
        for name in (
            "LITELLM_LOCAL_MODEL_COST_MAP",
            "LITELLM_LOCAL_BLOG_POSTS",
            "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS",
            "LITELLM_LOCAL_AUTOROUTER_PRESETS",
            "LITELLM_LOCAL_POLICY_TEMPLATES",
        )
    }
)


@dataclass(slots=True)
class Recorder:
    wire: Wire
    seen: list[Request] = field(default_factory=list)  # mutable-ok: drain() consumes the queue

    @property
    def url(self) -> str:
        return self.wire.url

    def requests(self) -> tuple[Request, ...]:
        self.seen.extend(self.wire.drain())
        return tuple(self.seen)

    def carrying(self, text: str) -> tuple[Request, ...]:
        """Requests whose body contains ``text``."""
        return tuple(request for request in self.requests() if text.encode() in request.body)


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    owned: OwnedProxy
    provider: Recorder
    sinks: Mapping[str, Recorder]
    canaries: Mapping[str, Canary]
    own_headers: Mapping[str, tuple[str, str]] = field(default_factory=lambda: MappingProxyType({}))
    egress: Callable[[], tuple[bytes, ...]] = field(default=lambda: ())


def chat_upstream(request: Request) -> Reply:
    body: Final = json.loads(request.body or b"{}")
    messages: Final = body.get("messages") or [{"content": ""}]
    text: Final = str(messages[-1].get("content", ""))
    if PROVIDER_4XX in text:
        return Reply(
            status=400,
            body=json.dumps(
                {"error": {"type": "invalid_request_error", "code": "canary_rejected", "message": "rejected"}}
            ).encode(),
        )
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "echo " + text}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            }
        ).encode()
    )


def _sink_for(token: str) -> Callable[[Request], Reply]:
    def sink(request: Request) -> Reply:
        assert request.headers.get("authorization") == f"Bearer {token}", "sink double got a foreign bearer"
        return Reply()

    return sink


@contextmanager
def _egress_trap() -> Iterator[tuple[str, Callable[[], tuple[bytes, ...]]]]:
    """A forward-proxy stand-in: records the first line of every connection, answers 403."""
    attempts: Final[list[bytes]] = []  # mutable-ok: appended by the accept thread
    server: Final = socket.create_server(("127.0.0.1", 0))
    server.settimeout(0.2)
    stopped: Final = threading.Event()

    def serve() -> None:
        while not stopped.is_set():
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with connection:
                connection.settimeout(2)
                try:
                    attempts.append(connection.recv(512).split(b"\r\n", 1)[0])
                    connection.sendall(b"HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                except OSError:
                    pass

    thread: Final = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.getsockname()[1]}", lambda: tuple(attempts)
    finally:
        stopped.set()
        thread.join(timeout=5)
        server.close()


def _config(
    root: Path, provider_url: str, b1: Canary, configure: Callable[[dict[str, object], str], None] | None
) -> Path:
    config: Final = yaml.safe_load(STOCK_CONFIG.read_text())
    config["model_list"] = [
        {
            "model_name": CONFIG_MODEL,
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": provider_url + "/v1", "api_key": b1.value},
        }
    ]
    config["general_settings"]["store_prompts_in_spend_logs"] = True
    config["litellm_settings"]["cache_params"]["ttl"] = 600
    config["litellm_settings"].update({"callbacks": [GENERIC_SINK], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1})
    if configure is not None:
        configure(config, provider_url)
    path: Final = root / f"canary-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def canary_rig(
    root: Path,
    *,
    configure: Callable[[dict[str, object], str], None] | None = None,
    environment: Mapping[str, str] | None = None,
    upstream: Callable[[Request], Reply] | None = None,
    sink_token: str | Canary = SINK_TOKEN,
) -> Iterator[Rig]:
    b1: Final = canary("B1")
    token: Final = sink_token.value if isinstance(sink_token, Canary) else sink_token
    own_headers: Final = MappingProxyType(
        {GENERIC_SINK: ("authorization", sink_token.slot)} if isinstance(sink_token, Canary) else {}
    )
    planted: Final = {"B1": b1, **({sink_token.slot: sink_token} if isinstance(sink_token, Canary) else {})}
    with (
        gateway_from_environment() as gateway,
        wire_server(upstream or chat_upstream) as provider,
        wire_server(_sink_for(token)) as sink,
        _egress_trap() as (trap_url, egress),
    ):
        config: Final = _config(root, provider.url, b1, configure)
        overrides: Final = {
            **LOCAL_CATALOGS,
            **{name: trap_url for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")},
            **{name: "127.0.0.1,localhost" for name in ("NO_PROXY", "no_proxy")},
            "OPENAI_BASE_URL": provider.url + "/v1",
            "OPENAI_API_BASE": provider.url + "/v1",
            "GENERIC_LOGGER_ENDPOINT": sink.url,
            "GENERIC_LOGGER_HEADERS": f"Authorization=Bearer {token}",
            **(environment or {}),
        }
        with owned_proxy_process(gateway, root, overrides, config=config) as owned:
            yield Rig(
                owned.gateway,
                owned,
                Recorder(provider),
                MappingProxyType({GENERIC_SINK: Recorder(sink)}),
                MappingProxyType(planted),
                own_headers,
                egress,
            )
        assert egress() == (), f"Owned proxy tried to reach external hosts: {sorted(set(egress()))}"


@dataclass(frozen=True, slots=True)
class Caller:
    team_id: str
    user_id: str
    key: str

    def callers(self, rig: Rig) -> Mapping[str, str]:
        return {"admin": rig.proxy.key, "internal_user": self.key}


def team_caller(scenario: Scenario) -> Caller:
    team: Final = scenario.team()
    user: Final = scenario.user(user_role="internal_user")
    scenario.gateway.post("/team/member_add", {"team_id": team, "member": {"user_id": user, "role": "user"}})
    key: Final = scenario.key(team_id=team, user_id=user, models=[CONFIG_MODEL])
    return Caller(team, user, key)


def settle(rig: Rig, request_id: str, marker: Canary) -> None:
    eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)),
        lambda rows: len(rows) == 1,
        seconds=70,
    )
    for sink in rig.sinks.values():
        eventually(lambda sink=sink: sink.carrying(marker.core), bool, seconds=30)
