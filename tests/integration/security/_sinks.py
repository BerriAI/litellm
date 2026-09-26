"""The owned proxy every canary scenario runs against, with its provider and sink doubles.

``canary_rig(root)`` starts a provider double and a ``generic_api`` sink double, writes an owned
config derived from ``tests/integration/proxy_config.yaml`` and starts an owned proxy on it:

- ``store_prompts_in_spend_logs`` is on, so the stored request body exists for every sweep;
- spend logs flush every second (``proxy_batch_write_at``) and callbacks flush every second
  (``DEFAULT_FLUSH_INTERVAL_SECONDS``), so ``eventually`` converges quickly;
- provider-default routes (file, batch, container lists with no deployment) resolve to the
  provider double through ``OPENAI_BASE_URL``, so a route sweep never leaves the machine;
- the config ``model_list`` declares ``CONFIG_MODEL`` whose ``api_key`` is a fresh slot B1
  canary, reaching the provider double at ``<provider>/v1``.

API:

- ``canary_rig(root, *, configure=None, environment=None, upstream=None) -> Iterator[Rig]``.
  ``configure(config, provider_url)`` may edit the parsed config before it is written (add
  deployments, settings, callbacks); ``environment`` adds or overrides proxy environment
  variables; ``upstream`` replaces ``chat_upstream`` as the provider double's handler.
- ``Rig.proxy``: the owned proxy ``Gateway`` (master key). ``Rig.canaries``: config-held
  canaries by slot id. ``Rig.provider`` and ``Rig.sinks[name]``: ``Recorder`` objects whose
  ``requests()`` returns every request received so far (the underlying queue is drained into a
  list, so repeated polls keep earlier requests).
- ``chat_upstream(request)``: an OpenAI chat double that echoes the last user message and
  answers HTTP 400 when the message contains ``PROVIDER_4XX``.
- ``SINK_TOKEN``: the static bearer the ``generic_api`` sink authenticates with.
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
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
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


def _sink(request: Request) -> Reply:
    assert request.headers.get("authorization") == f"Bearer {SINK_TOKEN}", request.headers
    return Reply()


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
) -> Iterator[Rig]:
    b1: Final = canary("B1")
    with (
        gateway_from_environment() as gateway,
        wire_server(upstream or chat_upstream) as provider,
        wire_server(_sink) as sink,
    ):
        config: Final = _config(root, provider.url, b1, configure)
        overrides: Final = {
            "OPENAI_BASE_URL": provider.url + "/v1",
            "OPENAI_API_BASE": provider.url + "/v1",
            "GENERIC_LOGGER_ENDPOINT": sink.url,
            "GENERIC_LOGGER_HEADERS": f"Authorization=Bearer {SINK_TOKEN}",
            **(environment or {}),
        }
        with owned_proxy_process(gateway, root, overrides, config=config) as owned:
            yield Rig(owned.gateway, owned, Recorder(provider), {GENERIC_SINK: Recorder(sink)}, {"B1": b1})


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
