"""Shared rig for control plane route tests: a routed fake transport and a licensed app over two dataplanes."""

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Final

import httpx
from fastapi.testclient import TestClient

from litellm.litellm_core_utils.oauth_token_exchange import OAuthTokenExchangeConfig
from litellm.proxy.control_plane.app import create_app
from litellm.proxy.control_plane.auth.jwt import SessionTokenCodec
from litellm.proxy.control_plane.auth.rbac import VIEWER_ROLE, RolePolicy, build_principal
from litellm.proxy.control_plane.container import build_container
from litellm.proxy.control_plane.settings import ControlPlaneSettings

MASTER_KEY: Final = "01234567890123456789012345678901"
DATAPLANES: Final = {
    "eu": {"url": "https://eu.test", "service_token": {"token": "sk-eu"}},
    "us": {"url": "https://us.test", "name": "US East", "service_token": {"token": "sk-us"}},
}


class Licensed:
    def is_premium(self) -> bool:
        return True


@dataclass
class RoutedTransport:  # mutable-ok: records calls for assertions
    """Answers each URL from its own queue of outcomes, so fan-out order does not matter to a test."""

    routes: Mapping[str, Iterator[httpx.Response | Exception]]
    calls: list[tuple[str, tuple[tuple[str, str], ...] | None, dict[str, str] | None]] = field(default_factory=list)

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: Sequence[tuple[str, str]] | None = None,
        json: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float,
    ) -> httpx.Response:
        assert method == "GET", "fleet reads never write"
        self.calls.append((url, None if params is None else tuple(params), headers))
        outcome: Final = next(self.routes[url], None)
        if outcome is None:
            raise AssertionError(f"no scripted outcome left for {url}")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def no_exchange(config: OAuthTokenExchangeConfig, subject_token: str) -> str:
    raise AssertionError("no oidc dataplane configured")


def response(status: int, body: object = None) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("GET", "https://x"))


@dataclass(frozen=True)
class Minter:
    codec: SessionTokenCodec
    policy: RolePolicy

    def headers(self, *dataplanes: str, role: str = VIEWER_ROLE) -> dict[str, str]:
        principal: Final = build_principal(
            self.policy, "alice", "alice@x.test", frozenset({role}), None, frozenset(dataplanes), "session"
        )
        return {"Authorization": f"Bearer {self.codec.mint(principal)}"}


def rig(
    routes: Mapping[str, list[httpx.Response | Exception]], usage_max_window_days: int = 366
) -> tuple[TestClient, RoutedTransport, Minter]:
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "dataplanes": DATAPLANES,
            "usage_max_window_days": usage_max_window_days,
        }
    )
    queues: defaultdict[str, Iterator[httpx.Response | Exception]] = defaultdict(lambda: iter(()))
    queues.update({url: iter(outcomes) for url, outcomes in routes.items()})
    transport: Final = RoutedTransport(queues)
    container: Final = build_container(settings, transport=transport, exchanger=no_exchange)
    codec: Final = SessionTokenCodec(MASTER_KEY, settings.session_issuer, timedelta(hours=1), container.policy)
    return TestClient(create_app(container, Licensed())), transport, Minter(codec, container.policy)
