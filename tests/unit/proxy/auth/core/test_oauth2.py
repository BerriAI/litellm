from collections.abc import Mapping
from dataclasses import dataclass, field
from ipaddress import ip_network
from types import MappingProxyType
from typing import Final

import pytest

from litellm.proxy.auth.core.answers import Introspection
from litellm.proxy.auth.core.identity import (
    DisallowedProxyMappings,
    InactiveToken,
    IntrospectionNotConfigured,
    OAuth2Identity,
    OAuth2ProxyIdentity,
    ProxyMappingsMissing,
    TrustedProxyRangesMissing,
    UntrustedProxy,
)
from litellm.proxy.auth.core.inputs import OAuth2IntrospectionSettings, OAuth2ProxySettings
from litellm.proxy.auth.core.python_core import PythonCore

CORE: Final = PythonCore()
TRUSTED: Final = (ip_network("10.0.0.0/8"),)


@dataclass
class RecordingIntrospection:
    answer: Introspection
    calls: list[str] = field(default_factory=list)  # mutable-ok: records host calls under test

    async def introspect(self, token: str) -> Introspection:
        self.calls.append(token)
        return self.answer


@dataclass
class RecordingProxy:
    ip: str | None
    headers: Mapping[str, str]
    calls: list[str] = field(default_factory=list)  # mutable-ok: records host calls under test

    def client_ip(self) -> str | None:
        self.calls.append("client_ip")
        return self.ip

    def header(self, name: str) -> str | None:
        self.calls.append(name)
        return self.headers.get(name)


def _introspection(active: bool) -> RecordingIntrospection:
    return RecordingIntrospection(Introspection(active=active, user_id="user-1", role="internal_user", team_id=7))


@pytest.mark.parametrize(
    ("active", "requires_active", "expected"),
    [
        (True, True, OAuth2Identity(token="tok", user_id="user-1", role="internal_user", team_id=7)),
        (False, True, InactiveToken()),
        (False, False, OAuth2Identity(token="tok", user_id="user-1", role="internal_user", team_id=7)),
    ],
)
async def test_introspection_trusts_an_inactive_answer_only_from_a_token_info_endpoint(
    active: bool, requires_active: bool, expected: OAuth2Identity | InactiveToken
) -> None:
    host: Final = _introspection(active)
    settings: Final = OAuth2IntrospectionSettings(endpoint_configured=True, requires_active=requires_active)
    assert await CORE.oauth2().introspect(host, "tok", settings) == expected
    assert host.calls == ["tok"]


async def test_introspection_without_an_endpoint_never_calls_out() -> None:
    host: Final = _introspection(True)
    settings: Final = OAuth2IntrospectionSettings(endpoint_configured=False, requires_active=True)
    assert await CORE.oauth2().introspect(host, "tok", settings) == IntrospectionNotConfigured()
    assert host.calls == []


def _proxy_settings(mappings: Mapping[str, str], *, trusted: bool = True) -> OAuth2ProxySettings:
    return OAuth2ProxySettings(
        trusted_networks=TRUSTED if trusted else (), header_mappings=MappingProxyType(dict(mappings))
    )


def test_proxy_headers_assert_only_the_mapped_identity_fields_that_are_present() -> None:
    host: Final = RecordingProxy("10.1.2.3", {"x-user": "user-1", "x-team": "", "x-models": "gpt-a, gpt-b"})
    settings: Final = _proxy_settings({"user_id": "x-user", "team_id": "x-team", "models": "x-models"})
    assert CORE.oauth2().proxy(host, settings) == OAuth2ProxyIdentity(
        user_id="user-1", user_email=None, team_id=None, team_alias=None, org_id=None, models=("gpt-a", "gpt-b")
    )
    assert host.calls[0] == "client_ip"


@pytest.mark.parametrize("ip", ["10.1.2.3", "::ffff:10.1.2.3"])
def test_proxy_headers_trust_a_peer_inside_the_ranges_including_ipv4_mapped_addresses(ip: str) -> None:
    host: Final = RecordingProxy(ip, {"x-user": "user-1"})
    identity: Final = CORE.oauth2().proxy(host, _proxy_settings({"user_id": "x-user"}))
    assert isinstance(identity, OAuth2ProxyIdentity)
    assert identity.user_id == "user-1"


@pytest.mark.parametrize("ip", ["192.168.1.1", None, "not-an-ip"])
def test_proxy_headers_from_an_untrusted_peer_are_never_read(ip: str | None) -> None:
    host: Final = RecordingProxy(ip, {"x-user": "admin"})
    assert CORE.oauth2().proxy(host, _proxy_settings({"user_id": "x-user"})) == UntrustedProxy(client_ip=ip)
    assert host.calls == ["client_ip"]


def test_proxy_headers_need_trusted_ranges_before_anything_else() -> None:
    host: Final = RecordingProxy("10.1.2.3", {})
    assert CORE.oauth2().proxy(host, _proxy_settings({"user_id": "x"}, trusted=False)) == TrustedProxyRangesMissing()
    assert host.calls == []


def test_proxy_headers_need_mappings() -> None:
    assert CORE.oauth2().proxy(RecordingProxy("10.1.2.3", {}), _proxy_settings({})) == ProxyMappingsMissing()


def test_proxy_headers_refuse_every_non_identity_mapping_before_reading_headers() -> None:
    host: Final = RecordingProxy("10.1.2.3", {"x-role": "proxy_admin"})
    settings: Final = _proxy_settings({"user_role": "x-role", "user_id": "x-user", "max_budget": "x-budget"})
    assert CORE.oauth2().proxy(host, settings) == DisallowedProxyMappings(fields=("max_budget", "user_role"))
    assert host.calls == ["client_ip"]
