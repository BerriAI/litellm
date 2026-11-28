"""Azure OpenAI realtime over TLS: the upstream websocket URL, query and credential the proxy opens.

`/v1/realtime?model=<group>` on an `azure/<deployment>` model is served by `AzureOpenAIRealtime.async_realtime`
(`litellm/llms/azure/realtime/handler.py`). It always passes an SSL context to `websockets.connect`, so these
tests run an owned proxy that trusts the CA of a TLS scripted upstream (`owned_upstream(..., tls=True)`), the
same rig `test_realtime_unguarded_voice_turns.py` uses. The upstream records each websocket upgrade.

The contract, read from the handler and `litellm/realtime_api/main.py`:

- Protocol choice (`azure_realtime_protocol_for_client`): a deployment `realtime_protocol` wins; otherwise a
  translation or `intent=transcription` session is GA; otherwise a model whose cost map row says
  `realtime_ga_only` is GA; otherwise the client's `OpenAI-Beta: realtime=v1` header picks beta and its
  absence picks GA.
- GA URL: `wss://<base>/openai/v1/realtime?model=<deployment>` with no `api-version`.
  Beta URL: `wss://<base>/openai/realtime?api-version=<v>&deployment=<deployment>`, where `<v>` defaults to
  `2024-10-01-preview` when the deployment has none.
- Credentials: a key goes in the `api-key` header only; without a key an Entra token goes in
  `Authorization: Bearer` only; never both (`get_auth_headers`).
- An upstream handshake refusal reaches the client as an `error` event "Upstream realtime handshake rejected
  with HTTP <status>" and close 1008 for 401/403 (`realtime_errors.py`).

The Rust front door will take over the websocket upgrade; the URL and header choices are what Azure checks.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import gateway_from_environment
from integration._support.process import (
    OwnedProxy,
    UpstreamSlot,
    graceful_stop_seconds,
    owned_proxy_process,
    owned_upstream,
)
from integration.providers.dec1_azure3_support import (
    REALTIME_DEFAULT_BETA_API_VERSION,
    open_realtime,
    realtime_scenario,
    upgrades_for,
    ws_base,
)
from integration.providers.dec1_azure_support import register_model

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 120)]

UNMAPPED_DEPLOYMENT: Final = "dec1-voice-deployment"
"""Not in the cost map under azure/ or openai/, so `azure_realtime_requires_ga` is False for it."""
GA_ONLY_BASE_MODEL: Final = "azure/gpt-realtime-2.1"
"""Cost map row with `provider_specific_entry.realtime_ga_only: 1` (model_prices_and_context_window.json)."""
BETA_HEADERS: Final = {"OpenAI-Beta": "realtime=v1"}
DATED_PREVIEW: Final = "2025-04-01-preview"


def _write_config(directory: Path, ca_bundle: Path) -> Path:
    config: Final = directory / f"dec1_azure3_realtime_{uuid.uuid4().hex[:8]}.yaml"
    config.write_text(
        json.dumps(
            {
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                },
                "litellm_settings": {"ssl_verify": str(ca_bundle)},
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return config


@pytest.fixture(scope="module")
def tls_slot(tmp_path_factory: pytest.TempPathFactory) -> Iterator[UpstreamSlot]:
    with owned_upstream(tmp_path_factory.mktemp("dec1-azure3-tls-upstream"), tls=True) as slot:
        yield slot


@pytest.fixture(scope="module")
def azure_proxy(tmp_path_factory: pytest.TempPathFactory, tls_slot: UpstreamSlot) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("dec1-azure3-realtime-proxy")
    assert tls_slot.certificate is not None
    ca_bundle: Final = tls_slot.certificate.certificate
    with (
        gateway_from_environment() as rig,
        owned_proxy_process(
            rig,
            directory,
            {"DATABASE_URL": os.environ["DATABASE_URL"], "SSL_VERIFY": str(ca_bundle)},
            config=_write_config(directory, ca_bundle),
            remove_environment=("AZURE_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_AD_TOKEN", "AZURE_API_VERSION"),
            workers=1,
        ) as owned,
    ):
        yield owned


def _proxy_url(owned: OwnedProxy) -> str:
    return str(owned.gateway.client.base_url).rstrip("/")


def test_a_ga_session_opens_openai_v1_realtime_with_the_deployment_as_model(
    azure_proxy: OwnedProxy, tls_slot: UpstreamSlot
) -> None:
    """An unmapped `azure/dec1-voice-deployment` with a key and `api_version: 2025-04-01-preview`; the client
    sends no `OpenAI-Beta` header.

    The upstream must see exactly one upgrade to `/openai/v1/realtime` whose query is exactly
    `[("model", "dec1-voice-deployment")]`: no `api-version` and no `deployment` on the GA route, and the
    client gets `session.created`.
    """
    with azure_proxy.gateway.scenario() as scenario:
        handle: Final = realtime_scenario(scenario, tls_slot.url)
        group: Final = register_model(
            scenario,
            model=f"azure/{UNMAPPED_DEPLOYMENT}",
            api_base=ws_base(tls_slot.url),
            api_key=handle.scenario_id,
            api_version=DATED_PREVIEW,
        )
        seen: Final = open_realtime(_proxy_url(azure_proxy), scenario.key(), f"model={group}")
        assert seen.first_event is not None and seen.first_event["type"] == "session.created", seen
        upgrades: Final = upgrades_for(tls_slot.url, handle.scenario_id)
        assert [(u.path, u.query) for u in upgrades] == [("/openai/v1/realtime", (("model", UNMAPPED_DEPLOYMENT),))]


def test_the_ga_handshake_sends_the_key_as_api_key_and_no_bearer(
    azure_proxy: OwnedProxy, tls_slot: UpstreamSlot
) -> None:
    """Same GA session with a deployment key.

    The upgrade must carry the key in the `api-key` header and an empty `Authorization` header
    (`get_auth_headers` returns only `{"api-key": key}`), so the caller's virtual key never reaches Azure.
    """
    with azure_proxy.gateway.scenario() as scenario:
        handle: Final = realtime_scenario(scenario, tls_slot.url)
        group: Final = register_model(
            scenario,
            model=f"azure/{UNMAPPED_DEPLOYMENT}",
            api_base=ws_base(tls_slot.url),
            api_key=handle.scenario_id,
            api_version=DATED_PREVIEW,
        )
        open_realtime(_proxy_url(azure_proxy), scenario.key(), f"model={group}")
        (upgrade,) = upgrades_for(tls_slot.url, handle.scenario_id)
        assert (upgrade.api_key, upgrade.authorization) == (handle.scenario_id, "")


def test_a_beta_client_header_opens_the_classic_route_with_api_version_and_deployment(
    azure_proxy: OwnedProxy, tls_slot: UpstreamSlot
) -> None:
    """The client sends `OpenAI-Beta: realtime=v1` to an unmapped deployment with `api_version: 2025-04-01-preview`.

    The beta protocol is chosen, so the upstream must see `/openai/realtime` with the query exactly
    `[("api-version", "2025-04-01-preview"), ("deployment", "dec1-voice-deployment")]`, in that order
    (`_construct_url`).
    """
    with azure_proxy.gateway.scenario() as scenario:
        handle: Final = realtime_scenario(scenario, tls_slot.url)
        group: Final = register_model(
            scenario,
            model=f"azure/{UNMAPPED_DEPLOYMENT}",
            api_base=ws_base(tls_slot.url),
            api_key=handle.scenario_id,
            api_version=DATED_PREVIEW,
        )
        open_realtime(_proxy_url(azure_proxy), scenario.key(), f"model={group}", headers=BETA_HEADERS)
        assert [(u.path, u.query) for u in upgrades_for(tls_slot.url, handle.scenario_id)] == [
            ("/openai/realtime", (("api-version", DATED_PREVIEW), ("deployment", UNMAPPED_DEPLOYMENT)))
        ]


def test_a_beta_session_without_a_configured_version_uses_2024_10_01_preview(
    azure_proxy: OwnedProxy, tls_slot: UpstreamSlot
) -> None:
    """A beta client on a deployment with no `api_version` (and `AZURE_API_VERSION` removed from the proxy env).

    The dispatcher falls back to `2024-10-01-preview` (`realtime_api/main.py`), so the upstream query must be
    `[("api-version", "2024-10-01-preview"), ("deployment", ...)]`. This differs from the chat default
    (`2025-02-01-preview`), which a unified Rust default would silently change.
    """
    with azure_proxy.gateway.scenario() as scenario:
        handle: Final = realtime_scenario(scenario, tls_slot.url)
        group: Final = register_model(
            scenario, model=f"azure/{UNMAPPED_DEPLOYMENT}", api_base=ws_base(tls_slot.url), api_key=handle.scenario_id
        )
        open_realtime(_proxy_url(azure_proxy), scenario.key(), f"model={group}", headers=BETA_HEADERS)
        (upgrade,) = upgrades_for(tls_slot.url, handle.scenario_id)
        assert upgrade.query == (
            ("api-version", REALTIME_DEFAULT_BETA_API_VERSION),
            ("deployment", UNMAPPED_DEPLOYMENT),
        )


def test_a_deployment_realtime_protocol_beta_wins_over_a_ga_client(
    azure_proxy: OwnedProxy, tls_slot: UpstreamSlot
) -> None:
    """The deployment sets `realtime_protocol: beta`; the client sends no `OpenAI-Beta` header.

    A configured protocol wins (`azure_realtime_protocol_for_client`), so the upstream must see the classic
    `/openai/realtime` route with `api-version` and `deployment`, not `/openai/v1/realtime`.
    """
    with azure_proxy.gateway.scenario() as scenario:
        handle: Final = realtime_scenario(scenario, tls_slot.url)
        group: Final = register_model(
            scenario,
            model=f"azure/{UNMAPPED_DEPLOYMENT}",
            api_base=ws_base(tls_slot.url),
            api_key=handle.scenario_id,
            api_version=DATED_PREVIEW,
            realtime_protocol="beta",
        )
        open_realtime(_proxy_url(azure_proxy), scenario.key(), f"model={group}")
        assert [u.path for u in upgrades_for(tls_slot.url, handle.scenario_id)] == ["/openai/realtime"]


@pytest.mark.parametrize("protocol", ["GA", "v1"])
def test_a_deployment_realtime_protocol_ga_wins_over_a_beta_client(
    azure_proxy: OwnedProxy, tls_slot: UpstreamSlot, protocol: str
) -> None:
    """The deployment sets `realtime_protocol` to `GA` or `v1` (case-insensitive); the client sends the beta header.

    The configured GA protocol wins and the proxy bridges the beta client to GA, so the upstream must see
    `/openai/v1/realtime?model=dec1-voice-deployment` and the client still gets its first event.
    """
    with azure_proxy.gateway.scenario() as scenario:
        handle: Final = realtime_scenario(scenario, tls_slot.url)
        group: Final = register_model(
            scenario,
            model=f"azure/{UNMAPPED_DEPLOYMENT}",
            api_base=ws_base(tls_slot.url),
            api_key=handle.scenario_id,
            api_version=DATED_PREVIEW,
            realtime_protocol=protocol,
        )
        seen: Final = open_realtime(_proxy_url(azure_proxy), scenario.key(), f"model={group}", headers=BETA_HEADERS)
        assert seen.first_event is not None, seen
        assert [(u.path, u.query) for u in upgrades_for(tls_slot.url, handle.scenario_id)] == [
            ("/openai/v1/realtime", (("model", UNMAPPED_DEPLOYMENT),))
        ]


def test_a_ga_only_base_model_forces_ga_for_a_beta_client(azure_proxy: OwnedProxy, tls_slot: UpstreamSlot) -> None:
    """A custom deployment name with `base_model: azure/gpt-realtime-2.1` (a `realtime_ga_only` row); the client
    sends the beta header.

    `azure_realtime_requires_ga(base_model)` is True, so the upstream must see `/openai/v1/realtime` and never
    the classic route, even though the client asked for beta. Read from `_deployment_base_model` in
    `realtime_api/main.py`, which reads `litellm_params.base_model`.
    """
    with azure_proxy.gateway.scenario() as scenario:
        handle: Final = realtime_scenario(scenario, tls_slot.url)
        group: Final = register_model(
            scenario,
            model=f"azure/{UNMAPPED_DEPLOYMENT}",
            api_base=ws_base(tls_slot.url),
            api_key=handle.scenario_id,
            api_version=DATED_PREVIEW,
            base_model=GA_ONLY_BASE_MODEL,
        )
        open_realtime(_proxy_url(azure_proxy), scenario.key(), f"model={group}", headers=BETA_HEADERS)
        assert [u.path for u in upgrades_for(tls_slot.url, handle.scenario_id)] == ["/openai/v1/realtime"]


def test_an_entra_only_deployment_sends_a_bearer_and_no_api_key(
    azure_proxy: OwnedProxy, tls_slot: UpstreamSlot
) -> None:
    """The deployment has no key and a static `azure_ad_token` (the scenario id); the proxy env has no Azure key.

    `get_azure_ad_token` returns the static token, so the upgrade must carry `Authorization: Bearer <token>`
    and an empty `api-key` header (never both). UNVERIFIED: that `kwargs["azure_ad_token"]` reaches the
    dispatcher from `litellm_params` on the realtime route.
    """
    with azure_proxy.gateway.scenario() as scenario:
        handle: Final = realtime_scenario(scenario, tls_slot.url)
        group: Final = register_model(
            scenario,
            model=f"azure/{UNMAPPED_DEPLOYMENT}",
            api_base=ws_base(tls_slot.url),
            azure_ad_token=handle.scenario_id,
            api_version=DATED_PREVIEW,
        )
        open_realtime(_proxy_url(azure_proxy), scenario.key(), f"model={group}")
        (upgrade,) = upgrades_for(tls_slot.url, handle.scenario_id)
        assert (upgrade.authorization, upgrade.api_key) == (f"Bearer {handle.scenario_id}", "")


def test_a_transcription_intent_opens_ga_with_only_the_intent_query(
    azure_proxy: OwnedProxy, tls_slot: UpstreamSlot
) -> None:
    """The client connects with `?model=<group>&intent=transcription` and the beta header.

    A transcription session is always GA, and the GA query drops `model` when `intent=transcription`
    (`_construct_url`), so the upstream must see `/openai/v1/realtime` with the query exactly
    `[("intent", "transcription")]`.
    """
    with azure_proxy.gateway.scenario() as scenario:
        handle: Final = realtime_scenario(scenario, tls_slot.url)
        group: Final = register_model(
            scenario,
            model=f"azure/{UNMAPPED_DEPLOYMENT}",
            api_base=ws_base(tls_slot.url),
            api_key=handle.scenario_id,
            api_version=DATED_PREVIEW,
        )
        open_realtime(
            _proxy_url(azure_proxy), scenario.key(), f"model={group}&intent=transcription", headers=BETA_HEADERS
        )
        assert [(u.path, u.query) for u in upgrades_for(tls_slot.url, handle.scenario_id)] == [
            ("/openai/v1/realtime", (("intent", "transcription"),))
        ]


def test_an_https_api_base_is_dialled_as_wss(azure_proxy: OwnedProxy, tls_slot: UpstreamSlot) -> None:
    """The deployment `api_base` is the `https://` URL of the TLS upstream, not `wss://`.

    `_construct_url` rewrites `https://` to `wss://`, so the upgrade still reaches `/openai/v1/realtime` on the
    same host and the client gets `session.created`.
    """
    with azure_proxy.gateway.scenario() as scenario:
        handle: Final = realtime_scenario(scenario, tls_slot.url)
        group: Final = register_model(
            scenario,
            model=f"azure/{UNMAPPED_DEPLOYMENT}",
            api_base=tls_slot.url,
            api_key=handle.scenario_id,
            api_version=DATED_PREVIEW,
        )
        seen: Final = open_realtime(_proxy_url(azure_proxy), scenario.key(), f"model={group}")
        assert seen.first_event is not None and seen.first_event["type"] == "session.created", seen
        assert [u.path for u in upgrades_for(tls_slot.url, handle.scenario_id)] == ["/openai/v1/realtime"]


def test_an_upstream_handshake_refusal_is_an_error_event_and_close_1008(
    azure_proxy: OwnedProxy, tls_slot: UpstreamSlot
) -> None:
    """The deployment key names no registered realtime script, so the TLS upstream refuses the upgrade (HTTP 403).

    The client must receive one `error` event whose message is exactly "Upstream realtime handshake rejected
    with HTTP 403" and then close code 1008 (`upstream_handshake_close_code`), not a silent 1011.
    """
    with azure_proxy.gateway.scenario() as scenario:
        unregistered: Final = f"dec1-azure3-unregistered-{uuid.uuid4().hex[:8]}"
        group: Final = register_model(
            scenario,
            model=f"azure/{UNMAPPED_DEPLOYMENT}",
            api_base=ws_base(tls_slot.url),
            api_key=unregistered,
            api_version=DATED_PREVIEW,
        )
        seen: Final = open_realtime(_proxy_url(azure_proxy), scenario.key(), f"model={group}", read_events=2)
        assert seen.first_event is not None and seen.first_event["type"] == "error", seen
        error: Final = seen.first_event["error"]
        assert isinstance(error, dict) and error["message"] == "Upstream realtime handshake rejected with HTTP 403"
        assert seen.close_code == 1008, seen
