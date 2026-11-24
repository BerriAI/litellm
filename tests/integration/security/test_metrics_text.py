"""S7: the Prometheus ``/metrics/`` text never carries a credential canary.

Metric label values come from request fields (caller, model, route, user agent, exception
class), so a credential copied into one of them would be served to every scraper. The owned
proxy enables the ``prometheus`` callback, sends one successful and one provider-rejected chat
completion, and searches the whole scrape.

Positive control: the provider double must receive ``Authorization: Bearer <B1 canary>`` for
both requests (their content carries the fresh marker, so neither is served from the response
cache). Sensitivity control: both requests send the marker as their ``User-Agent``,
which the proxy exports as the ``user_agent`` label, so the scrape must carry the marker on
the success and the failure series before the credential search counts.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import eventually
from integration.security._canary import MARKER, Canary, canary, find_canary
from integration.security._sinks import CONFIG_MODEL, PROVIDER_4XX, Rig, canary_rig, team_caller
from integration.security._sweeps import Hit, assert_no_hits

METRICS_ROUTE: Final = "/metrics/"


def enable_prometheus(config: dict[str, object], _provider_url: str) -> None:
    settings: Final = config["litellm_settings"]
    assert isinstance(settings, dict)
    settings["callbacks"] = [*settings["callbacks"], "prometheus"]


def sweep_metrics(text: str, canaries: tuple[Canary, ...]) -> tuple[Hit, ...]:
    """Every canary in the scrape, attributed to the series line that holds it."""
    if not find_canary(text, canaries):
        return ()
    return tuple(
        Hit("S7", f"GET {METRICS_ROUTE} line {number}: {line[:160]!r}", match.slot, match.encoding)
        for number, line in enumerate(text.splitlines(), start=1)
        for match in find_canary(line, canaries)
    )


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    with canary_rig(tmp_path, configure=enable_prometheus) as value:
        yield value


def test_metrics_text_carries_no_credential(rig: Rig) -> None:
    b1: Final = rig.canaries["B1"]
    marker: Final = canary(MARKER)
    agent: Final = f"canary-agent/{marker.value}"
    with rig.proxy.scenario() as scenario:
        caller: Final = team_caller(scenario)
        responses: Final = tuple(
            rig.proxy.request(
                "POST",
                "/v1/chat/completions",
                {"model": CONFIG_MODEL, "messages": [{"role": "user", "content": text}]},
                key=caller.key,
                headers={"User-Agent": agent},
            )
            for text in (f"slot B1 metrics {marker.value}", f"slot B1 metrics {marker.value} {PROVIDER_4XX}")
        )
        assert [response.status_code for response in responses] == [200, 400], [r.text for r in responses]
        delivered: Final = rig.provider.carrying(marker.value)
        assert [request.headers.get("authorization") for request in delivered] == [f"Bearer {b1.value}"] * 2, (
            "Positive control: the provider double never received the B1 canary"
        )

        def scrape() -> str:
            response: Final = rig.proxy.request("GET", METRICS_ROUTE)
            assert response.status_code == 200, response.text
            return response.text

        def both_outcomes_exported(text: str) -> bool:
            lines: Final = text.splitlines()
            return all(
                any(marker.core in line and f'status_code="{status}"' in line for line in lines)
                for status in ("200", "400")
            )

        hits: Final = sweep_metrics(eventually(scrape, both_outcomes_exported, seconds=30), (marker, b1))
    assert any(hit.slot == MARKER for hit in hits), "Sensitivity control: the scrape never carried the marker"
    assert_no_hits(tuple(hit for hit in hits if hit.slot != MARKER), "slot B1, metrics text")
