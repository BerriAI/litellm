from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 120), pytest.mark.shares_worker_state]

ALERT_SECONDS: Final = 40


@dataclass(frozen=True, slots=True)
class AlertingProxy:
    gateway: Gateway
    alerts: Wire


def accept_alert(request: Request) -> Reply:
    return Reply(body=b"ok", content_type="text/plain")


def alerting_config(directory: Path) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "general_settings": {
            **object_value(base["general_settings"]),
            "alerting": ["slack"],
            "alert_types": ["budget_alerts"],
        },
    }
    path: Final = directory / "alerting.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def alerting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[AlertingProxy]:
    directory: Final = tmp_path_factory.mktemp("user-soft-budget-alerts")
    with gateway_from_environment() as shared, wire_server(accept_alert) as alerts:
        with owned_proxy(
            shared,
            directory,
            {"SLACK_WEBHOOK_URL": alerts.url, "DEFAULT_FLUSH_INTERVAL_SECONDS": "1"},
            config=alerting_config(directory),
        ) as gateway:
            yield AlertingProxy(gateway, alerts)


def alerts_naming(alerts: Wire, marker: str, *, seconds: float) -> tuple[str, ...]:
    def matching() -> tuple[str, ...]:
        texts: Final = (string_value(JSON_OBJECT.validate_json(request.body)["text"]) for request in alerts.drain())
        return tuple(text for text in texts if marker in text)

    return eventually(matching, lambda found: len(found) > 0, seconds=seconds)


def user_info(gateway: Gateway, user_id: str) -> dict[str, object]:
    return object_value(gateway.get("/user/info", {"user_id": user_id})["user_info"])


def test_user_soft_budget_round_trips_and_stays_below_max_budget(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(max_budget=100, soft_budget=50)
        assert user_info(gateway, user_id)["soft_budget"] == 50

        gateway.post("/user/update", {"user_id": user_id, "soft_budget": 75})
        assert user_info(gateway, user_id)["soft_budget"] == 75

        rejected: Final = gateway.request("POST", "/user/update", {"user_id": user_id, "soft_budget": 100})
        assert rejected.status_code == 400, rejected.text
        assert user_info(gateway, user_id)["soft_budget"] == 75

        gateway.post("/user/update", {"user_id": user_id, "soft_budget": None})
        assert user_info(gateway, user_id)["soft_budget"] is None


@pytest.mark.parametrize("max_budget", [None, 100.0])
def test_user_over_soft_budget_gets_alert_and_request_still_succeeds(
    alerting: AlertingProxy, max_budget: float | None
) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = scenario.user(soft_budget=0.5, max_budget=max_budget, spend=1.0)
        key: Final = scenario.key(user_id=user_id)

        response: Final = gateway.chat(model, key=key)

        assert response["choices"], response
        (alert, *_) = alerts_naming(alerting.alerts, user_id, seconds=ALERT_SECONDS)
        assert "Soft Budget Crossed" in alert, alert
        assert "soft_budget: 0.5" in alert, alert


def test_user_under_soft_budget_gets_no_alert(alerting: AlertingProxy) -> None:
    gateway: Final = alerting.gateway
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        under_id: Final = scenario.user(soft_budget=50.0, spend=1.0)
        gateway.chat(model, key=scenario.key(user_id=under_id))
        crossing_id: Final = scenario.user(soft_budget=0.5, spend=1.0)
        gateway.chat(model, key=scenario.key(user_id=crossing_id))

        def naming_either() -> tuple[str, ...]:
            texts: Final = (
                string_value(JSON_OBJECT.validate_json(request.body)["text"]) for request in alerting.alerts.drain()
            )
            return tuple(text for text in texts if under_id in text or crossing_id in text)

        seen: Final = eventually(naming_either, lambda found: len(found) > 0, seconds=ALERT_SECONDS)
        assert all(under_id not in text for text in seen), seen
        assert any(crossing_id in text for text in seen), seen
