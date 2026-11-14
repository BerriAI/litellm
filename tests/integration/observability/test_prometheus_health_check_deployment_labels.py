from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.database import read_rows, scratch_database
from tests.integration._support.process import owned_proxy
from tests.integration._support.prometheus_series import Sample, scrape
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_GROUP: Final = "plain"
_PAYING: Final = "provider-b-pay"
_SERVING: Final = "provider-a-model"
_PAYING_ID: Final = f"paying-{uuid.uuid4().hex[:8]}"
_SERVING_ID: Final = f"serving-{uuid.uuid4().hex[:8]}"
_HEALTH_CHECK_KEY: Final = "litellm-internal-health-check"
_SUCCESSES: Final = "litellm_deployment_success_responses_total"
_FAILURES: Final = "litellm_deployment_failure_responses_total"


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    upstream: Wire
    database_url: str


def _respond(request: Request) -> Reply:
    body: Final = json.loads(request.body)
    if "metadata" in body or "litellm_metadata" in body:
        return Reply(status=418, body=b'{"error": {"message": "proxy metadata leaked upstream"}}')
    if body["model"] == _PAYING:
        return Reply(
            status=402,
            body=json.dumps(
                {"error": {"message": "insufficient credits", "type": "payment_required", "code": "payment_required"}}
            ).encode(),
        )
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": _SERVING,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


def _deployment(model: str, model_id: str, order: int, upstream: Wire) -> dict[str, JsonValue]:
    return {
        "model_name": _GROUP,
        "litellm_params": {
            "model": f"openai/{model}",
            "api_key": "sk-fixture",
            "api_base": f"{upstream.url}/v1",
            "order": order,
        },
        "model_info": {"id": model_id},
    }


def _config(directory: Path, upstream: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        *configuration["model_list"],
        _deployment(_PAYING, _PAYING_ID, 1, upstream),
        _deployment(_SERVING, _SERVING_ID, 2, upstream),
    ]
    configuration["litellm_settings"]["callbacks"] = ["prometheus"]
    configuration["general_settings"] = {
        **configuration["general_settings"],
        "background_health_checks": True,
        "health_check_interval": 1,
        "background_health_check_model_groups": [_GROUP],
    }
    configuration["router_settings"] = {
        **configuration["router_settings"],
        "disable_cooldowns": False,
        "num_retries": 0,
    }
    path: Final = directory / "prometheus-health-check-deployment-labels.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("prometheus-health-check-deployment-labels")
    samples: Final = directory / "prometheus-multiproc"
    samples.mkdir()
    with (
        scratch_database() as database_url,
        wire_server(_respond) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"DATABASE_URL": database_url, "PROMETHEUS_MULTIPROC_DIR": str(samples)},
            config=_config(directory, upstream),
        ) as owned,
    ):
        yield _Rig(owned, upstream, database_url)


def _probes(rig: _Rig, name: str) -> tuple[Sample, ...]:
    return tuple(
        sample
        for sample in scrape(rig.gateway)
        if sample.name == name and sample.labels.get("api_key_alias") == _HEALTH_CHECK_KEY
    )


def _probe_of(rig: _Rig, name: str, model_id: str) -> Sample:
    found: Final = eventually(
        lambda: _probes(rig, name),
        lambda samples: any(sample.labels.get("model_id") == model_id for sample in samples),
        seconds=60,
        return_last_on_timeout=True,
    )
    labeled: Final = tuple(sample for sample in found if sample.labels.get("model_id") == model_id)
    assert labeled, f"no probe series for {model_id}, probe series seen: {[dict(s.labels) for s in found]}"
    return labeled[0]


def test_a_succeeding_probe_counts_under_its_deployment(rig: _Rig) -> None:
    sample: Final = _probe_of(rig, _SUCCESSES, _SERVING_ID)
    assert sample.labels["model_group"] == ""
    assert sample.labels["api_provider"] == "openai"
    assert sample.labels["api_base"] == f"{rig.upstream.url}/v1"


def test_a_failing_probe_counts_under_its_deployment(rig: _Rig) -> None:
    sample: Final = _probe_of(rig, _FAILURES, _PAYING_ID)
    assert sample.labels["exception_status"] == "402"
    assert sample.labels["model_group"] == ""
    assert sample.labels["requested_model"] == _PAYING
    assert sample.labels["api_provider"] == "openai"
    assert sample.labels["api_base"] == f"{rig.upstream.url}/v1"
    assert _PAYING in sample.labels["litellm_model_name"]


def _probe_failures(rig: _Rig) -> float:
    return sum(sample.value for sample in _probes(rig, _FAILURES) if sample.labels["exception_status"] == "402")


def test_the_probe_identity_never_reaches_the_upstream(rig: _Rig) -> None:
    eventually(lambda: _probe_failures(rig), lambda failures: failures >= 1, seconds=60)
    received: Final = eventually(rig.upstream.drain, bool, seconds=30)
    bodies: Final = tuple(json.loads(request.body) for request in received)
    assert {body["model"] for body in bodies} <= {_PAYING, _SERVING}, bodies
    assert all("metadata" not in body and "litellm_metadata" not in body for body in bodies), bodies


def test_failing_probes_do_not_cool_down_the_deployment_for_routed_traffic(rig: _Rig) -> None:
    eventually(lambda: _probe_failures(rig), lambda failures: failures >= 3, seconds=60)
    marker: Final = f"routed {uuid.uuid4().hex}"
    response: Final = rig.gateway.request(
        "POST", "/v1/chat/completions", {"model": _GROUP, "messages": [{"role": "user", "content": marker}]}
    )
    assert response.status_code == 200, response.text
    routed: Final = tuple(
        body["model"]
        for body in (json.loads(request.body) for request in rig.upstream.drain())
        if marker in json.dumps(body["messages"])
    )
    assert routed == (_PAYING, _SERVING), "the order 1 deployment was skipped, so the probes cooled it down"


def test_the_probe_spend_row_carries_its_deployment_and_the_health_check_key(rig: _Rig) -> None:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT model_id, model_group, api_key FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
            (_HEALTH_CHECK_KEY,),
            database_url=rig.database_url,
        ),
        lambda found: any(row["model_id"] == _SERVING_ID for row in found),
        seconds=70,
        return_last_on_timeout=True,
    )
    serving: Final = tuple(row for row in rows if row["model_id"] == _SERVING_ID)
    assert serving, rows
    assert {(row["model_group"], row["api_key"]) for row in serving} == {("", _HEALTH_CHECK_KEY)}
