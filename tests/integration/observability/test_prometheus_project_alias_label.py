from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration._support.prometheus_series import Sample, scrape
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_MODEL: Final = "prometheus-project-alias-model"
_RPM: Final = 60
_TPM: Final = 10000
_ALLOWED: Final = "litellm_project_model_rate_limit_allowed_metric"
_USED: Final = "litellm_project_model_rate_limit_used_metric"
_TOTAL_REQUESTS: Final = "litellm_proxy_total_requests_metric_total"
_LATENCY_COUNT: Final = "litellm_request_total_latency_metric_count"


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    upstream: Wire


def _respond(request: Request) -> Reply:
    assert json.loads(request.body)["model"] == "gpt-4o-mini"
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "metered"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


def _config(directory: Path, upstream: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        {
            "model_name": _MODEL,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "sk-fixture",
                "api_base": f"{upstream.url}/v1",
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            },
        }
    ]
    configuration["litellm_settings"]["callbacks"] = ["prometheus"]
    configuration["litellm_settings"]["custom_prometheus_metadata_labels"] = ["project_alias"]
    configuration.setdefault("router_settings", {})["num_retries"] = 0
    path: Final = directory / "prometheus-project-alias-label.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("prometheus-project-alias-label")
    samples: Final = directory / "prometheus-multiproc"
    samples.mkdir()
    with (
        wire_server(_respond) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"PROMETHEUS_MULTIPROC_DIR": str(samples)},
            config=_config(directory, upstream),
            workers=2,
        ) as owned,
    ):
        yield _Rig(owned, upstream)


def _ask(rig: _Rig, key: str) -> httpx.Response:
    return rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": _MODEL, "messages": [{"role": "user", "content": f"project alias {uuid.uuid4().hex}"}]},
        key=key,
    )


def _series(samples: Sequence[Sample], name: str, **labels: str) -> tuple[Sample, ...]:
    return tuple(
        sample
        for sample in samples
        if sample.name == name and all(sample.labels.get(label) == value for label, value in labels.items())
    )


def _until(rig: _Rig, name: str, **labels: str) -> tuple[Sample, ...]:
    return eventually(lambda: _series(scrape(rig.gateway), name, **labels), bool, seconds=30)


def _project_gauges(rig: _Rig, project: str) -> tuple[Sample, ...]:
    samples: Final = scrape(rig.gateway)
    return _series(samples, _ALLOWED, project_id=project) + _series(samples, _USED, project_id=project)


def test_a_project_key_with_per_model_limits_exports_its_project_gauges_with_project_alias_as_a_custom_label(
    rig: _Rig,
) -> None:
    rig.upstream.drain()
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team()
        alias: Final = f"integration-project-{uuid.uuid4().hex}"
        project: Final = scenario.project(
            team, project_alias=alias, model_rpm_limit={_MODEL: _RPM}, model_tpm_limit={_MODEL: _TPM}
        )
        key: Final = scenario.key(team_id=team, project_id=project)
        response: Final = _ask(rig, key)
        assert response.status_code == 200, response.text
        headers: Final = response.headers
        assert headers.get("x-ratelimit-model_per_project-limit-requests") == str(_RPM), headers
        assert headers.get("x-ratelimit-model_per_project-remaining-requests") == str(_RPM - 1), headers
        forwarded: Final = rig.upstream.drain()
        assert len(forwarded) == 1, forwarded
        used_tokens: Final = _TPM - int(headers["x-ratelimit-model_per_project-remaining-tokens"])
        gauges: Final = eventually(lambda: _project_gauges(rig, project), lambda found: len(found) == 4, seconds=30)
        assert {sample.identity() for sample in gauges} == {
            (
                ("project_alias", alias),
                ("project_id", project),
                ("rate_limit_type", kind),
                ("requested_model", _MODEL),
            )
            for kind in ("requests", "tokens")
        }, gauges
        assert {(sample.name, sample.labels["rate_limit_type"], int(sample.value)) for sample in gauges} == {
            (_ALLOWED, "requests", _RPM),
            (_USED, "requests", 1),
            (_ALLOWED, "tokens", _TPM),
            (_USED, "tokens", used_tokens),
        }, gauges


def test_a_project_key_with_per_model_limits_still_counts_its_request_and_latency(rig: _Rig) -> None:
    rig.upstream.drain()
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team()
        alias: Final = f"integration-project-{uuid.uuid4().hex}"
        project: Final = scenario.project(
            team, project_alias=alias, model_rpm_limit={_MODEL: _RPM}, model_tpm_limit={_MODEL: _TPM}
        )
        key: Final = scenario.key(team_id=team, project_id=project)
        response: Final = _ask(rig, key)
        assert response.status_code == 200, response.text
        forwarded: Final = rig.upstream.drain()
        assert len(forwarded) == 1, forwarded
        hashed: Final = sha256(key.encode()).hexdigest()
        totals: Final = _until(rig, _TOTAL_REQUESTS, hashed_api_key=hashed, status_code="200")
        assert [(sample.labels["project_alias"], sample.value) for sample in totals] == [(alias, 1.0)], totals
        latency: Final = _until(rig, _LATENCY_COUNT, hashed_api_key=hashed)
        assert [sample.value for sample in latency] == [1.0], latency
