from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration._support.prometheus_series import Sample, scrape, series_cap_config

_DEPRECATED: Final = "deprecated-label-past"
_CURRENT: Final = "deprecated-label-future"
_PROVIDER_MODEL: Final = "litellm-deprecated-label-synthetic"


@pytest.fixture(scope="module")
def deprecated_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("prometheus-deprecated-label")
    with gateway_from_environment() as shared:
        config: Final = series_cap_config(
            directory,
            {"prometheus_emit_deprecated_label": True},
            model_list=tuple(
                {
                    "model_name": group,
                    "litellm_params": {
                        "model": f"openai/{_PROVIDER_MODEL}",
                        "api_key": "synthetic-provider-key",
                        "api_base": f"{shared.upstream_url}/v1",
                        "input_cost_per_token": 0,
                        "output_cost_per_token": 0,
                    },
                    "model_info": {"deprecation_date": deprecation_date},
                }
                for group, deprecation_date in ((_DEPRECATED, "2000-01-01"), (_CURRENT, "2999-01-01"))
            ),
        )
        with owned_proxy(
            shared, directory, {}, config=config, remove_environment=("PROMETHEUS_MULTIPROC_DIR",)
        ) as proxy:
            yield proxy


def _series(gateway: Gateway, metric: str, group: str) -> tuple[Sample, ...]:
    return eventually(
        lambda: tuple(
            sample
            for sample in scrape(gateway)
            if sample.name == metric and sample.labels.get("requested_model") == group and sample.value > 0
        ),
        bool,
        seconds=30,
    )


@pytest.mark.parametrize("group, expected", ((_DEPRECATED, "true"), (_CURRENT, "false")))
@pytest.mark.parametrize("metric", ("litellm_proxy_total_requests_metric_total", "litellm_requests_metric_total"))
def test_success_counters_label_the_deployments_deprecation(
    deprecated_gateway: Gateway, group: str, expected: str, metric: str
) -> None:
    response: Final = deprecated_gateway.request(
        "POST", "/v1/chat/completions", {"model": group, "messages": [{"role": "user", "content": "label control"}]}
    )
    assert response.status_code == 200, response.text
    samples: Final = _series(deprecated_gateway, metric, group)
    assert all("deprecated" in sample.labels for sample in samples), f"deprecated label missing: {samples!r}"
    assert {sample.labels["deprecated"] for sample in samples} == {expected}


def test_upstream_error_labels_the_deprecated_deployment(deprecated_gateway: Gateway) -> None:
    with httpx.Client(base_url=deprecated_gateway.upstream_url, trust_env=False) as upstream:
        configured: Final = upstream.post(f"/__scripts/{_PROVIDER_MODEL}", json={"statuses": [400]})
        assert configured.status_code == 200, configured.text
        try:
            response: Final = deprecated_gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": _DEPRECATED, "messages": [{"role": "user", "content": "failure control"}]},
            )
            assert response.status_code == 400, response.text
            assert "Controlled provider failure" in response.text
            samples: Final = _series(deprecated_gateway, "litellm_proxy_failed_requests_metric_total", _DEPRECATED)
            assert all("deprecated" in sample.labels for sample in samples), f"deprecated label missing: {samples!r}"
            assert {sample.labels["deprecated"] for sample in samples} == {"true"}
        finally:
            removed: Final = upstream.delete(f"/__scripts/{_PROVIDER_MODEL}")
            assert removed.status_code == 200, removed.text
