"""Unit tests for SAP deployment URL selection and resolution."""

from typing import Final
from unittest.mock import MagicMock, patch

import pytest

import litellm.llms.sap.deployment as _depl
from litellm.llms.sap.chat.handler import GenAIHubOrchestrationError
from litellm.llms.sap.deployment import select_deployment_url, AICORE_DEPLOYMENT_URL_ENV_VAR

def _resources(*names: str) -> list[dict[str, str]]:
    return [
        {
            "scenarioId": "orchestration",
            "configurationId": f"cfg-{n}",
            "deploymentUrl": f"https://deploy-{n}.sap.com",
            "createdAt": f"2024-01-{i + 1:02d}T00:00:00Z",
            "configurationName": n,
        }
        for i, n in enumerate(names)
    ]


@pytest.fixture(autouse=True)
def _clear_deployment_cache():
    _depl._deployment_url_cache.clear()
    yield
    _depl._deployment_url_cache.clear()


class TestSelectDeploymentUrl:
    def test_single_deployment_returns_url(self):
        url: Final = select_deployment_url(_resources("orch-a"))
        assert url == "https://deploy-orch-a.sap.com"

    def test_multiple_deployments_picks_newest(self):
        url: Final = select_deployment_url(_resources("older", "newer"))
        assert url == "https://deploy-newer.sap.com"

    def test_no_deployments_raises(self):
        with pytest.raises(GenAIHubOrchestrationError) as exc:
            select_deployment_url([])
        assert "No orchestration deployment found" in str(exc.value)

    def test_skips_entries_without_deployment_url(self):
        resources: Final = [
            {"scenarioId": "orchestration", "configurationId": "a", "createdAt": "2024-01-02T00:00:00Z", "configurationName": "missing"},
            *_resources("valid"),
        ]
        url: Final = select_deployment_url(resources)
        assert url == "https://deploy-valid.sap.com"

    def test_multiple_deployments_emits_warning(self):
        with patch("litellm.llms.sap.deployment.verbose_logger") as mock_log:
            url: Final = select_deployment_url(_resources("older", "newer"))
        mock_log.warning.assert_called_once()
        assert mock_log.warning.call_args[0][1] == 2
        assert url == "https://deploy-newer.sap.com"


class TestResolveDeploymentUrl:
    def test_env_var_skips_fetcher(self):
        fetcher: Final = MagicMock()
        with patch.dict("os.environ", {AICORE_DEPLOYMENT_URL_ENV_VAR: "https://env.sap.com"}):
            url: Final = _depl.resolve_deployment_url("https://api.example", "default", fetcher)
        assert url == "https://env.sap.com"
        fetcher.assert_not_called()

    def test_fetcher_called_when_no_env_var(self):
        fetcher: Final = MagicMock(return_value="https://discovered.sap.com")
        with patch.dict("os.environ", {AICORE_DEPLOYMENT_URL_ENV_VAR: ""}):
            url: Final = _depl.resolve_deployment_url("https://api.example", "default", fetcher)
        assert url == "https://discovered.sap.com"
        fetcher.assert_called_once()

    def test_second_call_uses_cache(self):
        fetcher: Final = MagicMock(return_value="https://discovered.sap.com")
        with patch.dict("os.environ", {AICORE_DEPLOYMENT_URL_ENV_VAR: ""}):
            first: Final = _depl.resolve_deployment_url("https://api.example", "default", fetcher)
            second: Final = _depl.resolve_deployment_url("https://api.example", "default", fetcher)
        assert first == second == "https://discovered.sap.com"
        fetcher.assert_called_once()

    def test_different_keys_fetch_independently(self):
        call_count: list[int] = [0]  # mutable-ok: counter for test assertion

        def fetcher() -> str:
            call_count[0] += 1
            return f"https://deploy-{call_count[0]}.sap.com"

        with patch.dict("os.environ", {AICORE_DEPLOYMENT_URL_ENV_VAR: ""}):
            url_a: Final = _depl.resolve_deployment_url("https://api.example", "rg-a", fetcher)
            url_b: Final = _depl.resolve_deployment_url("https://api.example", "rg-b", fetcher)
        assert url_a != url_b
        assert call_count[0] == 2
