"""Tests for the Nadir classifier plugin (litellm/router_strategy/complexity_router/nadir_classifier.py)."""

import copy

import pytest

from litellm.router_strategy.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.nadir_classifier import (
    DEFAULT_NADIR_API_BASE,
    NadirComplexityClassifier,
    _bucket_url,
    nadir_classifier,
)
from litellm.types.router import ClassifierPlugin, RoutingContext


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    """Records the one request the classifier makes, and answers with a canned bucket."""

    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error
        self.calls = []

    async def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._payload)


@pytest.fixture
def fake_client(monkeypatch):
    def _install(payload=None, error=None):
        client = _FakeClient(payload=payload, error=error)
        monkeypatch.setattr(
            "litellm.router_strategy.complexity_router.nadir_classifier.get_async_httpx_client",
            lambda llm_provider: client,
        )
        return client

    return _install


def _context(messages=None):
    resolved = [{"role": "user", "content": "refactor the retry loop"}] if messages is None else messages
    return RoutingContext(
        raw_messages=resolved,
        structured_messages=resolved,
        candidate_models=["gpt-4o-mini", "gpt-4o"],
    )


class TestBucketURL:
    """The endpoint URL, including the documented double-/v1 footgun."""

    @pytest.mark.parametrize(
        "api_base",
        [
            "https://api.getnadir.com",
            "https://api.getnadir.com/",
            "https://api.getnadir.com/v1",
            "https://api.getnadir.com/v1/",
        ],
    )
    def test_v1_is_never_doubled(self, api_base):
        """Nadir's docs advertise the base URL with /v1, so operators paste it with and without."""
        assert _bucket_url(api_base) == "https://api.getnadir.com/v1/bucket"

    def test_self_hosted_host_with_a_path_prefix_is_preserved(self):
        assert _bucket_url("https://gateway.internal/nadir") == "https://gateway.internal/nadir/v1/bucket"


class TestClassify:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("bucket", "expected_tier"), [("simple", "SIMPLE"), ("medium", "MEDIUM"), ("complex", "COMPLEX")]
    )
    async def test_each_bucket_maps_to_its_default_tier(self, fake_client, bucket, expected_tier):
        client = fake_client(payload={"bucket": bucket, "confidence": 0.91})
        assert await NadirComplexityClassifier().classify(_context()) == expected_tier
        assert client.calls[0]["url"] == f"{DEFAULT_NADIR_API_BASE}/v1/bucket"
        assert client.calls[0]["json"]["messages"] == [{"role": "user", "content": "refactor the retry loop"}]
        assert client.calls[0]["json"]["source"] == "litellm"

    @pytest.mark.asyncio
    async def test_tier_map_override_serves_renamed_tiers(self, fake_client):
        """tier_labels/tier_definitions rename the tiers; the plugin must answer in the router's names."""
        fake_client(payload={"bucket": "complex"})
        classifier = NadirComplexityClassifier(tier_map={"simple": "cheap", "medium": "mid", "complex": "deep"})
        assert await classifier.classify(_context()) == "deep"

    @pytest.mark.asyncio
    async def test_api_key_is_sent_when_configured(self, fake_client):
        client = fake_client(payload={"bucket": "simple"})
        await NadirComplexityClassifier(api_key="ndr_test").classify(_context())
        assert client.calls[0]["headers"] == {"X-API-Key": "ndr_test"}

    @pytest.mark.asyncio
    async def test_env_supplies_key_and_base(self, fake_client, monkeypatch):
        client = fake_client(payload={"bucket": "simple"})
        monkeypatch.setenv("NADIR_API_KEY", "ndr_from_env")
        monkeypatch.setenv("NADIR_API_BASE", "https://nadir.internal")
        await NadirComplexityClassifier().classify(_context())
        assert client.calls[0]["headers"] == {"X-API-Key": "ndr_from_env"}
        assert client.calls[0]["url"] == "https://nadir.internal/v1/bucket"

    @pytest.mark.asyncio
    async def test_anonymous_when_no_key_is_configured(self, fake_client, monkeypatch):
        client = fake_client(payload={"bucket": "simple"})
        monkeypatch.delenv("NADIR_API_KEY", raising=False)
        await NadirComplexityClassifier().classify(_context())
        assert client.calls[0]["headers"] == {}

    @pytest.mark.asyncio
    async def test_no_messages_declines_without_a_call(self, fake_client):
        """An empty request is a 400 from the endpoint; decline locally instead of spending the trip."""
        client = fake_client(payload={"bucket": "simple"})
        assert await NadirComplexityClassifier().classify(_context(messages=[])) is None
        assert client.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", [{"bucket": "reasoning"}, {"bucket": None}, {}, ["not", "a", "mapping"]])
    async def test_unusable_verdicts_decline(self, fake_client, payload):
        """Anything outside tier_map is declined, so classifier_fallback decides rather than this guessing."""
        fake_client(payload=payload)
        assert await NadirComplexityClassifier().classify(_context()) is None

    @pytest.mark.asyncio
    async def test_bucket_name_is_matched_case_insensitively(self, fake_client):
        fake_client(payload={"bucket": " Complex "})
        assert await NadirComplexityClassifier().classify(_context()) == "COMPLEX"

    def test_module_instance_satisfies_the_plugin_protocol(self):
        """The proxy resolves the dotted path to this instance and interface-checks it at startup."""
        assert isinstance(nadir_classifier, ClassifierPlugin)


class TestDeploymentConfigSurvivesRouterInit:
    """The plugin instance travels inside a deployment's litellm_params, which get deepcopied."""

    @pytest.mark.parametrize("tier_map", [None, {"simple": "Cheap", "medium": "Standard", "complex": "Premium"}])
    def test_classifier_is_deepcopyable(self, tier_map):
        """A MappingProxyType here would raise `cannot pickle 'mappingproxy'` at Router init, i.e. at
        proxy startup, long before any request reaches the classifier."""
        classifier = NadirComplexityClassifier(tier_map=tier_map)
        copied = copy.deepcopy({"complexity_router_config": {"classifier_plugin": classifier}})
        assert isinstance(copied["complexity_router_config"]["classifier_plugin"], NadirComplexityClassifier)


class TestThroughTheComplexityRouter:
    """The plugin as the router actually drives it: verdict in, tier out, failures fall back."""

    def _router(self, mock_router_instance=None, **overrides):
        return ComplexityRouter(
            model_name="test-nadir-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"SIMPLE": "gpt-4o-mini", "MEDIUM": "gpt-4o", "COMPLEX": "o1-preview"},
                "classifier_type": "custom",
                "classifier_plugin": NadirComplexityClassifier(),
                **overrides,
            },
        )

    @pytest.mark.asyncio
    async def test_verdict_becomes_the_routed_tier(self, fake_client):
        fake_client(payload={"bucket": "complex"})
        outcome = await self._router().aclassify(
            prompt="port the scheduler to the new executor",
            raw_messages=[{"role": "user", "content": "port the scheduler to the new executor"}],
        )
        assert outcome.tier.value == "COMPLEX"
        assert outcome.cause == "classifier_plugin"

    @pytest.mark.asyncio
    async def test_network_failure_falls_back_to_the_local_scorer(self, fake_client):
        """A Nadir outage must never fail a completion: the heuristic scorer still places the request."""
        fake_client(error=ConnectionError("nadir unreachable"))
        outcome = await self._router().aclassify(prompt="hi", raw_messages=[{"role": "user", "content": "hi"}])
        assert outcome.cause != "classifier_plugin"
        assert outcome.tier is not None
