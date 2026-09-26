"""Tests for the Nadir classifier plugin (litellm/router_strategy/complexity_router/nadir_classifier.py)."""

import copy

import pytest

from litellm.router_strategy.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.nadir_classifier import (
    NadirComplexityClassifier,
    _bucket_url,
    nadir_classifier,
)
from litellm.types.router import ClassifierPlugin, RoutingContext

HOSTED_BUCKET_URL = "https://api.getnadir.com/v1/bucket"


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
        assert _bucket_url(api_base) == HOSTED_BUCKET_URL

    def test_self_hosted_host_with_a_path_prefix_is_preserved(self):
        assert _bucket_url("https://gateway.internal/nadir") == "https://gateway.internal/nadir/v1/bucket"


class TestClassify:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("bucket", "expected_tier"), [("simple", "SIMPLE"), ("medium", "MEDIUM"), ("complex", "COMPLEX")]
    )
    async def test_each_bucket_maps_to_its_default_tier(self, bucket, expected_tier):
        client = _FakeClient(payload={"bucket": bucket, "confidence": 0.91})
        assert await NadirComplexityClassifier(client=client).classify(_context()) == expected_tier
        assert client.calls[0]["url"] == HOSTED_BUCKET_URL
        assert client.calls[0]["json"]["messages"] == [{"role": "user", "content": "refactor the retry loop"}]
        assert client.calls[0]["json"]["source"] == "litellm"

    @pytest.mark.asyncio
    async def test_tier_map_override_serves_custom_tier_names(self):
        """A tier_definitions router names its own tiers; tier_map must answer in those names."""
        classifier = NadirComplexityClassifier(
            tier_map={"simple": "cheap", "medium": "mid", "complex": "deep"},
            client=_FakeClient(payload={"bucket": "complex"}),
        )
        assert await classifier.classify(_context()) == "deep"

    @pytest.mark.asyncio
    async def test_api_key_is_sent_when_configured(self):
        client = _FakeClient(payload={"bucket": "simple"})
        await NadirComplexityClassifier(api_key="ndr_test", client=client).classify(_context())
        assert client.calls[0]["headers"] == {"X-API-Key": "ndr_test"}

    @pytest.mark.asyncio
    async def test_env_supplies_key_and_base(self, monkeypatch):
        client = _FakeClient(payload={"bucket": "simple"})
        monkeypatch.setenv("NADIR_API_KEY", "ndr_from_env")
        monkeypatch.setenv("NADIR_API_BASE", "https://nadir.internal")
        await NadirComplexityClassifier(client=client).classify(_context())
        assert client.calls[0]["headers"] == {"X-API-Key": "ndr_from_env"}
        assert client.calls[0]["url"] == "https://nadir.internal/v1/bucket"

    @pytest.mark.asyncio
    async def test_env_key_never_follows_a_base_set_in_code(self, monkeypatch):
        """The environment names where NADIR_API_KEY may go; an api_base set in code brings its own key."""
        client = _FakeClient(payload={"bucket": "simple"})
        monkeypatch.setenv("NADIR_API_KEY", "ndr_from_env")
        monkeypatch.delenv("NADIR_API_BASE", raising=False)
        await NadirComplexityClassifier(api_base="https://elsewhere.example", client=client).classify(_context())
        assert client.calls[0]["url"] == "https://elsewhere.example/v1/bucket"
        assert client.calls[0]["headers"] == {}

    @pytest.mark.asyncio
    async def test_env_key_goes_to_the_hosted_api_named_in_code(self, monkeypatch):
        """Naming the hosted base in code, with or without /v1, is still the base the environment trusts."""
        client = _FakeClient(payload={"bucket": "simple"})
        monkeypatch.setenv("NADIR_API_KEY", "ndr_from_env")
        monkeypatch.delenv("NADIR_API_BASE", raising=False)
        await NadirComplexityClassifier(api_base="https://api.getnadir.com/v1", client=client).classify(_context())
        assert client.calls[0]["headers"] == {"X-API-Key": "ndr_from_env"}

    @pytest.mark.asyncio
    async def test_a_base_set_in_code_sends_its_own_key(self, monkeypatch):
        client = _FakeClient(payload={"bucket": "simple"})
        monkeypatch.setenv("NADIR_API_KEY", "ndr_from_env")
        classifier = NadirComplexityClassifier(
            api_base="https://nadir.onprem.internal", api_key="ndr_own", client=client
        )
        await classifier.classify(_context())
        assert client.calls[0]["headers"] == {"X-API-Key": "ndr_own"}

    @pytest.mark.asyncio
    async def test_anonymous_when_no_key_is_configured(self, monkeypatch):
        client = _FakeClient(payload={"bucket": "simple"})
        monkeypatch.delenv("NADIR_API_KEY", raising=False)
        await NadirComplexityClassifier(client=client).classify(_context())
        assert client.calls[0]["headers"] == {}

    @pytest.mark.asyncio
    async def test_no_messages_declines_without_a_call(self):
        """An empty request is a 400 from the endpoint; decline locally instead of spending the trip."""
        client = _FakeClient(payload={"bucket": "simple"})
        assert await NadirComplexityClassifier(client=client).classify(_context(messages=[])) is None
        assert client.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload", [{"bucket": "reasoning"}, {"bucket": None}, {"bucket": 3}, {}, ["not", "a", "mapping"]]
    )
    async def test_unusable_verdicts_decline(self, payload):
        """Anything outside tier_map is declined, so classifier_fallback decides rather than this guessing."""
        assert await NadirComplexityClassifier(client=_FakeClient(payload=payload)).classify(_context()) is None

    @pytest.mark.asyncio
    async def test_bucket_name_is_matched_case_insensitively(self):
        classifier = NadirComplexityClassifier(client=_FakeClient(payload={"bucket": " Complex "}))
        assert await classifier.classify(_context()) == "COMPLEX"

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

    def _router(self, client, **overrides):
        return ComplexityRouter(
            model_name="test-nadir-router",
            litellm_router_instance=None,
            complexity_router_config={
                "tiers": {"SIMPLE": "gpt-4o-mini", "MEDIUM": "gpt-4o", "COMPLEX": "o1-preview"},
                "classifier_type": "custom",
                "classifier_plugin": NadirComplexityClassifier(client=client),
                **overrides,
            },
        )

    @pytest.mark.asyncio
    async def test_verdict_becomes_the_routed_tier(self):
        router = self._router(_FakeClient(payload={"bucket": "complex"}))
        outcome = await router.aclassify(
            prompt="port the scheduler to the new executor",
            raw_messages=[{"role": "user", "content": "port the scheduler to the new executor"}],
        )
        assert outcome.tier.value == "COMPLEX"
        assert outcome.cause == "classifier_plugin"

    @pytest.mark.asyncio
    async def test_renamed_tier_labels_need_no_tier_map(self):
        """tier_labels only renames what the dashboard shows; the router still accepts the default names."""
        router = self._router(
            _FakeClient(payload={"bucket": "complex"}),
            tier_labels={"SIMPLE": "Cheap", "MEDIUM": "Standard", "COMPLEX": "Premium"},
        )
        outcome = await router.aclassify(prompt="hi", raw_messages=[{"role": "user", "content": "hi"}])
        assert outcome.tier.value == "COMPLEX"
        assert outcome.cause == "classifier_plugin"

    @pytest.mark.asyncio
    async def test_network_failure_falls_back_to_the_local_scorer(self):
        """A Nadir outage must never fail a completion: the heuristic scorer still places the request."""
        router = self._router(_FakeClient(error=ConnectionError("nadir unreachable")))
        outcome = await router.aclassify(prompt="hi", raw_messages=[{"role": "user", "content": "hi"}])
        assert outcome.cause != "classifier_plugin"
        assert outcome.tier is not None
