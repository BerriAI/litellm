import json
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm.interactions.http_handler import InteractionsHTTPHandler
from litellm.interactions.utils import get_provider_interactions_api_config
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.gemini.interactions.transformation import (
    GoogleAIStudioInteractionsConfig,
)
from litellm.llms.vertex_ai.interactions.transformation import (
    VertexAIInteractionsConfig,
)
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

GLOBAL_BASE = "https://aiplatform.googleapis.com/v1beta1/projects/test-proj/locations/global/interactions"


class MinterRecorder:
    def __init__(self, resolved_project: str = "creds-proj") -> None:
        self.calls: list[tuple[VERTEX_CREDENTIALS_TYPES | None, str | None]] = []
        self.resolved_project = resolved_project

    def __call__(
        self,
        credentials: VERTEX_CREDENTIALS_TYPES | None,
        project_id: str | None,
    ) -> tuple[str, str]:
        self.calls.append((credentials, project_id))
        return "test-token", project_id or self.resolved_project


@pytest.fixture
def minter():
    return MinterRecorder()


@pytest.fixture
def config(minter):
    return VertexAIInteractionsConfig(mint_access_token=minter)


@pytest.fixture
def litellm_params():
    return GenericLiteLLMParams(vertex_project="test-proj", vertex_credentials="creds.json")


class TestRegistration:
    def test_vertex_ai_returns_vertex_config(self):
        assert isinstance(get_provider_interactions_api_config("vertex_ai"), VertexAIInteractionsConfig)

    def test_vertex_ai_beta_returns_vertex_config(self):
        assert isinstance(get_provider_interactions_api_config("vertex_ai_beta"), VertexAIInteractionsConfig)

    def test_gemini_still_returns_google_ai_studio_config(self):
        gemini_config = get_provider_interactions_api_config("gemini")
        assert isinstance(gemini_config, GoogleAIStudioInteractionsConfig)
        assert not isinstance(gemini_config, VertexAIInteractionsConfig)

    def test_lazy_import_resolves(self):
        assert litellm.VertexAIInteractionsConfig is VertexAIInteractionsConfig

    def test_custom_llm_provider_is_vertex_ai(self, config):
        assert config.custom_llm_provider == LlmProviders.VERTEX_AI


class TestValidateEnvironment:
    def test_sets_bearer_auth_without_gemini_headers(self, config, minter, litellm_params):
        headers = config.validate_environment(
            headers={},
            model="gemini-omni-flash-preview",
            litellm_params=litellm_params,
        )

        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Content-Type"] == "application/json"
        assert "x-goog-api-key" not in headers
        assert "Api-Revision" not in headers
        assert minter.calls == [("creds.json", "test-proj")]

    def test_caller_authorization_wins(self, config, litellm_params):
        headers = config.validate_environment(
            headers={"Authorization": "Bearer caller-token"},
            model="gemini-omni-flash-preview",
            litellm_params=litellm_params,
        )

        assert headers["Authorization"] == "Bearer caller-token"


class TestGetCompleteUrl:
    def test_defaults_to_global_v1beta1(self, config, litellm_params):
        url = config.get_complete_url(
            api_base=None,
            model="gemini-omni-flash-preview",
            litellm_params=dict(litellm_params),
        )

        assert url == GLOBAL_BASE

    def test_stream_appends_alt_sse(self, config, litellm_params):
        url = config.get_complete_url(
            api_base=None,
            model="gemini-omni-flash-preview",
            litellm_params=dict(litellm_params),
            stream=True,
        )

        assert url == f"{GLOBAL_BASE}?alt=sse"

    def test_multi_region_location_uses_rep_host(self, config):
        url = config.get_complete_url(
            api_base=None,
            model="gemini-omni-flash-preview",
            litellm_params={"vertex_project": "test-proj", "vertex_location": "us"},
        )

        assert url == "https://aiplatform.us.rep.googleapis.com/v1beta1/projects/test-proj/locations/us/interactions"

    def test_regional_location_uses_regional_host(self, config):
        url = config.get_complete_url(
            api_base=None,
            model="gemini-omni-flash-preview",
            litellm_params={"vertex_project": "test-proj", "vertex_location": "us-central1"},
        )

        assert url == (
            "https://us-central1-aiplatform.googleapis.com"
            "/v1beta1/projects/test-proj/locations/us-central1/interactions"
        )

    def test_location_env_fallback_is_ignored(self, config, monkeypatch):
        monkeypatch.setenv("VERTEXAI_LOCATION", "us-east5")

        url = config.get_complete_url(
            api_base=None,
            model="gemini-omni-flash-preview",
            litellm_params={"vertex_project": "test-proj"},
        )

        assert url == GLOBAL_BASE

    def test_api_base_override(self, config, litellm_params):
        url = config.get_complete_url(
            api_base="https://proxy.example.test",
            model="gemini-omni-flash-preview",
            litellm_params=dict(litellm_params),
        )

        assert url == "https://proxy.example.test/v1beta1/projects/test-proj/locations/global/interactions"

    def test_project_resolved_from_credentials_when_not_passed(self, config, monkeypatch):
        monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
        monkeypatch.setattr(litellm, "vertex_project", None)

        url = config.get_complete_url(
            api_base=None,
            model="gemini-omni-flash-preview",
            litellm_params={"vertex_credentials": "creds.json"},
        )

        assert url == "https://aiplatform.googleapis.com/v1beta1/projects/creds-proj/locations/global/interactions"

    def test_invalid_location_rejected(self, config):
        with pytest.raises(ValueError, match="Invalid vertex_location"):
            config.get_complete_url(
                api_base=None,
                model="gemini-omni-flash-preview",
                litellm_params={"vertex_project": "test-proj", "vertex_location": "evil.com#"},
            )

    def test_missing_project_rejected(self, monkeypatch):
        monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
        monkeypatch.setattr(litellm, "vertex_project", None)

        def unresolved_minter(
            credentials: VERTEX_CREDENTIALS_TYPES | None,
            project_id: str | None,
        ) -> tuple[str, str]:
            return "test-token", ""

        with pytest.raises(ValueError, match="Vertex AI project is required"):
            VertexAIInteractionsConfig(mint_access_token=unresolved_minter).get_complete_url(
                api_base=None,
                model="gemini-omni-flash-preview",
                litellm_params={},
            )


class TestInteractionByIdRequests:
    def test_get_url(self, config, litellm_params):
        url, request_body = config.transform_get_interaction_request(
            interaction_id="abc123",
            api_base="",
            litellm_params=litellm_params,
            headers={},
        )

        assert url == f"{GLOBAL_BASE}/abc123"
        assert request_body == {}

    def test_get_url_encodes_interaction_id(self, config, litellm_params):
        url, _ = config.transform_get_interaction_request(
            interaction_id="id/with space",
            api_base="",
            litellm_params=litellm_params,
            headers={},
        )

        assert url == f"{GLOBAL_BASE}/id%2Fwith%20space"

    def test_delete_url(self, config, litellm_params):
        url, request_body = config.transform_delete_interaction_request(
            interaction_id="abc123",
            api_base="",
            litellm_params=litellm_params,
            headers={},
        )

        assert url == f"{GLOBAL_BASE}/abc123"
        assert request_body == {}

    def test_cancel_url(self, config, litellm_params):
        url, request_body = config.transform_cancel_interaction_request(
            interaction_id="abc123",
            api_base="",
            litellm_params=litellm_params,
            headers={},
        )

        assert url == f"{GLOBAL_BASE}/abc123:cancel"
        assert request_body == {}



LABEL_FIELDS: Final = ("user_api_key_alias", "user_api_key_team_alias", "spend_logs_metadata")
IDENTITY: Final = MappingProxyType({"user_api_key_alias": "chargeback-key", "user_api_key_team_alias": "ai-platform"})
PROXY_METADATA: Final = MappingProxyType({**IDENTITY, "spend_logs_metadata": {"cost-center": "cc-1"}})
FORGED: Final = MappingProxyType({"user_api_key_alias": "forged", "app": "bot"})
SENT_BODY: Final = TypeAdapter(Mapping[str, object])
SENT_LABELS: Final = TypeAdapter(Mapping[str, str])


async def _created_interaction_body(
    config: VertexAIInteractionsConfig, extra_body: Mapping[str, object], *, is_async: bool
) -> Mapping[str, object]:
    sent: Final[list[httpx.Request]] = []  # mutable-ok: MockTransport hands each request to a callback

    def reply(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"id": "interaction-1", "status": "completed", "outputs": []})

    transport: Final = httpx.MockTransport(reply)
    litellm_params: Final = GenericLiteLLMParams(vertex_project="test-proj", metadata=dict(PROXY_METADATA))
    logging_obj: Final = Logging(
        model="gemini-omni-flash-preview",
        messages=[],
        stream=False,
        call_type="acreate_interaction" if is_async else "create_interaction",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="labels-test",
        function_id="labels-test",
    )
    handler: Final = InteractionsHTTPHandler()
    if is_async:
        await handler.async_create_interaction(
            interactions_api_config=config,
            optional_params={},
            custom_llm_provider="vertex_ai",
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            model="gemini-omni-flash-preview",
            input="hi",
            extra_body=extra_body,
            client=AsyncHTTPHandler(transport=transport),
        )
    else:
        handler.create_interaction(
            interactions_api_config=config,
            optional_params={},
            custom_llm_provider="vertex_ai",
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            model="gemini-omni-flash-preview",
            input="hi",
            extra_body=extra_body,
            client=HTTPHandler(client=httpx.Client(transport=transport)),
        )
    return SENT_BODY.validate_python(json.loads(sent[0].content))


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_create_replaces_forged_extra_body_labels_with_identity_while_the_setting_is_on(
    config: VertexAIInteractionsConfig, monkeypatch: pytest.MonkeyPatch, is_async: bool
):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", LABEL_FIELDS)

    body: Final = await _created_interaction_body(config, {"labels": dict(FORGED)}, is_async=is_async)

    assert list(SENT_LABELS.validate_python(body["labels"]).items()) == [
        *IDENTITY.items(),
        ("app", "bot"),
        ("cost-center", "cc-1"),
    ]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_create_sends_caller_labels_as_is_while_the_setting_is_off(
    config: VertexAIInteractionsConfig, monkeypatch: pytest.MonkeyPatch, is_async: bool
):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", None)

    with_labels: Final = await _created_interaction_body(config, {"labels": dict(FORGED)}, is_async=is_async)
    without_labels: Final = await _created_interaction_body(config, {}, is_async=is_async)

    assert with_labels["labels"] == dict(FORGED)
    assert "labels" not in without_labels


def test_create_drops_a_caller_labels_header_only_while_the_setting_is_on(
    config: VertexAIInteractionsConfig, litellm_params: GenericLiteLLMParams, monkeypatch: pytest.MonkeyPatch
):
    caller_headers: Final = {"x-vertex-ai-labels": "eyJ1c2VyX2FwaV9rZXlfdGVhbV9hbGlhcyI6ICJmb3JnZWQifQ==", "x-trace-id": "t-1"}
    monkeypatch.setattr(litellm, "vertex_request_label_fields", None)
    setting_off: Final = config.validate_environment(caller_headers, "gemini-omni-flash-preview", litellm_params)
    monkeypatch.setattr(litellm, "vertex_request_label_fields", LABEL_FIELDS)
    setting_on: Final = config.validate_environment(caller_headers, "gemini-omni-flash-preview", litellm_params)

    assert setting_off["x-vertex-ai-labels"] == caller_headers["x-vertex-ai-labels"]
    assert "x-vertex-ai-labels" not in setting_on
    assert setting_on["x-trace-id"] == "t-1"
