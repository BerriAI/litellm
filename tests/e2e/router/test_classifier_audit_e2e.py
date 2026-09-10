import json
import os
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Final, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue, TypeAdapter

from e2e_config import unique_marker
from e2e_http import AnthropicHeaders, NoBody, unwrap
from models import ChatMessage, KeyGenerateBody, LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e


class AuditMetadata(BaseModel):
    source_marker: str
    authorization: str = "synthetic-audit-secret"


class AuditHeaders(AnthropicHeaders):
    enable_redaction: str | None = Field(default=None, serialization_alias="x-litellm-enable-message-redaction")


class AuditRequest(BaseModel):
    model: str
    messages: list[ChatMessage] | None = None
    system: str | None = None
    instructions: str | None = None
    input: str | None = None
    max_tokens: int | None = None
    max_output_tokens: int | None = None
    metadata: AuditMetadata | None = None
    litellm_metadata: AuditMetadata | None = None


class AuditResponse(BaseModel):
    id: str


class AuditDetail(BaseModel):
    proxy_server_request: dict[str, JsonValue] | str | None = None
    response: dict[str, JsonValue] | str | None = None


@dataclass(frozen=True, slots=True)
class AuditDeployment:
    alias: str
    key: str


@pytest.fixture
def audit_deployment(proxy: ProxyClient, provider: str) -> Iterator[AuditDeployment]:
    marker: Final = unique_marker()
    classifier: Final = f"audit-classifier-{marker}"
    target: Final = f"audit-target-{marker}"
    alias: Final = f"audit-router-{marker}"
    model: Final = os.environ.get(
        f"E2E_CHEAP_{provider.upper()}_MODEL", "gpt-5.6" if provider == "openai" else "claude-haiku-4-5"
    )
    params: Final = LiteLLMParamsBody(
        model=f"{provider}/{model}",
        api_key=os.environ.get(f"{provider.upper()}_API_KEY") or f"os.environ/{provider.upper()}_API_KEY",
        api_base=os.environ.get(f"{provider.upper()}_API_BASE"),
    )
    with ExitStack() as stack:
        for name in (classifier, target):
            stack.callback(proxy.delete_model, proxy.create_model(name, params))
        stack.callback(proxy.delete_model, proxy.create_model(alias, LiteLLMParamsBody(
            model="auto_router/complexity_router",
            complexity_router_config={
                "classifier_type": "llm",
                "classifier_llm_config": {"model": classifier, "timeout_ms": 30000},
                "tiers": {tier: target for tier in ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")},
            },
        )))
        key: Final = proxy.generate_key(KeyGenerateBody(models=[alias, classifier, target]))
        stack.callback(proxy.delete_key, key)
        yield AuditDeployment(alias, key)


class TestClassifierAudit:
    @pytest.mark.covers(
        "reliability.routing.classifier_audit.separates_provider_input_and_source",
        exercised_on=("chat_completions", "messages", "responses"),
    )
    @pytest.mark.parametrize("surface", ["chat_completions", "messages", "responses"])
    @pytest.mark.parametrize("provider", ["openai", "anthropic"])
    @pytest.mark.parametrize("redact", [False, True])
    def test_classifier_audit_separates_input_and_masked_source(
        self, proxy: ProxyClient, audit_deployment: AuditDeployment, surface: Literal["chat_completions", "messages", "responses"],
        redact: bool,
    ) -> None:
        marker: Final = unique_marker()
        source_marker: Final = f"source-only-{marker}"
        prompt: Final = f"Reply with hello. Request label {marker}"
        metadata: Final = AuditMetadata(source_marker=source_marker)
        body: Final = AuditRequest(
            model=audit_deployment.alias,
            messages=[ChatMessage(role="user", content=prompt)] if surface != "responses" else None,
            input=prompt if surface == "responses" else None,
            system="Be concise" if surface == "messages" else None,
            instructions="Be concise" if surface == "responses" else None,
            max_tokens=128 if surface != "responses" else None,
            max_output_tokens=128 if surface == "responses" else None,
            metadata=metadata if surface != "responses" else None,
            litellm_metadata=metadata if surface == "responses" else None,
        )
        path: Final = {"chat_completions": "/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}[surface]
        response: Final = unwrap(proxy.transport.post(
            path, headers=AuditHeaders(authorization=f"Bearer {audit_deployment.key}", enable_redaction="true" if redact else None),
            json=body, response_type=AuditResponse,
        ))
        assert response.id
        rows: Final = proxy.poll_logs_for_key(audit_deployment.key, min_rows=2)
        assert len(rows) == 2, "Expected a classifier spend row and a routed response spend row"
        details: Final = tuple(
            unwrap(proxy.transport.get(
                f"/spend/logs/ui/{row.request_id}", headers=proxy.transport.master,
                params=NoBody(), response_type=AuditDetail,
            )) for row in rows
        )
        adapter: Final = TypeAdapter(dict[str, JsonValue])
        requests: Final = tuple(
            adapter.validate_json(detail.proxy_server_request) if isinstance(detail.proxy_server_request, str)
            else detail.proxy_server_request or {} for detail in details
        )
        audits: Final = tuple(item for item in requests if "classifier_input" in item)
        if redact:
            assert audits == ()
            assert all("originating_request_masked" not in item for item in requests)
            return
        assert len(audits) == 1, "The audit belongs only to the classifier call"
        audit: Final = audits[0]
        assert marker in json.dumps(audit["classifier_input"])
        assert source_marker not in json.dumps(audit["classifier_input"])
        assert source_marker in json.dumps(audit["originating_request_masked"])
        assert "synthetic-audit-secret" not in json.dumps(audit)
        assert audit_deployment.alias in json.dumps(audit["originating_request_masked"])
        assert any("tier" in str(detail.response) for detail in details)
