from __future__ import annotations

import json
from typing import Final

from .....shared.parity.recorded_http import HttpHeader, RecordedHttpResponse
from ...models import RouteFixture, RouteSpec, TraceScenario, TraceSuite


def _fixture(model: str, document: dict[str, str] | None = None) -> RouteFixture:
    response: Final = json.dumps(
        {
            "pages": [{"index": 0, "markdown": "hello"}],
            "model": "mistral-ocr-latest",
            "usage_info": {"pages_processed": 1},
        }
    ).encode()
    return RouteFixture(
        kwargs={
            "model": model,
            "document": document or {"type": "document_url", "document_url": "https://example.com/document.pdf"},
            "pages": [0],
        },
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                200, (HttpHeader(name="content-type", value="application/json"),), response
            ),
        ),
    )


def _mistral_fixture(_base_url: str) -> RouteFixture:
    return _fixture("mistral/mistral-ocr-latest")


def _callback_fixture(*, failure: bool) -> RouteFixture:
    fixture: Final = _fixture("mistral/mistral-ocr-latest")
    provider_responses: Final = (
        (
            RecordedHttpResponse.from_bytes(
                400,
                (HttpHeader(name="content-type", value="application/json"),),
                b'{"message":"trace callback provider failure"}',
            ),
        )
        if failure
        else fixture.provider_responses
    )
    return RouteFixture(
        kwargs=fixture.kwargs,
        provider_responses=provider_responses,
        expected_failure=failure,
    )


def _mistral_callback_success_fixture(_base_url: str) -> RouteFixture:
    return _callback_fixture(failure=False)


def _mistral_callback_failure_fixture(_base_url: str) -> RouteFixture:
    return _callback_fixture(failure=True)


def _azure_fixture(_base_url: str) -> RouteFixture:
    return _fixture(
        "azure_ai/pixtral-12b-2409",
        {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
    )


def _vertex_deepseek_fixture(_base_url: str) -> RouteFixture:
    vertex: Final = {"vertex_project": "trace-project", "vertex_location": "us-central1"}
    return RouteFixture(
        kwargs={
            "model": "vertex_ai/deepseek-ocr-maas",
            "document": {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
            **vertex,
        },
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                200,
                (HttpHeader(name="content-type", value="application/json"),),
                json.dumps(
                    {
                        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }
                ).encode(),
            ),
        ),
    )


def _vertex_deepseek_credentials_fixture(base_url: str) -> RouteFixture:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    fixture: Final = _vertex_deepseek_fixture(base_url)
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    credentials: Final = json.dumps(
        {
            "type": "service_account",
            "project_id": "trace-project",
            "private_key_id": "trace-key",
            "private_key": private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ).decode(),
            "client_email": "trace@trace-project.iam.gserviceaccount.com",
            "token_uri": f"{base_url}/token",
        }
    )
    return RouteFixture(
        kwargs={**fixture.kwargs, "api_key": None},
        environment=(("VERTEXAI_CREDENTIALS", credentials), ("VERTEX_AI_API_KEY", "")),
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                200,
                (HttpHeader(name="content-type", value="application/json"),),
                b'{"access_token":"trace-token","token_type":"Bearer","expires_in":3600}',
            ),
            *fixture.provider_responses,
        ),
    )


def _cohere_fixture(_base_url: str) -> RouteFixture:
    return RouteFixture(
        kwargs={
            "model": "cohere/parse-v5.0",
            "document": {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
            "output_format": "blocks",
        },
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                200,
                (HttpHeader(name="content-type", value="application/json"),),
                json.dumps(
                    {
                        "pages": [{"index": 0, "blocks": [{"type": "text", "text": {"content": "hello"}}]}],
                        "meta": {"billed_units": {"pages": 1}},
                    }
                ).encode(),
            ),
        ),
    )


def _azure_document_intelligence_fixture(base_url: str) -> RouteFixture:
    completed: Final = json.dumps(
        {
            "status": "succeeded",
            "analyzeResult": {
                "content": "hello",
                "pages": [
                    {
                        "pageNumber": 1,
                        "width": 8.5,
                        "height": 11,
                        "unit": "inch",
                        "lines": [{"content": "hello"}],
                    }
                ],
            },
        }
    ).encode()
    return RouteFixture(
        kwargs={
            "model": "azure_ai/doc-intelligence/prebuilt-read",
            "document": {
                "type": "document_url",
                "document_url": "data:application/pdf;base64,aGVsbG8=",
            },
            "pages": [0],
        },
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                202,
                (
                    HttpHeader(name="content-type", value="application/json"),
                    HttpHeader(name="operation-location", value=f"{base_url}/operations/trace"),
                ),
                b"{}",
            ),
            RecordedHttpResponse.from_bytes(
                200,
                (HttpHeader(name="content-type", value="application/json"),),
                completed,
            ),
        ),
    )


SPEC: Final = RouteSpec("ocr", ("ocr", "aocr"), _mistral_fixture)
TRACE_SUITE: Final = TraceSuite(
    route=SPEC,
    scenarios=(
        TraceScenario(
            name="sync-mistral",
            fixture=_mistral_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-mistral",
            fixture=_mistral_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-mistral-callback-success",
            fixture=_mistral_callback_success_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-mistral-callback-success",
            fixture=_mistral_callback_success_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-mistral-callback-failure",
            fixture=_mistral_callback_failure_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-mistral-callback-failure",
            fixture=_mistral_callback_failure_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-azure-ai",
            fixture=_azure_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-azure-ai",
            fixture=_azure_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-azure-document-intelligence",
            fixture=_azure_document_intelligence_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-azure-document-intelligence",
            fixture=_azure_document_intelligence_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-vertex-deepseek",
            fixture=_vertex_deepseek_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-vertex-deepseek",
            fixture=_vertex_deepseek_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-vertex-deepseek-credentials",
            fixture=_vertex_deepseek_credentials_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-vertex-deepseek-credentials",
            fixture=_vertex_deepseek_credentials_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="async-cohere",
            fixture=_cohere_fixture,
            asynchronous=True,
        ),
    ),
)
