"""Live provider x auth x input coverage for ``litellm.ocr`` / ``litellm.aocr``.

Each ``Case`` is one hand-picked cell, not the full cross product: every provider
exercises each of its credential kinds in both ``explicit`` (kwargs) and ``env``
(monkeypatched environment) mode at least once, and every input kind a provider
accepts is exercised at least once. Sync and async are spread across the cells.
Every cell also checks the success callback saw the same response and cost.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import pytest

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.base_llm.ocr.transformation import OCRResponse

Document = Mapping[str, object]
AuthMode = Literal["explicit", "env"]
CallStyle = Literal["sync", "async"]


@dataclass(frozen=True, slots=True)
class LoggedCall:
    payload: Mapping[str, object]
    response: object


class RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger.__init__ is untyped
        self.calls: Final[list[LoggedCall]] = []  # mutable-ok: append-only sink the callback hooks write into

    def _record(self, kwargs: Mapping[str, object], response_obj: object) -> None:
        payload: Final = _string_keyed(kwargs.get("standard_logging_object"))
        self.calls.append(LoggedCall(payload, response_obj))

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._record(kwargs, response_obj)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._record(kwargs, response_obj)

    async def wait_for_call(self, timeout: float = 10.0) -> LoggedCall:
        deadline: Final = asyncio.get_running_loop().time() + timeout
        while not self.calls:
            assert asyncio.get_running_loop().time() < deadline, "success callback never fired"
            await asyncio.sleep(0.05)
        assert len(self.calls) == 1, self.calls
        return self.calls[0]


@pytest.fixture
def logger(monkeypatch: pytest.MonkeyPatch) -> RecordingLogger:
    recorder: Final = RecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    for registry in ("success_callback", "_async_success_callback", "failure_callback", "_async_failure_callback"):
        monkeypatch.setattr(litellm, registry, [])
    return recorder


TESTS_DIR: Final = Path(__file__).resolve().parents[1]
PDF_PATH: Final = TESTS_DIR / "llm_translation" / "fixtures" / "dummy.pdf"
PNG_PATH: Final = TESTS_DIR / "image_gen_tests" / "test_image.png"
PINNED_CDN: Final = "https://cdn.jsdelivr.net/gh/BerriAI/litellm@d769e81c90d453240c61fc572cdb27fae06a89d0"
PDF_URL: Final = f"{PINNED_CDN}/tests/llm_translation/fixtures/dummy.pdf"
PNG_URL: Final = f"{PINNED_CDN}/tests/image_gen_tests/test_image.png"
PDF_TEXT: Final = "Test PDF File"
PNG_TEXT: Final = "LiteLLM"


class _NamedReader(io.BytesIO):
    def __init__(self, path: Path) -> None:
        super().__init__(path.read_bytes())
        self.name: Final = path.name


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


@dataclass(frozen=True, slots=True)
class Input:
    id: str
    build: Callable[[], Document]
    expected_text: str


PDF_BY_URL: Final = Input("pdf_url", lambda: {"type": "document_url", "document_url": PDF_URL}, PDF_TEXT)
PNG_BY_URL: Final = Input("image_url", lambda: {"type": "image_url", "image_url": PNG_URL}, PNG_TEXT)
PDF_DATA_URI: Final = Input(
    "pdf_data_uri",
    lambda: {"type": "document_url", "document_url": _data_uri(PDF_PATH, "application/pdf")},
    PDF_TEXT,
)
PNG_DATA_URI: Final = Input(
    "image_data_uri", lambda: {"type": "image_url", "image_url": _data_uri(PNG_PATH, "image/png")}, PNG_TEXT
)
PDF_AS_PATH: Final = Input("pdf_path", lambda: {"type": "file", "file": PDF_PATH}, PDF_TEXT)
PDF_AS_BYTES: Final = Input(
    "pdf_bytes", lambda: {"type": "file", "file": PDF_PATH.read_bytes(), "mime_type": "application/pdf"}, PDF_TEXT
)
PNG_AS_BYTES: Final = Input(
    "image_bytes", lambda: {"type": "file", "file": PNG_PATH.read_bytes(), "mime_type": "image/png"}, PNG_TEXT
)
PNG_AS_FILE_OBJECT: Final = Input(
    "image_file_object", lambda: {"type": "file", "file": _NamedReader(PNG_PATH)}, PNG_TEXT
)


@dataclass(frozen=True, slots=True)
class Secret:
    """One credential value: the ``litellm.ocr`` kwarg it travels in, the env var litellm reads
    when the kwarg is omitted, and the env var that holds the value in the test process."""

    kwarg: str
    env: str
    source: str | None = None

    @property
    def source_env(self) -> str:
        return self.source or self.env


@dataclass(frozen=True, slots=True)
class Credential:
    id: str
    secrets: tuple[Secret, ...]


@dataclass(frozen=True, slots=True)
class Provider:
    id: str
    model: str
    credentials: tuple[Credential, ...]
    params: Mapping[str, str] = MappingProxyType({})

    @property
    def env_vars(self) -> frozenset[str]:
        return frozenset(secret.env for credential in self.credentials for secret in credential.secrets)


MISTRAL_KEY: Final = Credential("api_key", (Secret("api_key", "MISTRAL_API_KEY"),))
COHERE_KEY: Final = Credential("api_key", (Secret("api_key", "COHERE_API_KEY"),))
REDUCTO_KEY: Final = Credential("api_key", (Secret("api_key", "REDUCTO_API_KEY"),))

AZURE_ENTRA_SECRETS: Final = (
    Secret("tenant_id", "AZURE_TENANT_ID", "AZURE_FOUNDRY_TENANT_ID"),
    Secret("client_id", "AZURE_CLIENT_ID", "AZURE_FOUNDRY_ADMIN_CLIENT_ID"),
    Secret("client_secret", "AZURE_CLIENT_SECRET", "AZURE_FOUNDRY_ADMIN_CLIENT_SECRET"),
)
AZURE_AI_BASE: Final = Secret("api_base", "AZURE_AI_API_BASE")
AZURE_AI_KEY: Final = Credential("api_key", (AZURE_AI_BASE, Secret("api_key", "AZURE_AI_API_KEY")))
AZURE_AI_ENTRA: Final = Credential("entra", (AZURE_AI_BASE, *AZURE_ENTRA_SECRETS))

AZURE_DI_BASE: Final = Secret("api_base", "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_DI_KEY: Final = Credential("api_key", (AZURE_DI_BASE, Secret("api_key", "AZURE_DOCUMENT_INTELLIGENCE_API_KEY")))
AZURE_DI_ENTRA: Final = Credential("entra", (AZURE_DI_BASE, *AZURE_ENTRA_SECRETS))

VERTEX_SERVICE_ACCOUNT: Final = Credential(
    "service_account",
    (Secret("vertex_credentials", "VERTEXAI_CREDENTIALS"), Secret("vertex_project", "VERTEXAI_PROJECT")),
)

MISTRAL: Final = Provider("mistral", "mistral/mistral-ocr-latest", (MISTRAL_KEY,))
AZURE_AI_MISTRAL: Final = Provider(
    "azure_ai_mistral", "azure_ai/mistral-document-ai-2512", (AZURE_AI_KEY, AZURE_AI_ENTRA)
)
AZURE_DOC_INTELLIGENCE: Final = Provider(
    "azure_doc_intelligence", "azure_ai/doc-intelligence/prebuilt-layout", (AZURE_DI_KEY, AZURE_DI_ENTRA)
)
COHERE: Final = Provider("cohere", "cohere/parse-v5.0", (COHERE_KEY,))
REDUCTO_V3: Final = Provider("reducto_v3", "reducto/parse-v3", (REDUCTO_KEY,))
REDUCTO_LEGACY: Final = Provider("reducto_legacy", "reducto/parse-legacy", (REDUCTO_KEY,))
VERTEX_MISTRAL: Final = Provider(
    "vertex_mistral",
    "vertex_ai/mistral-ocr-2505",
    (VERTEX_SERVICE_ACCOUNT,),
    MappingProxyType({"vertex_location": "us-central1"}),
)


@dataclass(frozen=True, slots=True)
class Case:
    provider: Provider
    credential: Credential
    auth: AuthMode
    document: Input
    call: CallStyle

    @property
    def id(self) -> str:
        return f"{self.provider.id}-{self.credential.id}-{self.auth}-{self.document.id}-{self.call}"

    def bind_credentials(self, monkeypatch: pytest.MonkeyPatch) -> Mapping[str, str]:
        """Clear every env var the provider could fall back to, then supply this case's values via kwargs or env."""
        values: Final = {secret: os.environ.get(secret.source_env) for secret in self.credential.secrets}
        missing: Final = tuple(secret.source_env for secret, value in values.items() if not value)
        if missing:
            pytest.skip(f"{', '.join(missing)} not set")
        for env_var in self.provider.env_vars:
            monkeypatch.delenv(env_var, raising=False)
        if self.auth == "explicit":
            return {secret.kwarg: value for secret, value in values.items() if value}
        for secret, value in values.items():
            monkeypatch.setenv(secret.env, value or "")
        return {}

    async def run(self, credentials: Mapping[str, str]) -> OCRResponse:
        kwargs: Final = {**self.provider.params, **credentials}
        document: Final = self.document.build()
        response: Final = (
            await litellm.aocr(model=self.provider.model, document=document, **kwargs)  # pyright: ignore[reportUnknownMemberType]  # @client erases the signature
            if self.call == "async"
            else litellm.ocr(model=self.provider.model, document=document, **kwargs)
        )
        assert isinstance(response, OCRResponse)
        return response


CASES: Final = (
    Case(MISTRAL, MISTRAL_KEY, "explicit", PDF_BY_URL, "sync"),
    Case(MISTRAL, MISTRAL_KEY, "env", PNG_BY_URL, "async"),
    Case(MISTRAL, MISTRAL_KEY, "explicit", PDF_AS_PATH, "sync"),
    Case(MISTRAL, MISTRAL_KEY, "explicit", PNG_AS_BYTES, "async"),
    Case(MISTRAL, MISTRAL_KEY, "explicit", PNG_AS_FILE_OBJECT, "sync"),
    Case(AZURE_AI_MISTRAL, AZURE_AI_KEY, "explicit", PDF_BY_URL, "sync"),
    Case(AZURE_AI_MISTRAL, AZURE_AI_KEY, "env", PNG_BY_URL, "async"),
    Case(AZURE_AI_MISTRAL, AZURE_AI_ENTRA, "explicit", PDF_AS_PATH, "sync"),
    Case(AZURE_AI_MISTRAL, AZURE_AI_ENTRA, "env", PDF_DATA_URI, "async"),
    Case(AZURE_DOC_INTELLIGENCE, AZURE_DI_KEY, "explicit", PDF_BY_URL, "sync"),
    Case(AZURE_DOC_INTELLIGENCE, AZURE_DI_KEY, "env", PNG_AS_BYTES, "async"),
    Case(AZURE_DOC_INTELLIGENCE, AZURE_DI_ENTRA, "explicit", PNG_BY_URL, "async"),
    Case(AZURE_DOC_INTELLIGENCE, AZURE_DI_ENTRA, "env", PDF_AS_PATH, "sync"),
    Case(COHERE, COHERE_KEY, "explicit", PNG_BY_URL, "sync"),
    Case(COHERE, COHERE_KEY, "env", PNG_DATA_URI, "async"),
    Case(REDUCTO_V3, REDUCTO_KEY, "explicit", PDF_AS_PATH, "sync"),
    Case(REDUCTO_V3, REDUCTO_KEY, "env", PNG_AS_BYTES, "async"),
    Case(REDUCTO_V3, REDUCTO_KEY, "explicit", PDF_DATA_URI, "async"),
    Case(REDUCTO_LEGACY, REDUCTO_KEY, "explicit", PDF_AS_BYTES, "sync"),
    Case(VERTEX_MISTRAL, VERTEX_SERVICE_ACCOUNT, "explicit", PDF_BY_URL, "sync"),
    Case(VERTEX_MISTRAL, VERTEX_SERVICE_ACCOUNT, "env", PNG_BY_URL, "async"),
)


def _response_cost(response: OCRResponse) -> float:
    response_cost: Final[object] = response._hidden_params.get("response_cost")  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]  # response_cost is only surfaced on _hidden_params
    assert isinstance(response_cost, float) and response_cost > 0
    return response_cost


def _assert_ocr_response(response: OCRResponse, model: str, expected_text: str) -> None:
    assert response.object == "ocr"
    assert response.model == model.split("/", 1)[1]
    assert [page.index for page in response.pages] == list(range(len(response.pages)))
    text: Final = re.sub(r"\s+", " ", " ".join(page.markdown for page in response.pages))
    assert expected_text.lower() in text.lower(), text
    assert response.usage_info is not None
    assert response.usage_info.pages_processed == len(response.pages)
    _response_cost(response)


def _string_keyed(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping), type(value)
    items: Final = tuple(value.items())  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]  # narrowed from object
    return MappingProxyType({str(key): value for key, value in items})  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # narrowed from object


def _assert_logged(logged: LoggedCall, response: OCRResponse, model: str, logged_model: str, call: CallStyle) -> None:
    assert isinstance(logged.response, OCRResponse)
    assert logged.response.pages == response.pages
    assert logged.payload["status"] == "success"
    assert logged.payload["call_type"] == ("aocr" if call == "async" else "ocr")
    assert logged.payload["custom_llm_provider"] == model.split("/", 1)[0]
    assert logged.payload["model"] == logged_model
    assert logged.payload["response_cost"] == _response_cost(response)


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
async def test_ocr(case: Case, monkeypatch: pytest.MonkeyPatch, logger: RecordingLogger) -> None:
    credentials: Final = case.bind_credentials(monkeypatch)
    response: Final = await case.run(credentials)
    _assert_ocr_response(response, case.provider.model, case.document.expected_text)
    _assert_logged(await logger.wait_for_call(), response, case.provider.model, response.model, case.call)


async def test_router_aocr(monkeypatch: pytest.MonkeyPatch, logger: RecordingLogger) -> None:
    case: Final = Case(MISTRAL, MISTRAL_KEY, "explicit", PDF_BY_URL, "async")
    router: Final = Router(
        model_list=[
            {
                "model_name": "ocr-alias",
                "litellm_params": {"model": MISTRAL.model, **case.bind_credentials(monkeypatch)},
            }
        ]
    )
    response: Final = await router.aocr(model="ocr-alias", document=PDF_BY_URL.build())  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # Router.aocr is untyped
    assert isinstance(response, OCRResponse)
    _assert_ocr_response(response, MISTRAL.model, PDF_TEXT)
    _assert_logged(await logger.wait_for_call(), response, MISTRAL.model, MISTRAL.model, case.call)
