"""One scenario per non-management inference endpoint, run over every catalog
provider that serves it and every way the proxy can hold that provider's secret.

A cell is (endpoint case, provider, auth mode). The case knows how to call the
endpoint and what a meaningful answer looks like; the provider knows its backend
model and credential fields; the auth mode decides whether the deployment carries
`os.environ/` references, literal values, or a `litellm_credential_name`. The
same scenario therefore hits `/v1/messages`, `/v1/responses`, `/chat/completions`
and the rest identically, so a translation bug fixed on one endpoint but not
another shows up as one red cell next to green ones.

Every cell calls the proxy the way a customer does: the OpenAI SDK for the
OpenAI-compatible surface and the Anthropic SDK for `/v1/messages`, each holding
a fresh virtual key. OpenAI and Anthropic cells are edge-wired and replayable;
every other provider runs live in every fixture mode. Streaming cases only assert
grammar and content, never provider timing, so replay stays fast.

The selected cells' deployments are registered as one batch before the first cell
runs, so the module pays the data-plane reload budget once instead of once per
cell; each cell still calls through its own fresh virtual key.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from _pytest.mark import ParameterSet
from anthropic.types import RawContentBlockDeltaEvent, TextBlock, TextDelta
from e2e_config import SLOW_PROVIDER_TIMEOUT_SECONDS, provider_edge_base, unique_marker
from endpoint_matrix import (
    AuthMode,
    LlmRoute,
    MatrixEndpoint,
    Provider,
    Streaming,
    credential_values,
    deployment_params,
    selected_auth_modes,
    selected_providers,
)
from lifecycle import ResourceManager
from models import CredentialCreateBody, ModelInfoBody, ModelNewBody
from openai.types.responses import ResponseTextDeltaEvent
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients, response_header

pytestmark = [pytest.mark.e2e]

THIS_MODULE: Final = Path(__file__).resolve()
MAX_TOKENS: Final = 256
WEATHER_WAV: Final = Path(__file__).resolve().parent / "realtime" / "fixtures" / "weather_question_24k.wav"
EDIT_PNG: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PMQ0AAAwDoPo3"
    "3UrYvQQckD4XAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AYHLAMpT0sIcNbcEAAAAAElFTkSuQmCC"
)


def _greeting_prompt() -> str:
    return f"Reply with one short friendly sentence. Request {unique_marker()}"


def _counting_prompt() -> str:
    return f"Count from 1 to 10, one number per line. Request {unique_marker()}"


def _chat(sdk: SdkClients, key: str, model: str) -> None:
    completion: Final = sdk.openai(key).chat.completions.create(
        model=model, messages=[{"role": "user", "content": _greeting_prompt()}], max_tokens=MAX_TOKENS
    )
    assert completion.choices, f"/chat/completions returned no choices: {completion!r}"
    assert (completion.choices[0].message.content or "").strip(), (
        f"/chat/completions returned no assistant text: {completion!r}"
    )


def _chat_stream(sdk: SdkClients, key: str, model: str) -> None:
    chunks: Final = tuple(
        sdk.openai(key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _counting_prompt()}],
            max_tokens=MAX_TOKENS,
            stream=True,
        )
    )
    assert len(chunks) > 1, f"/chat/completions stream delivered a single chunk: {chunks!r}"
    text: Final = "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
    assert text.strip(), f"/chat/completions stream carried no content deltas: {chunks[:3]!r}"


def _completions(sdk: SdkClients, key: str, model: str) -> None:
    completion: Final = sdk.openai(key).completions.create(
        model=model, prompt=_greeting_prompt(), max_tokens=MAX_TOKENS, extra_body=NO_PROXY_CACHE
    )
    assert completion.choices, f"/v1/completions returned no choices: {completion!r}"
    assert completion.choices[0].text.strip(), f"/v1/completions returned no completion text: {completion!r}"


def _messages(sdk: SdkClients, key: str, model: str) -> None:
    message: Final = sdk.anthropic(key).messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": _greeting_prompt()}],
        extra_body=NO_PROXY_CACHE,
    )
    assert message.role == "assistant", f"/v1/messages did not answer as the assistant: {message!r}"
    text: Final = "".join(block.text for block in message.content if isinstance(block, TextBlock))
    assert text.strip(), f"/v1/messages returned no text block: {message.content!r}"


def _messages_stream(sdk: SdkClients, key: str, model: str) -> None:
    events: Final = tuple(
        sdk.anthropic(key).messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": _counting_prompt()}],
            stream=True,
            extra_body=NO_PROXY_CACHE,
        )
    )
    types: Final = tuple(event.type for event in events)
    assert types and types[0] == "message_start", f"/v1/messages stream did not open with message_start: {types[:3]}"
    assert types[-1] == "message_stop", f"/v1/messages stream did not close with message_stop: {types[-3:]}"
    text: Final = "".join(
        event.delta.text
        for event in events
        if isinstance(event, RawContentBlockDeltaEvent) and isinstance(event.delta, TextDelta)
    )
    assert text.strip(), f"/v1/messages stream carried no text deltas: {types}"


def _responses(sdk: SdkClients, key: str, model: str) -> None:
    response: Final = sdk.openai(key).responses.create(model=model, input=_greeting_prompt(), extra_body=NO_PROXY_CACHE)
    assert response.status == "completed", f"/v1/responses did not complete: {response!r}"
    assert response.output_text.strip(), f"/v1/responses returned no output text: {response.output!r}"


def _responses_stream(sdk: SdkClients, key: str, model: str) -> None:
    events: Final = tuple(
        sdk.openai(key).responses.create(model=model, input=_counting_prompt(), stream=True, extra_body=NO_PROXY_CACHE)
    )
    types: Final = tuple(event.type for event in events)
    assert types and types[-1] == "response.completed", (
        f"/v1/responses stream did not end with response.completed: {types[-3:]}"
    )
    text: Final = "".join(event.delta for event in events if isinstance(event, ResponseTextDeltaEvent))
    assert text.strip(), f"/v1/responses stream carried no output_text deltas: {types}"


def _embeddings(sdk: SdkClients, key: str, model: str) -> None:
    embeddings: Final = sdk.openai(key).embeddings.create(
        model=model, input=f"the quick brown fox {unique_marker()}", encoding_format="float", extra_body=NO_PROXY_CACHE
    )
    assert embeddings.data, f"/embeddings returned no data: {embeddings!r}"
    vector: Final = embeddings.data[0].embedding
    assert len(vector) > 1, f"/embeddings returned no vector: {embeddings!r}"
    assert any(component != 0.0 for component in vector), "/embeddings returned an all-zero vector"


def _audio_speech(sdk: SdkClients, key: str, model: str) -> None:
    response: Final = sdk.openai(key).audio.speech.with_raw_response.create(
        model=model, voice="alloy", input="The matrix speaks."
    )
    content_type: Final = response_header(response.headers, "content-type")
    assert "audio" in (content_type or ""), f"/v1/audio/speech content-type is not audio: {content_type!r}"
    assert response.content, "/v1/audio/speech returned an empty body"


def _audio_speech_stream(sdk: SdkClients, key: str, model: str) -> None:
    with sdk.openai(key).audio.speech.with_streaming_response.create(
        model=model,
        voice="alloy",
        input="Streaming speech should arrive in several audio chunks so playback can start early.",
    ) as response:
        content_type: Final = response_header(response.headers, "content-type")
        transfer_encoding: Final = response_header(response.headers, "transfer-encoding")
        total_bytes: Final = sum(len(chunk) for chunk in response.iter_bytes(chunk_size=8192))
    assert "audio" in (content_type or ""), f"streamed speech content-type is not audio: {content_type!r}"
    assert "chunked" in (transfer_encoding or ""), (
        f"/v1/audio/speech did not stream: transfer-encoding={transfer_encoding!r}"
    )
    assert total_bytes > 0, "streamed speech delivered no audio bytes"


def _audio_transcriptions(sdk: SdkClients, key: str, model: str) -> None:
    transcription: Final = sdk.openai(key).audio.transcriptions.create(
        model=model, file=(WEATHER_WAV.name, WEATHER_WAV.read_bytes(), "audio/wav")
    )
    text: Final = transcription.text.strip()
    assert "weather" in text.lower(), f"transcript of a spoken weather question does not mention weather: {text!r}"


def _images_generations(sdk: SdkClients, key: str, model: str) -> None:
    images: Final = sdk.openai(key).images.generate(
        model=model,
        prompt="A single red circle on a white background",
        n=1,
        size="1024x1024",
        timeout=SLOW_PROVIDER_TIMEOUT_SECONDS,
    )
    assert images.data, f"/v1/images/generations returned no images: {images!r}"
    assert images.data[0].b64_json or images.data[0].url, f"generated image has no payload: {images.data[0]!r}"


def _images_edits(sdk: SdkClients, key: str, model: str) -> None:
    edited: Final = sdk.openai(key).images.edit(
        model=model,
        image=("image.png", EDIT_PNG, "image/png"),
        prompt="Add a small red circle in the center",
        timeout=SLOW_PROVIDER_TIMEOUT_SECONDS,
    )
    assert edited.data, f"/v1/images/edits returned no images: {edited!r}"
    assert edited.data[0].b64_json or edited.data[0].url, f"edited image has no payload: {edited.data[0]!r}"


def _moderations(sdk: SdkClients, key: str, model: str) -> None:
    moderation: Final = sdk.openai(key).moderations.create(
        model=model, input=f"I enjoy long walks on sunny days. {unique_marker()}"
    )
    assert len(moderation.results) == 1, f"/v1/moderations did not return one verdict for one input: {moderation!r}"
    assert not moderation.results[0].flagged, f"/v1/moderations flagged a benign sentence: {moderation.results[0]!r}"


@dataclass(frozen=True, slots=True)
class EndpointCase:
    endpoint: MatrixEndpoint
    streaming: Streaming
    run: Callable[[SdkClients, str, str], None]


ENDPOINT_CASES: Final[tuple[EndpointCase, ...]] = (
    EndpointCase("chat_completions", "nonstream", _chat),
    EndpointCase("chat_completions", "stream", _chat_stream),
    EndpointCase("completions", "nonstream", _completions),
    EndpointCase("messages", "nonstream", _messages),
    EndpointCase("messages", "stream", _messages_stream),
    EndpointCase("responses", "nonstream", _responses),
    EndpointCase("responses", "stream", _responses_stream),
    EndpointCase("embeddings", "nonstream", _embeddings),
    EndpointCase("audio_speech", "nonstream", _audio_speech),
    EndpointCase("audio_speech", "stream", _audio_speech_stream),
    EndpointCase("audio_transcriptions", "nonstream", _audio_transcriptions),
    EndpointCase("images_generations", "nonstream", _images_generations),
    EndpointCase("images_edits", "nonstream", _images_edits),
    EndpointCase("moderations", "nonstream", _moderations),
)


@dataclass(frozen=True, slots=True)
class MatrixCell:
    case: EndpointCase
    provider: Provider
    auth_mode: AuthMode

    @property
    def registry_id(self) -> str:
        route: Final = self.provider.id_route(self.case.endpoint)
        return f"llm.{self.case.endpoint}.{route}.basic.{self.case.streaming}.works"

    @property
    def test_id(self) -> str:
        return f"{self.case.endpoint}-{self.case.streaming}-{self.provider.route}-{self.auth_mode}"

    @property
    def deployment(self) -> DeploymentKey:
        return (self.provider.route, self.case.endpoint, self.auth_mode)


type DeploymentKey = tuple[LlmRoute, MatrixEndpoint, AuthMode]


def _cells() -> tuple[MatrixCell, ...]:
    return tuple(
        MatrixCell(case, provider, auth_mode)
        for provider in selected_providers()
        for case in ENDPOINT_CASES
        if case.endpoint in provider.backends
        for auth_mode in selected_auth_modes()
    )


def _param(cell: MatrixCell) -> ParameterSet:
    replay: Final = (pytest.mark.replayable,) if cell.provider.replayable() else ()
    return pytest.param(cell, id=cell.test_id, marks=(pytest.mark.covers(cell.registry_id), *replay))


def _selected_cells(session: pytest.Session) -> tuple[MatrixCell, ...]:
    """The cells pytest will actually run from this module, after -m / -k deselection,
    so replay registers no deployment for a provider it holds no credential for."""
    return tuple(
        cell
        for item in session.items
        if isinstance(item, pytest.Function) and item.path == THIS_MODULE
        for cell in (item.callspec.params.get("cell"),)
        if isinstance(cell, MatrixCell)
    )


def _deployment_body(key: DeploymentKey, provider: Provider, credential_name: str | None) -> ModelNewBody:
    _, endpoint, auth_mode = key
    edge_base: Final = None if provider.edge_mount is None else provider_edge_base(provider.edge_mount)
    return ModelNewBody(
        model_name=f"e2e-matrix-{endpoint}-{unique_marker()}",
        litellm_params=deployment_params(
            provider, endpoint, auth_mode, edge_base=edge_base, credential_name=credential_name
        ),
        model_info=ModelInfoBody(),
    )


@pytest.fixture(scope="module")
def deployments(request: pytest.FixtureRequest, proxy: ProxyClient) -> Iterator[Mapping[DeploymentKey, str]]:
    """One deployment per (provider, endpoint, auth mode) among the selected cells, written
    as a single batch, plus one stored credential per provider that any selected cell
    references by name. Yields deployment key -> model alias; tears everything down."""
    cells: Final = _selected_cells(request.session)
    providers: Final[Mapping[LlmRoute, Provider]] = MappingProxyType(
        {cell.provider.route: cell.provider for cell in cells}
    )
    stored: Final[frozenset[LlmRoute]] = frozenset(
        cell.provider.route for cell in cells if cell.auth_mode == "stored_credential"
    )
    credentials: Final[Mapping[LlmRoute, str]] = MappingProxyType(
        {route: f"e2e-matrix-cred-{unique_marker()}" for route in stored}
    )
    for route, name in credentials.items():
        proxy.create_credential(
            CredentialCreateBody(credential_name=name, credential_values=dict(credential_values(providers[route])))
        )
    try:
        keys: Final = tuple(dict.fromkeys(cell.deployment for cell in cells))
        bodies: Final = tuple(
            _deployment_body(key, providers[key[0]], credentials[key[0]] if key[2] == "stored_credential" else None)
            for key in keys
        )
        model_ids: Final = proxy.register_models(bodies)
        try:
            yield MappingProxyType(dict(zip(keys, (body.model_name for body in bodies), strict=True)))
        finally:
            for model_id in model_ids:
                proxy.delete_model(model_id)
    finally:
        for name in credentials.values():
            proxy.delete_credential(name)


class TestEndpointMatrix:
    @pytest.mark.parametrize("cell", tuple(_param(cell) for cell in _cells()))
    def test_endpoint_answers_through_every_provider_and_auth_mode(  # test-quality-ok: the cell runner asserts
        self,
        cell: MatrixCell,
        sdk: SdkClients,
        resources: ResourceManager,
        deployments: Mapping[DeploymentKey, str],
    ) -> None:
        cell.case.run(sdk, resources.key(), deployments[cell.deployment])
