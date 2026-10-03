"""Live e2e: the non-conversational endpoints on the providers the OpenAI-only tests
in this directory never reach; Cohere embeddings, and Vertex AI speech and images.

Each test registers its own deployment through /model/new (deleted on teardown) and
asserts the customer-observable payload: a real non-zero vector for /embeddings, and
for the two binary endpoints the decoded bytes themselves. A content-type header is
the provider's claim rather than evidence, and so is a magic number on its own, so
the audio and image cases read the container's own structure and check it against
itself: a RIFF/WAVE whose declared sizes match the bytes that arrived, whose fmt
chunk describes a coherent stream, and whose chunk list carries a data chunk holding
real samples, and a PNG whose IHDR gives real dimensions and whose final chunk is
the IEND that marks a complete file. Credentials stay in the proxy's env and are
referenced as `os.environ/...`, so no secret travels in the request.

Structural checks stop where a real decoder would begin. They prove the container is
internally consistent and complete rather than that every sample decodes, which is
the line worth holding in a coverage test.

Vertex text-to-speech goes to the Google Cloud synthesize endpoint rather than a
regional Vertex endpoint, so the deployment only needs the project; image generation
is served by gemini-2.5-flash-image, the image model this project has access to.
Vertex synthesizes LINEAR16, so the audio arrives as a RIFF/WAVE container even
though the gateway labels every non-gemini speech response audio/mpeg.
"""

from __future__ import annotations

import base64

import pytest

from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from endpoints_client import EmbeddingsResult, EndpointsClient, ImagesResult
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

VERTEX_LOCATION = "us-central1"

MIN_AUDIO_BYTES = 4096
MIN_IMAGE_BYTES = 1024

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"IEND\xaeB`\x82"
PNG_MAX_DIMENSION = 16384

WAV_PCM_FORMATS = (1, 3, 0xFFFE)
WAV_BIT_DEPTHS = (8, 16, 24, 32)
WAV_MIN_SAMPLE_RATE = 8000
WAV_MAX_SAMPLE_RATE = 192000
WAV_MAX_CHANNELS = 8
WAV_MIN_DATA_BYTES = 1024


def _wav_fmt_defect(raw: bytes, at: int, size: int) -> str | None:
    if size < 16 or at + 16 > len(raw):
        return f"fmt chunk declares {size} bytes, too few to describe a stream"
    audio_format = int.from_bytes(raw[at : at + 2], "little")
    channels = int.from_bytes(raw[at + 2 : at + 4], "little")
    sample_rate = int.from_bytes(raw[at + 4 : at + 8], "little")
    byte_rate = int.from_bytes(raw[at + 8 : at + 12], "little")
    block_align = int.from_bytes(raw[at + 12 : at + 14], "little")
    bits_per_sample = int.from_bytes(raw[at + 14 : at + 16], "little")
    if audio_format not in WAV_PCM_FORMATS:
        return f"fmt chunk declares audio format {audio_format}, which is not PCM"
    if not 1 <= channels <= WAV_MAX_CHANNELS:
        return f"fmt chunk declares {channels} channels"
    if not WAV_MIN_SAMPLE_RATE <= sample_rate <= WAV_MAX_SAMPLE_RATE:
        return f"fmt chunk declares a {sample_rate} Hz sample rate"
    if bits_per_sample not in WAV_BIT_DEPTHS:
        return f"fmt chunk declares {bits_per_sample} bits per sample"
    expected_align = channels * bits_per_sample // 8
    if block_align != expected_align:
        return (
            f"fmt chunk declares a {block_align} byte block for {channels} channels "
            f"of {bits_per_sample} bits, which needs {expected_align}"
        )
    if byte_rate != sample_rate * expected_align:
        return (
            f"fmt chunk declares {byte_rate} bytes per second, but {sample_rate} Hz "
            f"at {expected_align} bytes per frame is {sample_rate * expected_align}"
        )
    return None


def _wav_defect(raw: bytes) -> str | None:
    """The self-consistency a real RIFF/WAVE clip has and a malformed blob does not:
    the container's declared sizes agree with the bytes that arrived, every chunk
    fits inside the payload, the chunk list consumes the payload exactly, the fmt
    chunk describes a stream whose rates multiply out, and a data chunk actually
    carries samples. The chunk list is walked rather than probed at fixed offsets,
    since a valid file may carry LIST or fact chunks before its data.

    Each chunk's pad byte to the next even boundary is part of the chunk, so a walk
    that lands one byte past the end has read a final odd-length chunk whose trailing
    pad was omitted at EOF, which carries no samples and is left to pass. Landing
    short is different: 1 to 7 bytes cannot begin another chunk, so the response was
    cut off mid-header."""
    size = len(raw)
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return f"not a RIFF/WAVE container; first bytes are {raw[:12]!r}"
    riff_size = int.from_bytes(raw[4:8], "little")
    if riff_size != size - 8:
        return (
            f"RIFF declares {riff_size} bytes but {size - 8} arrived after the "
            f"8 byte header, so the payload is truncated or padded"
        )
    seen: tuple[bytes, ...] = ()
    data_bytes = 0
    offset = 12
    while offset + 8 <= size:
        chunk_id = raw[offset : offset + 4]
        chunk_size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        payload_at = offset + 8
        if payload_at + chunk_size > size:
            return (
                f"{chunk_id!r} chunk declares {chunk_size} bytes with only "
                f"{size - payload_at} left in the response, so it overruns the end"
            )
        if chunk_id == b"fmt ":
            defect = _wav_fmt_defect(raw, payload_at, chunk_size)
            if defect is not None:
                return defect
        elif chunk_id == b"data":
            if chunk_size < WAV_MIN_DATA_BYTES:
                return f"data chunk carries only {chunk_size} bytes of samples"
            data_bytes = chunk_size
        seen = (*seen, chunk_id)
        offset = payload_at + chunk_size + chunk_size % 2
    if b"fmt " not in seen:
        return f"no fmt chunk in the chunk list {seen!r}"
    if data_bytes == 0:
        return f"no data chunk in the chunk list {seen!r}, so it carries no audio"
    trailing = size - offset
    if trailing > 0:
        return (
            f"the response carries {trailing} more byte{'s' if trailing > 1 else ''} "
            f"than the chunk list {seen!r} accounts for, too few to form another "
            f"chunk header, so it was cut off mid-chunk"
        )
    return None


def _png_defect(raw: bytes) -> str | None:
    """A PNG states its own dimensions in the IHDR chunk that must open the file and
    ends with the IEND marker, so a truncated or corrupted image fails one of them."""
    if not raw.startswith(PNG_MAGIC):
        return f"not a PNG; first bytes are {raw[:12]!r}"
    if len(raw) < 33:
        return f"PNG stops after {len(raw)} bytes, before IHDR ends"
    ihdr_size = int.from_bytes(raw[8:12], "big")
    if raw[12:16] != b"IHDR" or ihdr_size != 13:
        return f"PNG signature is not followed by a 13 byte IHDR; found {raw[12:16]!r}"
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    if not 0 < width <= PNG_MAX_DIMENSION or not 0 < height <= PNG_MAX_DIMENSION:
        return f"IHDR declares implausible dimensions {width}x{height}"
    if not raw.endswith(PNG_IEND):
        return (
            f"PNG does not end with IEND, so the image is truncated; last bytes are "
            f"{raw[-8:]!r}"
        )
    return None


def _assert_real_vector(body: str) -> None:
    parsed = EmbeddingsResult.model_validate_json(body)
    assert parsed.first_vector, f"/embeddings returned no vector: {body[:300]}"
    assert any(component != 0.0 for component in parsed.first_vector), (
        f"embedding vector is all zeros: {body[:300]}"
    )


def _assert_real_audio(result: StreamingResponse) -> None:
    assert "audio" in (result.content_type or ""), (
        f"/audio/speech content-type is not audio: {result.content_type!r}"
    )
    assert len(result.content) >= MIN_AUDIO_BYTES, (
        f"/audio/speech returned only {len(result.content)} bytes, too short to be "
        f"spoken audio"
    )
    defect = _wav_defect(result.content)
    assert defect is None, (
        f"/audio/speech body is not a coherent WAV: {defect}. Vertex synthesizes "
        f"LINEAR16; a deployment returning another container needs that container's "
        f"structural check added here"
    )


def _assert_real_image(body: str) -> None:
    parsed = ImagesResult.model_validate_json(body)
    assert parsed.data, f"/images/generations returned no data: {body[:300]}"
    first = parsed.data[0]
    assert first.b64_json, (
        f"generated image carries no b64_json (url={first.url!r}), so its bytes "
        f"cannot be checked"
    )
    raw = base64.b64decode(first.b64_json)
    assert len(raw) >= MIN_IMAGE_BYTES, (
        f"generated image decodes to only {len(raw)} bytes, too small to be a picture"
    )
    defect = _png_defect(raw)
    assert defect is None, f"generated image is not a coherent PNG: {defect}"


class TestCohereEmbeddings:
    @pytest.mark.covers(
        "llm.embeddings.cohere.basic.nonstream.works", exercised_on=["embeddings"]
    )
    def test_cohere_embeddings_returns_vector(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-cohere-embeddings-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="cohere/embed-v4.0", api_key="os.environ/COHERE_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key(models=[model])

        result = endpoints_client.embeddings(key, model, "Say this is a test!")
        require_successful_call(result)
        _assert_real_vector(result.body)


class TestVertexNonConversational:
    def _register(
        self,
        endpoints_client: EndpointsClient,
        resources: ResourceManager,
        name: str,
        backend: str,
    ) -> tuple[str, str]:
        model = f"e2e-{name}-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model=backend,
                vertex_project="os.environ/VERTEXAI_PROJECT",
                vertex_location=VERTEX_LOCATION,
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        return model, resources.key(models=[model])

    @pytest.mark.covers(
        "llm.audio_speech.vertex.basic.nonstream.works", exercised_on=["audio_speech"]
    )
    def test_vertex_audio_speech_returns_audio(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = self._register(
            endpoints_client, resources, "vertex-speech", "vertex_ai/chirp"
        )

        result = endpoints_client.audio_speech(key, model, "Hello!")
        require_successful_call(result)
        _assert_real_audio(result)

    @pytest.mark.covers(
        "llm.images_generations.vertex.basic.nonstream.works",
        exercised_on=["images_generations"],
    )
    def test_vertex_image_generation_returns_image(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = self._register(
            endpoints_client, resources, "vertex-image", "vertex_ai/gemini-2.5-flash-image"
        )

        result = endpoints_client.images(key, model, "Draw a cute cat")
        require_successful_call(result)
        _assert_real_image(result.body)
