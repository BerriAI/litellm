import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
import litellm.main
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.multimodal_embeddings.transformation import (
    VertexAIMultimodalEmbeddingConfig,
)
from litellm.types.llms.vertex_ai import Instance, InstanceImage, InstanceVideo


class TestVertexMultimodalEmbedding:
    def setup_method(self):
        self.config = VertexAIMultimodalEmbeddingConfig()

    def test_process_openai_embedding_input(self):
        input_data = [
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=",
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=",
        ]
        expected_output = [
            Instance(
                image=InstanceImage(bytesBase64Encoded=input_data[0].split(",")[1])
            ),
            Instance(
                image=InstanceImage(bytesBase64Encoded=input_data[1].split(",")[1])
            ),
        ]
        assert self.config._process_input_element(input_data[0]) == expected_output[0]
        assert self.config._process_input_element(input_data[1]) == expected_output[1]

    def test_process_str_and_video_input(self):
        input_data = ["hi", "gs://my-bucket/embeddings/supermarket-video.mp4"]
        expected_output = [
            Instance(
                text="hi",
                video=InstanceVideo(
                    gcsUri="gs://my-bucket/embeddings/supermarket-video.mp4"
                ),
            ),
        ]
        assert self.config.process_openai_embedding_input(input_data) == expected_output

    def test_process_list_of_str_and_str_input(self):
        input_data = ["hi", "hello"]
        expected_output = [
            Instance(text="hi"),
            Instance(text="hello"),
        ]
        assert self.config.process_openai_embedding_input(input_data) == expected_output

    def test_process_list_of_str_and_video_input(self):
        input_data = [
            "hi",
            "hello",
            "gs://my-bucket/embeddings/supermarket-video.mp4",
            "hey",
        ]
        expected_output = [
            Instance(text="hi"),
            Instance(
                text="hello",
                video=InstanceVideo(
                    gcsUri="gs://my-bucket/embeddings/supermarket-video.mp4"
                ),
            ),
            Instance(text="hey"),
        ]
        assert (
            self.config.process_openai_embedding_input(input_data) == expected_output
        ), f"Expected {expected_output}, but got {self.config.process_openai_embedding_input(input_data)}"

    def test_process_text_and_base64_image_input(self):
        """Test that text + base64 image combinations are correctly merged into a single instance."""
        base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
        input_data = ["describe this image", base64_image]
        expected_output = [
            Instance(
                text="describe this image",
                image=InstanceImage(bytesBase64Encoded=base64_image.split(",")[1]),
            ),
        ]
        result = self.config.process_openai_embedding_input(input_data)
        assert (
            result == expected_output
        ), f"Expected {expected_output}, but got {result}"

    def test_process_multiple_text_and_base64_image_pairs(self):
        """Test multiple text + base64 image pairs in a single request."""
        base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
        input_data = [
            "first description",
            base64_image,
            "second description",
            base64_image,
        ]
        expected_output = [
            Instance(
                text="first description",
                image=InstanceImage(bytesBase64Encoded=base64_image.split(",")[1]),
            ),
            Instance(
                text="second description",
                image=InstanceImage(bytesBase64Encoded=base64_image.split(",")[1]),
            ),
        ]
        result = self.config.process_openai_embedding_input(input_data)
        assert (
            result == expected_output
        ), f"Expected {expected_output}, but got {result}"

    def test_process_base64_image_only_in_list(self):
        """Test that standalone base64 images in a list are processed correctly."""
        base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
        input_data = [base64_image, base64_image]
        expected_output = [
            Instance(
                image=InstanceImage(bytesBase64Encoded=base64_image.split(",")[1])
            ),
            Instance(
                image=InstanceImage(bytesBase64Encoded=base64_image.split(",")[1])
            ),
        ]
        result = self.config.process_openai_embedding_input(input_data)
        assert (
            result == expected_output
        ), f"Expected {expected_output}, but got {result}"

    def test_process_text_and_gcs_image_input(self):
        """Test that text + GCS image combinations are correctly merged."""
        gcs_uri = "gs://my-bucket/image.png"
        input_data = ["describe this image", gcs_uri]
        expected_output = [
            Instance(
                text="describe this image",
                image=InstanceImage(gcsUri=gcs_uri),
            ),
        ]
        result = self.config.process_openai_embedding_input(input_data)
        assert (
            result == expected_output
        ), f"Expected {expected_output}, but got {result}"

    def test_dimensions_parameter_forwarded_as_dimension(self):
        """Test that the dimensions parameter is forwarded as dimension
        in the request body parameters field.

        Fixes: https://github.com/BerriAI/litellm/issues/24392
        """
        optional_params = {}
        optional_params = self.config.map_openai_params(
            non_default_params={"dimensions": 128},
            optional_params=optional_params,
            model="multimodalembedding",
            drop_params=False,
        )
        request = self.config.transform_embedding_request(
            model="multimodalembedding",
            input="Hello world",
            optional_params=optional_params,
            headers={},
        )
        assert "parameters" in request, "Request should contain 'parameters' field"
        assert request["parameters"]["dimension"] == 128

    def test_no_parameters_field_when_dimensions_not_set(self):
        """Test that no parameters field is added when dimensions is not set."""
        request = self.config.transform_embedding_request(
            model="multimodalembedding",
            input="Hello world",
            optional_params={},
            headers={},
        )
        assert "parameters" not in request


LABEL_FIELDS: Final = ("user_api_key_alias", "user_api_key_team_alias", "spend_logs_metadata")
IDENTITY: Final = MappingProxyType({"user_api_key_alias": "chargeback-key", "user_api_key_team_alias": "ai-platform"})
PROXY_METADATA: Final = MappingProxyType({**IDENTITY, "spend_logs_metadata": {"cost-center": "cc-1"}})
FORGED: Final = MappingProxyType({"user_api_key_alias": "forged", "app": "bot"})
TODAYS_INSTANCES: Final = [{"text": "a red apple"}]
SENT_BODY: Final = TypeAdapter(Mapping[str, object])
SENT_LABELS: Final = TypeAdapter(Mapping[str, str])


async def multimodal_embedding_body(
    is_async: bool, monkeypatch: pytest.MonkeyPatch, request_labels: Mapping[str, object]
) -> Mapping[str, object]:
    reply: Final = MagicMock(
        side_effect=lambda _request: httpx.Response(200, json={"predictions": [{"textEmbedding": [0.1, 0.2]}]})
    )
    monkeypatch.setattr(litellm.main.vertex_multimodal_embedding, "_ensure_access_token", lambda **_: ("token", "proj"))
    transport: Final = httpx.MockTransport(reply)
    embed: Final = {
        "model": "vertex_ai/multimodalembedding@001",
        "input": "a red apple",
        "vertex_location": "us-central1",
        "metadata": dict(PROXY_METADATA),
        **request_labels,
    }
    if is_async:
        await litellm.aembedding(**embed, client=AsyncHTTPHandler(transport=transport))
    else:
        litellm.embedding(**embed, client=HTTPHandler(client=httpx.Client(transport=transport)))

    sent: Final[httpx.Request] = reply.call_args.args[0]
    return SENT_BODY.validate_json(sent.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("request_labels", [{}, {"labels": dict(FORGED)}], ids=["no-caller-labels", "caller-labels"])
async def test_setting_off_sends_todays_multimodal_embedding_body(
    is_async: bool, request_labels: Mapping[str, object], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", None)

    assert await multimodal_embedding_body(is_async, monkeypatch, request_labels) == {"instances": TODAYS_INSTANCES}


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "request_labels, expected_labels",
    [
        ({}, [*IDENTITY.items(), ("cost-center", "cc-1")]),
        ({"labels": dict(FORGED)}, [*IDENTITY.items(), ("app", "bot"), ("cost-center", "cc-1")]),
    ],
    ids=["no-caller-labels", "forged-caller-labels"],
)
async def test_multimodal_embedding_sends_identity_labels_a_caller_cannot_forge(
    is_async: bool,
    request_labels: Mapping[str, object],
    expected_labels: Sequence[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", LABEL_FIELDS)

    body: Final = await multimodal_embedding_body(is_async, monkeypatch, request_labels)

    assert body["instances"] == TODAYS_INSTANCES
    assert list(SENT_LABELS.validate_python(body["labels"]).items()) == list(expected_labels)
