import json

import pytest

from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.embed.twelvelabs_marengo_3_transformation import (
    build_marengo_3_request,
    is_marengo_3_model,
)
from litellm.llms.bedrock.embed.twelvelabs_marengo_transformation import (
    TwelveLabsMarengoEmbeddingConfig,
)

MARENGO_3_BASE = "twelvelabs.marengo-embed-3-0-v1:0"
MARENGO_3_US = "us.twelvelabs.marengo-embed-3-0-v1:0"
MARENGO_27_US = "us.twelvelabs.marengo-embed-2-7-v1:0"
DUCK_DATA_URL = "data:image/png;base64,ZHVjaw=="
OUTPUT_S3_URI = "s3://out-bucket/marengo/"


@pytest.mark.parametrize(
    "model,expected",
    [
        (MARENGO_3_BASE, True),
        (MARENGO_3_US, True),
        ("eu.twelvelabs.marengo-embed-3-0-v1:0", True),
        ("async_invoke/twelvelabs.marengo-embed-3-0-v1:0", True),
        (MARENGO_27_US, False),
        ("twelvelabs.marengo-embed-2-7-v1:0", False),
        (None, False),
    ],
)
def test_is_marengo_3_model(model, expected):
    assert is_marengo_3_model(model) is expected


def wire(request: object) -> object:
    return json.loads(json.dumps(request))


def test_text_request_nests_input_text_under_text():
    assert build_marengo_3_request("a dog on the beach", {"input_type": "text"}) == {
        "inputType": "text",
        "text": {"inputText": "a dog on the beach"},
    }


def test_missing_input_type_defaults_to_text():
    assert build_marengo_3_request("hello", {})["inputType"] == "text"


def test_camel_case_input_type_wins_over_snake_case():
    request = build_marengo_3_request(DUCK_DATA_URL, {"inputType": "image", "input_type": "text"})
    assert request["inputType"] == "image"


def test_image_request_strips_data_url_prefix():
    assert build_marengo_3_request(DUCK_DATA_URL, {"input_type": "image"}) == {
        "inputType": "image",
        "image": {"mediaSource": {"base64String": "ZHVjaw=="}},
    }


def test_image_request_from_s3_carries_bucket_owner():
    request = build_marengo_3_request("s3://media/duck.png", {"input_type": "image", "bucketOwner": "123456789012"})
    assert request == {
        "inputType": "image",
        "image": {"mediaSource": {"s3Location": {"uri": "s3://media/duck.png", "bucketOwner": "123456789012"}}},
    }


def test_s3_media_without_bucket_owner_omits_the_key():
    request = build_marengo_3_request("s3://media/duck.png", {"input_type": "image"})
    assert request["image"]["mediaSource"] == {"s3Location": {"uri": "s3://media/duck.png"}}


def test_text_image_request_pairs_text_with_media_source():
    request = build_marengo_3_request(
        "a duck", {"input_type": "text_image", "media_source": DUCK_DATA_URL, "output_s3_uri": OUTPUT_S3_URI}
    )
    assert request == {
        "inputType": "text_image",
        "text_image": {"inputText": "a duck", "mediaSource": {"base64String": "ZHVjaw=="}},
    }


def test_text_image_request_requires_media_source():
    with pytest.raises(BedrockError, match=r"text_image.*media_source") as excinfo:
        build_marengo_3_request("a duck", {"input_type": "text_image"})
    assert excinfo.value.status_code == 400


def test_multi_input_request_names_each_media_source():
    request = build_marengo_3_request(
        "a photo of <@bird> next to <@dog>",
        {
            "input_type": "multi_input",
            "media_sources": {"bird": DUCK_DATA_URL, "dog": "s3://media/dog.png"},
            "bucketOwner": "123456789012",
        },
    )
    assert wire(request) == {
        "inputType": "multi_input",
        "multi_input": {
            "inputText": "a photo of <@bird> next to <@dog>",
            "mediaSources": [
                {"name": "bird", "mediaType": "image", "base64String": "ZHVjaw=="},
                {
                    "name": "dog",
                    "mediaType": "image",
                    "s3Location": {"uri": "s3://media/dog.png", "bucketOwner": "123456789012"},
                },
            ],
        },
    }


def test_multi_input_without_text_omits_input_text():
    request = build_marengo_3_request("", {"input_type": "multi_input", "media_sources": {"bird": DUCK_DATA_URL}})
    assert "inputText" not in request["multi_input"]
    assert request["multi_input"]["mediaSources"][0]["name"] == "bird"


@pytest.mark.parametrize("params", [{"input_type": "multi_input"}, {"input_type": "multi_input", "media_sources": {}}])
def test_multi_input_request_requires_media_sources(params):
    with pytest.raises(BedrockError, match=r"multi_input.*media_sources") as excinfo:
        build_marengo_3_request("<@bird>", params)
    assert excinfo.value.status_code == 400


@pytest.mark.parametrize("input_type", ["video", "audio"])
def test_timed_media_request_nests_every_option_under_the_media_key(input_type):
    request = build_marengo_3_request(
        "s3://media/clip.mp4",
        {
            "input_type": input_type,
            "startSec": 2,
            "endSec": 12.5,
            "segmentation": {"method": "dynamic", "dynamic": {"minDurationSec": 4}},
            "embeddingOption": ["visual", "audio"],
            "embeddingType": ["fused_embedding"],
            "embeddingScope": ["clip", "asset"],
            "inferenceId": "req-42",
        },
    )
    assert wire(request) == {
        "inputType": input_type,
        input_type: {
            "mediaSource": {"s3Location": {"uri": "s3://media/clip.mp4"}},
            "startSec": 2.0,
            "endSec": 12.5,
            "segmentation": {"method": "dynamic", "dynamic": {"minDurationSec": 4}},
            "embeddingOption": ["visual", "audio"],
            "embeddingType": ["fused_embedding"],
            "embeddingScope": ["clip", "asset"],
        },
        "inferenceId": "req-42",
    }


def test_timed_media_request_without_options_carries_only_the_media_source():
    request = build_marengo_3_request("s3://media/clip.mp4", {"input_type": "video"})
    assert request["video"] == {"mediaSource": {"s3Location": {"uri": "s3://media/clip.mp4"}}}


@pytest.mark.parametrize(
    "params",
    [
        {"input_type": "clip"},
        {"input_type": "video", "embeddingOption": ["visual-text"]},
        {"input_type": "video", "segmentation": {"method": "fixed", "dynamic": {"minDurationSec": 4}}},
        {"input_type": "multi_input", "media_sources": ["not", "a", "mapping"]},
    ],
)
def test_invalid_marengo_3_params_are_rejected_before_the_request_is_sent(params):
    with pytest.raises(BedrockError, match=r"Invalid Marengo 3\.0 parameters") as excinfo:
        build_marengo_3_request("s3://media/clip.mp4", params)
    assert excinfo.value.status_code == 400


def test_config_sends_the_nested_payload_for_marengo_3_and_the_flat_one_for_2_7():
    nested = TwelveLabsMarengoEmbeddingConfig(model=MARENGO_3_US)._transform_request(
        input="hello", inference_params={"input_type": "text"}
    )
    flat = TwelveLabsMarengoEmbeddingConfig(model=MARENGO_27_US)._transform_request(
        input="hello", inference_params={"input_type": "text"}
    )
    assert nested == {"inputType": "text", "text": {"inputText": "hello"}}
    assert flat == {"inputType": "text", "inputText": "hello", "textTruncate": "end"}


def test_config_without_a_model_keeps_the_2_7_payload():
    request = TwelveLabsMarengoEmbeddingConfig()._transform_request(input="hello", inference_params={})
    assert request == {"inputType": "text", "inputText": "hello", "textTruncate": "end"}


@pytest.mark.parametrize("input_type", ["video", "audio"])
def test_marengo_3_video_and_audio_still_require_the_async_route(input_type):
    with pytest.raises(ValueError, match=f"Input type '{input_type}' requires async_invoke route"):
        TwelveLabsMarengoEmbeddingConfig(model=MARENGO_3_BASE)._transform_request(
            input="s3://media/clip.mp4", inference_params={"input_type": input_type}
        )


def test_marengo_3_async_invoke_wraps_the_nested_payload_with_the_base_model_id():
    request = TwelveLabsMarengoEmbeddingConfig(model=MARENGO_3_BASE)._transform_request(
        input="s3://media/clip.mp4",
        inference_params={"input_type": "video", "embeddingOption": ["visual"], "output_s3_uri": OUTPUT_S3_URI},
        async_invoke_route=True,
        model_id="async_invoke%2Ftwelvelabs.marengo-embed-3-0-v1%3A0",
        output_s3_uri=OUTPUT_S3_URI,
    )
    assert wire(request) == {
        "modelId": MARENGO_3_BASE,
        "modelInput": {
            "inputType": "video",
            "video": {"mediaSource": {"s3Location": {"uri": "s3://media/clip.mp4"}}, "embeddingOption": ["visual"]},
        },
        "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": OUTPUT_S3_URI}},
    }


def test_marengo_3_async_invoke_requires_an_output_s3_uri():
    with pytest.raises(ValueError, match="output_s3_uri cannot be empty"):
        TwelveLabsMarengoEmbeddingConfig(model=MARENGO_3_BASE)._transform_request(
            input="hello",
            inference_params={"input_type": "text"},
            async_invoke_route=True,
            model_id=MARENGO_3_BASE,
            output_s3_uri="",
        )


def test_encoding_format_float_no_longer_injects_2_7_embedding_options_for_marengo_3():
    marengo_3 = TwelveLabsMarengoEmbeddingConfig(model=MARENGO_3_US).map_openai_params(
        non_default_params={"encoding_format": "float"}, optional_params={}
    )
    marengo_27 = TwelveLabsMarengoEmbeddingConfig(model=MARENGO_27_US).map_openai_params(
        non_default_params={"encoding_format": "float"}, optional_params={}
    )
    assert marengo_3 == {}
    assert marengo_27 == {"embeddingOption": ["visual-text", "visual-image"]}


def test_marengo_3_only_params_are_forwarded_by_map_openai_params():
    mapped = TwelveLabsMarengoEmbeddingConfig(model=MARENGO_3_US).map_openai_params(
        non_default_params={
            "input_type": "text_image",
            "media_source": DUCK_DATA_URL,
            "media_sources": {"bird": DUCK_DATA_URL},
            "endSec": 5,
            "segmentation": {"method": "fixed", "fixed": {"durationSec": 6}},
            "embeddingType": ["separate_embedding"],
            "embeddingScope": ["clip"],
            "inferenceId": "req-1",
        },
        optional_params={},
    )
    assert mapped == {
        "inputType": "text_image",
        "media_source": DUCK_DATA_URL,
        "media_sources": {"bird": DUCK_DATA_URL},
        "endSec": 5,
        "segmentation": {"method": "fixed", "fixed": {"durationSec": 6}},
        "embeddingType": ["separate_embedding"],
        "embeddingScope": ["clip"],
        "inferenceId": "req-1",
    }
