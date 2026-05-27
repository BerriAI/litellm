"""
Regression tests for the Bedrock Nova multimodal embedding video data-URL
path: ``_transform_request()`` must add the Nova-required ``embeddingMode``
field to the synthesized ``video`` block, both for synchronous
(``SINGLE_EMBEDDING``) and asynchronous / segmented (``SEGMENTED_EMBEDDING``)
requests.
"""

import base64

from litellm.llms.bedrock.embed.amazon_nova_transformation import (
    AmazonNovaEmbeddingConfig,
)


def _video_data_url() -> str:
    b64 = base64.b64encode(b"FAKE_MP4_BYTES_FOR_TEST").decode()
    return f"data:video/mp4;base64,{b64}"


class TestBedrockNovaVideoEmbeddingMode:
    def test_sync_video_data_url_defaults_embedding_mode(self):
        cfg = AmazonNovaEmbeddingConfig()
        req = cfg._transform_request(
            input=_video_data_url(),
            inference_params={"embeddingPurpose": "VIDEO_RETRIEVAL"},
            async_invoke_route=False,
        )
        video = req["singleEmbeddingParams"]["video"]
        assert video["format"] == "mp4"
        assert "bytes" in video["source"]
        assert video["embeddingMode"] == "AUDIO_VIDEO_COMBINED"

    def test_caller_can_override_embedding_mode(self):
        cfg = AmazonNovaEmbeddingConfig()
        req = cfg._transform_request(
            input=_video_data_url(),
            inference_params={
                "embeddingPurpose": "VIDEO_RETRIEVAL",
                "embeddingMode": "AUDIO_VIDEO_SEPARATE",
            },
            async_invoke_route=False,
        )
        video = req["singleEmbeddingParams"]["video"]
        assert video["embeddingMode"] == "AUDIO_VIDEO_SEPARATE"
        assert "embeddingMode" not in req["singleEmbeddingParams"]

    def test_async_segmented_video_data_url_defaults_embedding_mode(self):
        cfg = AmazonNovaEmbeddingConfig()
        req = cfg._transform_request(
            input=_video_data_url(),
            inference_params={"embeddingPurpose": "VIDEO_RETRIEVAL"},
            async_invoke_route=True,
            model_id="amazon.nova-2-multimodal-embeddings-v1",
            output_s3_uri="s3://bucket/out/",
        )
        video = req["modelInput"]["segmentedEmbeddingParams"]["video"]
        assert video["format"] == "mp4"
        assert video["embeddingMode"] == "AUDIO_VIDEO_COMBINED"

    def test_user_supplied_video_dict_preserved(self):
        cfg = AmazonNovaEmbeddingConfig()
        req = cfg._transform_request(
            input="ignored",
            inference_params={
                "embeddingPurpose": "VIDEO_RETRIEVAL",
                "video": {
                    "format": "mp4",
                    "source": {"s3Location": {"uri": "s3://x/y.mp4"}},
                    "embeddingMode": "AUDIO_VIDEO_SEPARATE",
                },
            },
            async_invoke_route=False,
        )
        video = req["singleEmbeddingParams"]["video"]
        assert video["embeddingMode"] == "AUDIO_VIDEO_SEPARATE"
        assert video["source"]["s3Location"]["uri"] == "s3://x/y.mp4"

    def test_user_supplied_video_without_embedding_mode_gets_default(self):
        """Pre-populated ``video`` dict missing ``embeddingMode`` still gets
        the default (regression for the Greptile P2 gap)."""
        cfg = AmazonNovaEmbeddingConfig()
        req = cfg._transform_request(
            input="ignored",
            inference_params={
                "embeddingPurpose": "VIDEO_RETRIEVAL",
                "video": {
                    "format": "mp4",
                    "source": {"s3Location": {"uri": "s3://x/y.mp4"}},
                },
            },
            async_invoke_route=False,
        )
        video = req["singleEmbeddingParams"]["video"]
        assert video["embeddingMode"] == "AUDIO_VIDEO_COMBINED"

    def test_user_supplied_video_without_embedding_mode_respects_override(self):
        """Top-level ``embeddingMode`` override applies to a pre-populated
        ``video`` dict that omits the field."""
        cfg = AmazonNovaEmbeddingConfig()
        req = cfg._transform_request(
            input="ignored",
            inference_params={
                "embeddingPurpose": "VIDEO_RETRIEVAL",
                "embeddingMode": "AUDIO_VIDEO_SEPARATE",
                "video": {
                    "format": "mp4",
                    "source": {"s3Location": {"uri": "s3://x/y.mp4"}},
                },
            },
            async_invoke_route=False,
        )
        video = req["singleEmbeddingParams"]["video"]
        assert video["embeddingMode"] == "AUDIO_VIDEO_SEPARATE"
        assert "embeddingMode" not in req["singleEmbeddingParams"]

    def test_non_video_paths_unchanged(self):
        cfg = AmazonNovaEmbeddingConfig()
        b64 = base64.b64encode(b"BYTES").decode()
        img = cfg._transform_request(
            input=f"data:image/png;base64,{b64}",
            inference_params={"embeddingPurpose": "IMAGE_RETRIEVAL"},
            async_invoke_route=False,
        )
        assert "embeddingMode" not in img["singleEmbeddingParams"]["image"]
        aud = cfg._transform_request(
            input=f"data:audio/mp3;base64,{b64}",
            inference_params={"embeddingPurpose": "AUDIO_RETRIEVAL"},
            async_invoke_route=False,
        )
        assert "embeddingMode" not in aud["singleEmbeddingParams"]["audio"]
        txt = cfg._transform_request(
            input="hello world",
            inference_params={"embeddingPurpose": "TEXT_RETRIEVAL"},
            async_invoke_route=False,
        )
        assert "text" in txt["singleEmbeddingParams"]

    def test_batch_does_not_leak_embedding_mode_or_mutate_inference_params(self):
        """
        Regression for Greptile P2 on PR #29026: batched embedding calls.

        ``embedding()`` calls ``_transform_request`` once per input in the
        list, passing the *same* ``inference_params`` dict each time. The
        pre-populated ``video`` block must not be mutated in-place, and a
        top-level ``embeddingMode`` override must not leak into
        ``singleEmbeddingParams`` of subsequent batch elements.
        """
        import copy
        cfg = AmazonNovaEmbeddingConfig()
        inference_params = {
            "embeddingPurpose": "VIDEO_RETRIEVAL",
            "embeddingMode": "AUDIO_VIDEO_SEPARATE",
            "video": {
                "format": "mp4",
                "source": {"s3Location": {"uri": "s3://b/v.mp4"}},
            },
        }
        before = copy.deepcopy(inference_params)
        req1 = cfg._transform_request(input="s3://b/v.mp4", inference_params=inference_params, async_invoke_route=False)
        req2 = cfg._transform_request(input="s3://b/v.mp4", inference_params=inference_params, async_invoke_route=False)
        for r in (req1, req2):
            p = r["singleEmbeddingParams"]
            assert p["video"]["embeddingMode"] == "AUDIO_VIDEO_SEPARATE"
            assert "embeddingMode" not in p
        assert inference_params == before

    def test_batch_data_url_override_does_not_leak(self):
        """Same as above but for the data-URL synthesis path."""
        import copy
        cfg = AmazonNovaEmbeddingConfig()
        inference_params = {
            "embeddingPurpose": "VIDEO_RETRIEVAL",
            "embeddingMode": "AUDIO_VIDEO_SEPARATE",
        }
        before = copy.deepcopy(inference_params)
        req1 = cfg._transform_request(input=_video_data_url(), inference_params=inference_params, async_invoke_route=False)
        req2 = cfg._transform_request(input=_video_data_url(), inference_params=inference_params, async_invoke_route=False)
        for r in (req1, req2):
            p = r["singleEmbeddingParams"]
            assert p["video"]["embeddingMode"] == "AUDIO_VIDEO_SEPARATE"
            assert "embeddingMode" not in p
        assert inference_params == before

