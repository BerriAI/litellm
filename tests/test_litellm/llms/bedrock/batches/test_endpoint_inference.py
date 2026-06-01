"""
Tests for BedrockBatchesConfig endpoint propagation.

Before these changes, both `transform_create_batch_response` and
`transform_retrieve_batch_response` reported `endpoint="/v1/chat/completions"`
unconditionally, which mis-labelled embedding batches and broke OpenAI
clients that introspect the batch object.

The new helper `_infer_openai_endpoint_from_model_id` is a best-effort
mapper used as a fallback when the caller didn't (or couldn't) pass the
original `endpoint` through `litellm_params["original_batch_request"]`.
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig


class TestInferOpenaiEndpointFromModelId:
    """Direct tests for the static inference helper."""

    def test_titan_v2_text_embed_routes_to_embeddings(self):
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "amazon.titan-embed-text-v2:0"
            )
            == "/v1/embeddings"
        )

    def test_titan_multimodal_embed_routes_to_embeddings(self):
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "amazon.titan-embed-image-v1"
            )
            == "/v1/embeddings"
        )

    def test_nova_multimodal_embed_routes_to_embeddings(self):
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "amazon.nova-2-multimodal-embeddings-v1:0"
            )
            == "/v1/embeddings"
        )

    def test_cohere_embed_routes_to_embeddings(self):
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "cohere.embed-english-v3"
            )
            == "/v1/embeddings"
        )

    def test_arn_form_embed_routes_to_embeddings(self):
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "arn:aws:bedrock:us-east-1:123:foundation-model/amazon.titan-embed-text-v2:0"
            )
            == "/v1/embeddings"
        )

    def test_anthropic_chat_routes_to_chat_completions(self):
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "anthropic.claude-3-5-sonnet-20240620-v1:0"
            )
            == "/v1/chat/completions"
        )

    def test_cross_region_inference_profile_chat_routes_to_chat(self):
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "us.anthropic.claude-haiku-4-5-20251001-v1:0"
            )
            == "/v1/chat/completions"
        )

    def test_nova_chat_routes_to_chat_completions(self):
        # Nova Pro is a chat model, not an embedding model; the name has
        # no "embed" so we must NOT misclassify it.
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "us.amazon.nova-pro-v1:0"
            )
            == "/v1/chat/completions"
        )

    def test_none_model_defaults_to_chat(self):
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(None)
            == "/v1/chat/completions"
        )

    def test_empty_model_defaults_to_chat(self):
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id("")
            == "/v1/chat/completions"
        )

    def test_case_insensitive_matching(self):
        # AWS model ids are lowercase but defensive coercion shouldn't hurt.
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "AMAZON.TITAN-EMBED-TEXT-V2:0"
            )
            == "/v1/embeddings"
        )


class TestTransformCreateBatchResponseEndpoint:
    """End-to-end checks that the create-batch response carries the right endpoint."""

    @staticmethod
    def _mock_raw_response(payload):
        m = MagicMock()
        m.json.return_value = payload
        return m

    def test_create_response_uses_original_request_endpoint_when_set(self):
        """Explicit endpoint on the original request wins over the heuristic."""
        config = BedrockBatchesConfig()
        raw = self._mock_raw_response(
            {
                "jobArn": "arn:aws:bedrock:us-east-1:123:model-invocation-job/abc",
                "status": "Submitted",
            }
        )
        batch = config.transform_create_batch_response(
            model="amazon.titan-embed-text-v2:0",
            raw_response=raw,
            logging_obj=MagicMock(),
            litellm_params={"original_batch_request": {"endpoint": "/v1/embeddings"}},
        )
        assert batch.endpoint == "/v1/embeddings"

    def test_create_response_falls_back_to_model_inference_for_embed(self):
        """Without `original_batch_request.endpoint`, infer from model id."""
        config = BedrockBatchesConfig()
        raw = self._mock_raw_response(
            {
                "jobArn": "arn:aws:bedrock:us-east-1:123:model-invocation-job/abc",
                "status": "Submitted",
            }
        )
        batch = config.transform_create_batch_response(
            model="amazon.titan-embed-text-v2:0",
            raw_response=raw,
            logging_obj=MagicMock(),
            litellm_params={},
        )
        assert batch.endpoint == "/v1/embeddings"

    def test_create_response_chat_default_when_no_signal(self):
        """No endpoint and no embed marker in model id -> chat completions."""
        config = BedrockBatchesConfig()
        raw = self._mock_raw_response(
            {
                "jobArn": "arn:aws:bedrock:us-east-1:123:model-invocation-job/abc",
                "status": "Submitted",
            }
        )
        batch = config.transform_create_batch_response(
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            raw_response=raw,
            logging_obj=MagicMock(),
            litellm_params={},
        )
        assert batch.endpoint == "/v1/chat/completions"

    def test_create_response_chat_when_model_is_none_but_request_says_chat(self):
        """If model is None but original request says chat, that wins."""
        config = BedrockBatchesConfig()
        raw = self._mock_raw_response(
            {
                "jobArn": "arn:aws:bedrock:us-east-1:123:model-invocation-job/abc",
                "status": "Submitted",
            }
        )
        batch = config.transform_create_batch_response(
            model=None,
            raw_response=raw,
            logging_obj=MagicMock(),
            litellm_params={
                "original_batch_request": {"endpoint": "/v1/chat/completions"}
            },
        )
        assert batch.endpoint == "/v1/chat/completions"

    def test_create_response_preserves_explicit_empty_endpoint(self):
        """Explicit empty-string endpoint is preserved, not replaced by heuristic.

        Pins the `is not None` semantics: only a missing `endpoint` key
        falls through to the model-id heuristic. An explicitly-provided
        empty string keeps that explicit signal so an upstream caller
        that means 'no endpoint' isn't silently overridden.
        """
        config = BedrockBatchesConfig()
        raw = self._mock_raw_response(
            {
                "jobArn": "arn:aws:bedrock:us-east-1:123:model-invocation-job/abc",
                "status": "Submitted",
            }
        )
        batch = config.transform_create_batch_response(
            model="amazon.titan-embed-text-v2:0",
            raw_response=raw,
            logging_obj=MagicMock(),
            litellm_params={"original_batch_request": {"endpoint": ""}},
        )
        assert batch.endpoint == ""


class TestTransformRetrieveBatchResponseEndpoint:
    """The retrieve path has no original request to read from, so it must
    infer the endpoint from the AWS response's modelId."""

    @staticmethod
    def _mock_raw_response(payload):
        m = MagicMock()
        m.json.return_value = payload
        m.status_code = 200
        return m

    def test_retrieve_response_infers_embeddings_from_modelid(self):
        config = BedrockBatchesConfig()
        raw = self._mock_raw_response(
            {
                "jobArn": "arn:aws:bedrock:us-east-1:123:model-invocation-job/abc",
                "status": "Completed",
                "modelId": "amazon.titan-embed-text-v2:0",
                "inputDataConfig": {"s3InputDataConfig": {"s3Uri": "s3://b/in.jsonl"}},
                "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://b/out/"}},
            }
        )
        batch = config.transform_retrieve_batch_response(
            model=None,
            raw_response=raw,
            logging_obj=MagicMock(),
            litellm_params={},
        )
        assert batch.endpoint == "/v1/embeddings"

    def test_retrieve_response_infers_chat_from_modelid(self):
        config = BedrockBatchesConfig()
        raw = self._mock_raw_response(
            {
                "jobArn": "arn:aws:bedrock:us-east-1:123:model-invocation-job/abc",
                "status": "Completed",
                "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "inputDataConfig": {"s3InputDataConfig": {"s3Uri": "s3://b/in.jsonl"}},
                "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://b/out/"}},
            }
        )
        batch = config.transform_retrieve_batch_response(
            model=None,
            raw_response=raw,
            logging_obj=MagicMock(),
            litellm_params={},
        )
        assert batch.endpoint == "/v1/chat/completions"

    def test_retrieve_response_model_arg_wins_over_response_modelid(self):
        """If the caller passed `model`, prefer it over the AWS payload."""
        config = BedrockBatchesConfig()
        raw = self._mock_raw_response(
            {
                "jobArn": "arn:aws:bedrock:us-east-1:123:model-invocation-job/abc",
                "status": "Completed",
                # AWS-reported model is chat; caller-supplied model is embed.
                "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "inputDataConfig": {"s3InputDataConfig": {"s3Uri": "s3://b/in.jsonl"}},
                "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://b/out/"}},
            }
        )
        batch = config.transform_retrieve_batch_response(
            model="amazon.titan-embed-text-v2:0",
            raw_response=raw,
            logging_obj=MagicMock(),
            litellm_params={},
        )
        assert batch.endpoint == "/v1/embeddings"

    def test_retrieve_response_defaults_to_chat_when_no_modelid(self):
        config = BedrockBatchesConfig()
        raw = self._mock_raw_response(
            {
                "jobArn": "arn:aws:bedrock:us-east-1:123:model-invocation-job/abc",
                "status": "Submitted",
                "inputDataConfig": {"s3InputDataConfig": {"s3Uri": "s3://b/in.jsonl"}},
                "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": "s3://b/out/"}},
            }
        )
        batch = config.transform_retrieve_batch_response(
            model=None,
            raw_response=raw,
            logging_obj=MagicMock(),
            litellm_params={},
        )
        assert batch.endpoint == "/v1/chat/completions"


class TestRegistryDrivenClassification:
    """The model registry (`model_prices_and_context_window.json`) is the
    source of truth - these tests pin that we use it, not just substring
    matching, for canonical Bedrock model ids."""

    def test_registry_classifies_known_embed_id(self, mocker):
        """get_model_info short-circuits the substring fallback for known ids."""
        mocked = mocker.patch(
            "litellm.get_model_info", return_value={"mode": "embedding"}
        )
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "amazon.titan-embed-text-v2:0"
            )
            == "/v1/embeddings"
        )
        mocked.assert_called_once_with("amazon.titan-embed-text-v2:0")

    def test_registry_chat_mode_wins_even_if_name_contains_embed_word(self, mocker):
        """Hypothetical model whose name has 'embed' but registry says chat
        must NOT route to embeddings - registry wins over substring."""
        mocker.patch("litellm.get_model_info", return_value={"mode": "chat"})
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "some.embed-themed-chat-v1:0"
            )
            == "/v1/chat/completions"
        )

    def test_substring_fallback_used_when_registry_raises(self, mocker):
        """For unmapped ids (ARN forms, cross-region profiles) we keep the
        old substring heuristic so embedding batches still get routed."""
        mocker.patch("litellm.get_model_info", side_effect=Exception("not mapped"))
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "us.amazon.titan-embed-text-v2:0"
            )
            == "/v1/embeddings"
        )

    def test_substring_fallback_chat_default_when_registry_raises(self, mocker):
        """Unmapped id without 'embed' marker -> chat (preserves prior behavior)."""
        mocker.patch("litellm.get_model_info", side_effect=Exception("not mapped"))
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "arn:aws:bedrock:us-east-1:123:custom-model/some-llm"
            )
            == "/v1/chat/completions"
        )

    def test_registry_no_mode_field_falls_back_to_substring(self, mocker):
        """Registry returns info dict without a `mode` key -> substring path."""
        mocker.patch("litellm.get_model_info", return_value={})
        # No 'embed' in name -> chat
        assert (
            BedrockBatchesConfig._infer_openai_endpoint_from_model_id(
                "anthropic.claude-3-5-sonnet-20240620-v1:0"
            )
            == "/v1/chat/completions"
        )


class TestLookupRegistryMode:
    """Direct tests for the extracted `_lookup_registry_mode` helper.

    Pinning this separately lets us assert the registry-vs-fallback split
    in isolation, instead of inferring it from end-to-end endpoint output.
    """

    def test_returns_mode_when_registry_resolves(self, mocker):
        mocker.patch("litellm.get_model_info", return_value={"mode": "embedding"})
        assert (
            BedrockBatchesConfig._lookup_registry_mode("amazon.titan-embed-text-v2:0")
            == "embedding"
        )

    def test_returns_none_when_registry_raises(self, mocker):
        mocker.patch("litellm.get_model_info", side_effect=Exception("not mapped"))
        assert BedrockBatchesConfig._lookup_registry_mode("anything") is None

    def test_returns_none_when_registry_returns_non_dict(self, mocker):
        # Defensive: get_model_info isn't supposed to return non-dict but if
        # it ever does we treat it the same as 'not mapped' rather than
        # crashing the batch transformer.
        mocker.patch("litellm.get_model_info", return_value="not a dict")
        assert BedrockBatchesConfig._lookup_registry_mode("anything") is None

    def test_returns_none_when_mode_missing(self, mocker):
        mocker.patch("litellm.get_model_info", return_value={})
        assert BedrockBatchesConfig._lookup_registry_mode("anything") is None

    def test_returns_none_when_mode_is_empty_string(self, mocker):
        mocker.patch("litellm.get_model_info", return_value={"mode": ""})
        assert BedrockBatchesConfig._lookup_registry_mode("anything") is None

    def test_returns_none_when_mode_is_non_string(self, mocker):
        # If a malformed registry entry has a non-string mode we don't trust
        # it and fall through to the substring heuristic.
        mocker.patch("litellm.get_model_info", return_value={"mode": 42})
        assert BedrockBatchesConfig._lookup_registry_mode("anything") is None
