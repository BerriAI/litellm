"""Tests for ``encode_path_segment`` / ``encode_url_path`` and provider call sites."""

import pytest

from litellm.llms.base_llm._url_utils import encode_path_segment, encode_url_path


SAMPLE_INPUT = "../../v1/messages/batches"
SAMPLE_INPUT_ENCODED = "..%2F..%2Fv1%2Fmessages%2Fbatches"


def _assert_input_is_encoded(url: str, prefix: str) -> None:
    assert SAMPLE_INPUT_ENCODED in url
    tail = url.split(prefix, 1)[1]
    assert "../" not in tail, f"raw ``../`` after {prefix!r}: {tail!r}"


class TestEncodePathSegment:
    def test_encodes_dot_slash_sequences(self):
        assert encode_path_segment(SAMPLE_INPUT) == SAMPLE_INPUT_ENCODED

    def test_encodes_query_and_fragment(self):
        assert encode_path_segment("file?admin=1") == "file%3Fadmin%3D1"
        assert encode_path_segment("file#frag") == "file%23frag"

    def test_leaves_normal_ids_unchanged(self):
        assert encode_path_segment("file-abc123") == "file-abc123"
        assert encode_path_segment("file_abc123") == "file_abc123"

    def test_rejects_bare_dotdot(self):
        with pytest.raises(ValueError):
            encode_path_segment("..")
        with pytest.raises(ValueError):
            encode_path_segment(".")

    def test_rejects_none_and_empty(self):
        with pytest.raises(ValueError, match="identifier is required"):
            encode_path_segment(None)
        with pytest.raises(ValueError, match="identifier is required"):
            encode_path_segment("")


class TestEncodeUrlPath:
    def test_preserves_legitimate_slashes_and_at(self):
        assert (
            encode_url_path("@cf/meta/llama-3.1-8b-instruct")
            == "@cf/meta/llama-3.1-8b-instruct"
        )
        assert encode_url_path("google/gemma-3-4b-it") == "google/gemma-3-4b-it"

    def test_rejects_dotdot_segment(self):
        with pytest.raises(ValueError):
            encode_url_path("../../etc/passwd")
        with pytest.raises(ValueError):
            encode_url_path("@cf/../secret")

    def test_rejects_single_dot_segment(self):
        with pytest.raises(ValueError):
            encode_url_path("./secret")

    def test_rejects_empty_segments(self):
        with pytest.raises(ValueError):
            encode_url_path("foo//bar")
        with pytest.raises(ValueError):
            encode_url_path("/foo")
        with pytest.raises(ValueError):
            encode_url_path("foo/")

    def test_encodes_query_fragment_and_colon(self):
        assert encode_url_path("model?x=1") == "model%3Fx%3D1"
        assert encode_url_path("model#frag") == "model%23frag"
        assert encode_url_path("evil.com:80/x") == "evil.com%3A80/x"

    def test_none_becomes_empty(self):
        assert encode_url_path(None) == ""


class TestAnthropicFilesEncoding:
    def _config(self):
        from litellm.llms.anthropic.files.transformation import AnthropicFilesConfig

        return AnthropicFilesConfig()

    def test_retrieve(self):
        url, _ = self._config().transform_retrieve_file_request(
            file_id=SAMPLE_INPUT,
            optional_params={},
            litellm_params={},
        )
        _assert_input_is_encoded(url, "/v1/files/")

    def test_delete(self):
        url, _ = self._config().transform_delete_file_request(
            file_id=SAMPLE_INPUT,
            optional_params={},
            litellm_params={},
        )
        _assert_input_is_encoded(url, "/v1/files/")

    def test_content(self):
        url, _ = self._config().transform_file_content_request(
            file_content_request={"file_id": SAMPLE_INPUT},
            optional_params={},
            litellm_params={},
        )
        _assert_input_is_encoded(url, "/v1/files/")

    @pytest.mark.parametrize(
        "method_name",
        [
            "transform_retrieve_file_request",
            "transform_delete_file_request",
        ],
    )
    def test_missing_file_id_raises(self, method_name):
        with pytest.raises(ValueError, match="identifier is required"):
            getattr(self._config(), method_name)(
                file_id="", optional_params={}, litellm_params={}
            )

    def test_missing_file_id_content_raises(self):
        with pytest.raises(ValueError, match="identifier is required"):
            self._config().transform_file_content_request(
                file_content_request={},  # no file_id
                optional_params={},
                litellm_params={},
            )


class TestAnthropicBatchesEncoding:
    def test_retrieve_url(self):
        from litellm.llms.anthropic.batches.transformation import AnthropicBatchesConfig

        url = AnthropicBatchesConfig().get_retrieve_batch_url(
            api_base="https://api.anthropic.com",
            batch_id=SAMPLE_INPUT,
            optional_params={},
            litellm_params={},
        )
        _assert_input_is_encoded(url, "/batches/")


class TestAnthropicSkillsEncoding:
    def test_get_complete_url(self):
        from litellm.llms.anthropic.skills.transformation import AnthropicSkillsConfig

        url = AnthropicSkillsConfig().get_complete_url(
            api_base="https://api.anthropic.com",
            endpoint="skills",
            skill_id=SAMPLE_INPUT,
        )
        _assert_input_is_encoded(url, "/v1/skills/")


class TestOpenAIVideosEncoding:
    def _config(self):
        from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
        from litellm.types.router import GenericLiteLLMParams

        return OpenAIVideoConfig(), GenericLiteLLMParams()

    def test_content(self):
        cfg, params = self._config()
        url, _ = cfg.transform_video_content_request(
            video_id=SAMPLE_INPUT,
            api_base="https://api.openai.com/v1/videos",
            litellm_params=params,
            headers={},
        )
        _assert_input_is_encoded(url, "/v1/videos/")

    def test_content_variant_is_encoded(self):
        cfg, params = self._config()
        url, _ = cfg.transform_video_content_request(
            video_id="vid_ok",
            api_base="https://api.openai.com/v1/videos",
            litellm_params=params,
            headers={},
            variant="bad&inject=1",
        )
        assert "?variant=bad%26inject%3D1" in url

    def test_delete(self):
        cfg, params = self._config()
        url, _ = cfg.transform_video_delete_request(
            video_id=SAMPLE_INPUT,
            api_base="https://api.openai.com/v1/videos",
            litellm_params=params,
            headers={},
        )
        _assert_input_is_encoded(url, "/v1/videos/")

    def test_status_retrieve(self):
        cfg, params = self._config()
        url, _ = cfg.transform_video_status_retrieve_request(
            video_id=SAMPLE_INPUT,
            api_base="https://api.openai.com/v1/videos",
            litellm_params=params,
            headers={},
        )
        _assert_input_is_encoded(url, "/v1/videos/")

    def test_remix(self):
        cfg, params = self._config()
        url, _ = cfg.transform_video_remix_request(
            video_id=SAMPLE_INPUT,
            prompt="x",
            api_base="https://api.openai.com/v1/videos",
            litellm_params=params,
            headers={},
        )
        _assert_input_is_encoded(url, "/v1/videos/")


class TestOpenAIContainersEncoding:
    def _config(self):
        from litellm.llms.openai.containers.transformation import (
            OpenAIContainerConfig,
        )
        from litellm.types.router import GenericLiteLLMParams

        return OpenAIContainerConfig(), GenericLiteLLMParams()

    def test_retrieve(self):
        cfg, params = self._config()
        url, _ = cfg.transform_container_retrieve_request(
            container_id=SAMPLE_INPUT,
            api_base="https://api.openai.com/v1/containers",
            litellm_params=params,
            headers={},
        )
        _assert_input_is_encoded(url, "/v1/containers/")

    def test_delete(self):
        cfg, params = self._config()
        url, _ = cfg.transform_container_delete_request(
            container_id=SAMPLE_INPUT,
            api_base="https://api.openai.com/v1/containers",
            litellm_params=params,
            headers={},
        )
        _assert_input_is_encoded(url, "/v1/containers/")

    def test_file_list(self):
        cfg, params = self._config()
        url, _ = cfg.transform_container_file_list_request(
            container_id=SAMPLE_INPUT,
            api_base="https://api.openai.com/v1/containers",
            litellm_params=params,
            headers={},
        )
        _assert_input_is_encoded(url, "/v1/containers/")

    def test_file_content(self):
        cfg, params = self._config()
        url, _ = cfg.transform_container_file_content_request(
            container_id="cntr_ok",
            file_id=SAMPLE_INPUT,
            api_base="https://api.openai.com/v1/containers",
            litellm_params=params,
            headers={},
        )
        _assert_input_is_encoded(url, "/cntr_ok/files/")


class TestOpenAIVectorStoresEncoding:
    def test_search(self):
        from litellm.llms.openai.vector_stores.transformation import (
            OpenAIVectorStoreConfig,
        )

        url, _ = OpenAIVectorStoreConfig().transform_search_vector_store_request(
            vector_store_id=SAMPLE_INPUT,
            query="x",
            vector_store_search_optional_params={},
            api_base="https://api.openai.com/v1/vector_stores",
            litellm_logging_obj=None,
            litellm_params={},
        )
        _assert_input_is_encoded(url, "/v1/vector_stores/")


class TestOpenAIVectorStoreFilesEncoding:
    def _config(self):
        from litellm.llms.openai.vector_store_files.transformation import (
            OpenAIVectorStoreFilesConfig,
        )

        return OpenAIVectorStoreFilesConfig()

    def test_get_complete_url(self):
        url = self._config().get_complete_url(
            api_base="https://api.openai.com/v1",
            vector_store_id=SAMPLE_INPUT,
            litellm_params={},
        )
        _assert_input_is_encoded(url, "/vector_stores/")

    def test_retrieve(self):
        url, _ = self._config().transform_retrieve_vector_store_file_request(
            vector_store_id="vs_ok",
            file_id=SAMPLE_INPUT,
            api_base="https://api.openai.com/v1/vector_stores/vs_ok/files",
        )
        _assert_input_is_encoded(url, "/files/")

    def test_content(self):
        url, _ = self._config().transform_retrieve_vector_store_file_content_request(
            vector_store_id="vs_ok",
            file_id=SAMPLE_INPUT,
            api_base="https://api.openai.com/v1/vector_stores/vs_ok/files",
        )
        _assert_input_is_encoded(url, "/files/")

    def test_update(self):
        url, _ = self._config().transform_update_vector_store_file_request(
            vector_store_id="vs_ok",
            file_id=SAMPLE_INPUT,
            update_request={"attributes": None},
            api_base="https://api.openai.com/v1/vector_stores/vs_ok/files",
        )
        _assert_input_is_encoded(url, "/files/")

    def test_delete(self):
        url, _ = self._config().transform_delete_vector_store_file_request(
            vector_store_id="vs_ok",
            file_id=SAMPLE_INPUT,
            api_base="https://api.openai.com/v1/vector_stores/vs_ok/files",
        )
        _assert_input_is_encoded(url, "/files/")


class TestGeminiInteractionsEncoding:
    def _config(self):
        from litellm.llms.gemini.interactions.transformation import (
            GoogleAIStudioInteractionsConfig,
        )
        from litellm.types.router import GenericLiteLLMParams

        return (
            GoogleAIStudioInteractionsConfig(),
            GenericLiteLLMParams(api_key="sk-test"),
        )

    def test_get(self):
        cfg, params = self._config()
        url, _ = cfg.transform_get_interaction_request(
            interaction_id=SAMPLE_INPUT,
            api_base="https://generativelanguage.googleapis.com",
            litellm_params=params,
            headers={},
        )
        _assert_input_is_encoded(url, "/interactions/")

    def test_delete(self):
        cfg, params = self._config()
        url, _ = cfg.transform_delete_interaction_request(
            interaction_id=SAMPLE_INPUT,
            api_base="https://generativelanguage.googleapis.com",
            litellm_params=params,
            headers={},
        )
        _assert_input_is_encoded(url, "/interactions/")

    def test_cancel(self):
        cfg, params = self._config()
        url, _ = cfg.transform_cancel_interaction_request(
            interaction_id=SAMPLE_INPUT,
            api_base="https://generativelanguage.googleapis.com",
            litellm_params=params,
            headers={},
        )
        _assert_input_is_encoded(url, "/interactions/")


class TestBedrockCountTokensEncoding:
    def _config(self):
        from litellm.llms.bedrock.count_tokens.transformation import (
            BedrockCountTokensConfig,
        )

        return BedrockCountTokensConfig()

    def test_endpoint(self):
        url = self._config().get_bedrock_count_tokens_endpoint(
            model=SAMPLE_INPUT,
            aws_region_name="us-east-1",
            api_base=None,
            aws_bedrock_runtime_endpoint=None,
        )
        _assert_input_is_encoded(url, "/model/")

    def test_versioned_model_id_preserves_colon(self):
        url = self._config().get_bedrock_count_tokens_endpoint(
            model="amazon.nova-pro-v1:0",
            aws_region_name="us-east-1",
            api_base=None,
            aws_bedrock_runtime_endpoint=None,
        )
        assert "/model/amazon.nova-pro-v1:0/count-tokens" in url


class TestBedrockInvokeOpenAIEncoding:
    def _config(self):
        from litellm.llms.bedrock.chat.invoke_transformations.amazon_openai_transformation import (
            AmazonBedrockOpenAIConfig,
        )

        return AmazonBedrockOpenAIConfig()

    def test_versioned_model_id_preserves_colon(self):
        cfg = self._config()
        url = cfg.get_complete_url(
            api_base=None,
            api_key=None,
            model="bedrock/openai/amazon.nova-pro-v1:0",
            optional_params={"aws_region_name": "us-east-1"},
            litellm_params={},
            stream=False,
        )
        assert "/model/amazon.nova-pro-v1:0/invoke" in url

    def test_traversal_input_is_encoded(self):
        cfg = self._config()
        url = cfg.get_complete_url(
            api_base=None,
            api_key=None,
            model="bedrock/openai/../../foo",
            optional_params={"aws_region_name": "us-east-1"},
            litellm_params={},
            stream=False,
        )
        assert "..%2F..%2Ffoo" in url
        assert "/model/../" not in url


class TestCloudflareEncoding:
    def test_rejects_dot_segment(self, monkeypatch):
        from litellm.llms.cloudflare.chat.transformation import CloudflareChatConfig

        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
        cfg = CloudflareChatConfig()
        with pytest.raises(ValueError):
            cfg.get_complete_url(
                api_base=None,
                api_key="x",
                model=SAMPLE_INPUT,
                optional_params={},
                litellm_params={},
                stream=False,
            )

    def test_legitimate_model_preserved(self, monkeypatch):
        from litellm.llms.cloudflare.chat.transformation import CloudflareChatConfig

        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
        cfg = CloudflareChatConfig()
        url = cfg.get_complete_url(
            api_base=None,
            api_key="x",
            model="@cf/meta/llama-3.1-8b-instruct",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url.endswith("@cf/meta/llama-3.1-8b-instruct")


class TestBytezEncoding:
    def test_rejects_dot_segment(self):
        from litellm.llms.bytez.chat.transformation import BytezChatConfig

        with pytest.raises(ValueError):
            BytezChatConfig().get_complete_url(
                api_base=None,
                api_key="x",
                model=SAMPLE_INPUT,
                optional_params={},
                litellm_params={},
                stream=False,
            )

    def test_legitimate_model_preserved(self):
        from litellm.llms.bytez.chat.transformation import BytezChatConfig

        url = BytezChatConfig().get_complete_url(
            api_base=None,
            api_key="x",
            model="google/gemma-3-4b-it",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url.endswith("google/gemma-3-4b-it")


class TestRagflowEncoding:
    def test_rejects_dot_segment(self):
        from litellm.llms.ragflow.chat.transformation import RAGFlowConfig

        cfg = RAGFlowConfig()
        with pytest.raises(ValueError):
            cfg.get_complete_url(
                api_base="http://ragflow.example",
                api_key="x",
                model="ragflow/chat/../../v1/messages/batches/llama",
                optional_params={},
                litellm_params={},
                stream=False,
            )

    def test_query_chars_in_entity_id_are_encoded(self):
        from litellm.llms.ragflow.chat.transformation import RAGFlowConfig

        cfg = RAGFlowConfig()
        url = cfg.get_complete_url(
            api_base="http://ragflow.example",
            api_key="x",
            model="ragflow/chat/id?admin=1/llama",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert "id%3Fadmin%3D1" in url
        assert "?admin=1" not in url


class TestPGVectorEncoding:
    def test_search(self):
        from litellm.llms.pg_vector.vector_stores.transformation import (
            PGVectorStoreConfig,
        )

        url, _ = PGVectorStoreConfig().transform_search_vector_store_request(
            vector_store_id=SAMPLE_INPUT,
            query="x",
            vector_store_search_optional_params={},
            api_base="http://pg.example/v1/vector_stores",
            litellm_logging_obj=None,
            litellm_params={},
        )
        _assert_input_is_encoded(url, "/v1/vector_stores/")


class TestContainerHandlerBuildUrl:
    def test_path_params_encoded(self):
        from litellm.llms.custom_httpx.container_handler import _build_url

        url = _build_url(
            api_base="https://api.openai.com/v1/containers",
            path_template="/containers/{container_id}/files/{file_id}",
            path_params={"container_id": "cntr_ok", "file_id": SAMPLE_INPUT},
        )
        _assert_input_is_encoded(url, "/files/")


class TestOpenAIEvalsEncoding:
    def _config(self):
        from litellm.llms.openai.evals.transformation import OpenAIEvalsConfig

        return OpenAIEvalsConfig()

    def test_get_complete_url(self):
        cfg = self._config()
        url = cfg.get_complete_url(
            api_base="https://api.openai.com",
            endpoint="evals",
            eval_id=SAMPLE_INPUT,
        )
        _assert_input_is_encoded(url, "/v1/evals/")

    def test_get_run(self):
        from litellm.types.router import GenericLiteLLMParams

        url, _ = self._config().transform_get_run_request(
            eval_id="eval_ok",
            run_id=SAMPLE_INPUT,
            api_base="https://api.openai.com",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        _assert_input_is_encoded(url, "/runs/")


class TestOpenAIResponsesEncoding:
    def _config(self):
        from litellm.llms.openai.responses.transformation import (
            OpenAIResponsesAPIConfig,
        )
        from litellm.types.router import GenericLiteLLMParams

        return OpenAIResponsesAPIConfig(), GenericLiteLLMParams()

    def test_delete(self):
        cfg, params = self._config()
        url, _ = cfg.transform_delete_response_api_request(
            response_id=SAMPLE_INPUT,
            api_base="https://api.openai.com/v1/responses",
            litellm_params=params,
            headers={},
        )
        _assert_input_is_encoded(url, "/v1/responses/")


class TestManusFilesEncoding:
    def _config(self):
        from litellm.llms.manus.files.transformation import ManusFilesConfig

        return ManusFilesConfig()

    def test_retrieve(self):
        url, _ = self._config().transform_retrieve_file_request(
            file_id=SAMPLE_INPUT,
            optional_params={},
            litellm_params={"api_base": "https://api.manus.im/v1/files"},
        )
        _assert_input_is_encoded(url, "/files/")


class TestAzureAIAgentsEncoding:
    def _handler(self):
        from litellm.llms.azure_ai.agents.handler import AzureAIAgentsHandler

        return AzureAIAgentsHandler.__new__(AzureAIAgentsHandler)

    def test_messages_url(self):
        handler = self._handler()
        url = handler._build_messages_url(
            api_base="https://ai.example/api/projects/p",
            thread_id=SAMPLE_INPUT,
            api_version="2025-05-01",
        )
        _assert_input_is_encoded(url, "/threads/")

    def test_run_status_url(self):
        handler = self._handler()
        url = handler._build_run_status_url(
            api_base="https://ai.example/api/projects/p",
            thread_id="thread_ok",
            run_id=SAMPLE_INPUT,
            api_version="2025-05-01",
        )
        _assert_input_is_encoded(url, "/runs/")


class TestVertexVideosEncoding:
    def test_operation_status_rejects_dot_segments(self):
        from litellm.llms.vertex_ai.videos.transformation import (
            VertexAIVideoConfig,
        )
        from litellm.types.router import GenericLiteLLMParams

        cfg = VertexAIVideoConfig()
        with pytest.raises(ValueError):
            cfg.transform_video_status_retrieve_request(
                video_id="../../v1/models",
                api_base="https://aiplatform.googleapis.com",
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )


class TestNvidiaNimEncoding:
    def test_rejects_dot_segment(self):
        from litellm.llms.nvidia_nim.rerank.transformation import (
            NvidiaNimRerankConfig,
        )

        cfg = NvidiaNimRerankConfig()
        with pytest.raises(ValueError):
            cfg.get_complete_url(
                api_base="https://integrate.api.nvidia.com/v1",
                model="nvidia_nim/../../v1/models",
                optional_params={},
            )

    def test_legitimate_model_preserved(self):
        from litellm.llms.nvidia_nim.rerank.transformation import (
            NvidiaNimRerankConfig,
        )

        url = NvidiaNimRerankConfig().get_complete_url(
            api_base="https://integrate.api.nvidia.com",
            model="nvidia_nim/nvidia/nv-rerankqa-mistral-4b-v3",
            optional_params={},
        )
        assert url.endswith("nvidia/nv-rerankqa-mistral-4b-v3/reranking")


class TestHuggingfaceEmbeddingEncoding:
    def test_ids_encoded(self):
        assert encode_url_path("BAAI/bge-large-en-v1.5") == "BAAI/bge-large-en-v1.5"


class TestElevenLabsEncoding:
    def test_voice_id_encoded(self):
        from litellm.llms.elevenlabs.text_to_speech.transformation import (
            ElevenLabsTextToSpeechConfig,
        )

        cfg = ElevenLabsTextToSpeechConfig()
        url = cfg.get_complete_url(
            model="eleven_multilingual_v2",
            api_base="https://api.elevenlabs.io",
            litellm_params={
                cfg.ELEVENLABS_VOICE_ID_KEY: SAMPLE_INPUT,
            },
        )
        assert SAMPLE_INPUT_ENCODED in url
        assert "../" not in url.split("/text-to-speech/", 1)[1]
