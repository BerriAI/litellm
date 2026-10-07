from typing import Final
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.embed.amazon_nova_transformation import AmazonNovaEmbeddingConfig


def test_nova_in_model_name() -> None:
    """Test that models with 'nova' in the name are detected."""
    test_models: Final = ["amazon.nova-2-multimodal-embeddings-v1:0", "us.amazon.nova-2-multimodal-embeddings-v1:0"]
    for model in test_models:
        provider: Final = BaseAWSLLM.get_bedrock_embedding_provider(model)
        assert provider is not None


def test_nova_provider_detection() -> None:
    """Test that Nova provider is correctly detected."""
    provider: Final = BaseAWSLLM.get_bedrock_embedding_provider("amazon.nova-2-multimodal-embeddings-v1:0")
    assert provider in ["amazon", "nova"]


def test_audio_embedding_request() -> None:
    """Test audio embedding request transformation."""
    config: Final = AmazonNovaEmbeddingConfig()
    inference_params: Final = {
        "embeddingPurpose": "AUDIO_RETRIEVAL",
        "embeddingDimension": 1024,
        "audio": {"format": "mp3", "source": {"s3Location": {"uri": "s3://my-bucket/audio.mp3"}}},
    }
    request: Final = config.transform_request(
        input="s3://my-bucket/audio.mp3", inference_params=inference_params, async_invoke_route=False
    )
    params: Final = request["singleEmbeddingParams"]
    assert params["embeddingPurpose"] == "AUDIO_RETRIEVAL"
    assert params["embeddingDimension"] == 1024
    assert params["audio"]["format"] == "mp3"
    assert params["audio"]["source"]["s3Location"]["uri"] == "s3://my-bucket/audio.mp3"


def test_data_url_audio_parsing() -> None:
    """Test that data URL audio files are properly parsed."""
    config: Final = AmazonNovaEmbeddingConfig()
    audio_data_url: Final = "data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAA"
    request: Final = config.transform_request(input=audio_data_url, inference_params={}, async_invoke_route=False)
    params: Final = request["singleEmbeddingParams"]
    assert "audio" in params
    assert params["audio"]["format"] == "mp3"
    assert params["audio"]["source"]["bytes"] == "SUQzBAAAAAAAI1RTU0UAAAA"


def test_data_url_image_parsing() -> None:
    """Test that data URL images are properly parsed and transformed."""
    config: Final = AmazonNovaEmbeddingConfig()
    jpeg_data_url: Final = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAASABIAAD"
    request: Final = config.transform_request(
        input=jpeg_data_url, inference_params={"dimensions": 1024}, async_invoke_route=False
    )
    params: Final = request["singleEmbeddingParams"]
    assert "image" in params
    assert params["image"]["format"] == "jpeg"
    assert "source" in params["image"]
    assert params["image"]["source"]["bytes"] == "/9j/4AAQSkZJRgABAQAASABIAAD"
    assert params["embeddingDimension"] == 1024
    assert params["embeddingPurpose"] == "GENERIC_INDEX"


def test_data_url_jpg_format_conversion() -> None:
    """Test that jpg format is converted to jpeg."""
    config: Final = AmazonNovaEmbeddingConfig()
    jpg_data_url: Final = "data:image/jpg;base64,/9j/4AAQSkZJRg"
    request: Final = config.transform_request(input=jpg_data_url, inference_params={}, async_invoke_route=False)
    params: Final = request["singleEmbeddingParams"]
    assert params["image"]["format"] == "jpeg"


def test_data_url_png_image_parsing() -> None:
    """Test that data URL PNG images are properly parsed."""
    config: Final = AmazonNovaEmbeddingConfig()
    png_data_url: Final = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    request: Final = config.transform_request(input=png_data_url, inference_params={}, async_invoke_route=False)
    params: Final = request["singleEmbeddingParams"]
    assert "image" in params
    assert params["image"]["format"] == "png"
    assert params["image"]["source"]["bytes"] == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"


def test_data_url_video_parsing() -> None:
    """Test that data URL videos are properly parsed."""
    config: Final = AmazonNovaEmbeddingConfig()
    video_data_url: Final = "data:video/mp4;base64,AAAAIGZ0eXBpc29t"
    request: Final = config.transform_request(input=video_data_url, inference_params={}, async_invoke_route=False)
    params: Final = request["singleEmbeddingParams"]
    assert "video" in params
    assert params["video"]["format"] == "mp4"
    assert params["video"]["source"]["bytes"] == "AAAAIGZ0eXBpc29t"


def test_default_embedding_dimension() -> None:
    """Test default embedding dimension is 3072."""
    config: Final = AmazonNovaEmbeddingConfig()
    request: Final = config.transform_request(input="Test text", inference_params={}, async_invoke_route=False)
    params: Final = request["singleEmbeddingParams"]
    assert params["embeddingDimension"] == 3072


def test_default_embedding_purpose() -> None:
    """Test default embedding purpose is GENERIC_INDEX."""
    config: Final = AmazonNovaEmbeddingConfig()
    request: Final = config.transform_request(input="Test text", inference_params={}, async_invoke_route=False)
    params: Final = request["singleEmbeddingParams"]
    assert params["embeddingPurpose"] == "GENERIC_INDEX"


def test_image_embedding_request() -> None:
    """Test image embedding request transformation."""
    config: Final = AmazonNovaEmbeddingConfig()
    image_data: Final = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    inference_params: Final = {
        "embeddingPurpose": "IMAGE_RETRIEVAL",
        "embeddingDimension": 1024,
        "image": {"format": "png", "source": {"bytes": image_data}, "detailLevel": "STANDARD_IMAGE"},
    }
    request: Final = config.transform_request(
        input=image_data, inference_params=inference_params, async_invoke_route=False
    )
    params: Final = request["singleEmbeddingParams"]
    assert params["embeddingPurpose"] == "IMAGE_RETRIEVAL"
    assert params["embeddingDimension"] == 1024
    assert params["image"]["format"] == "png"
    assert params["image"]["detailLevel"] == "STANDARD_IMAGE"
    assert "source" in params["image"]
    assert "bytes" in params["image"]["source"]


def test_text_embedding_async_request() -> None:
    """Test asynchronous text embedding request transformation."""
    config: Final = AmazonNovaEmbeddingConfig()
    inference_params: Final = {
        "embeddingPurpose": "TEXT_RETRIEVAL",
        "embeddingDimension": 3072,
        "text": {"value": "Long text content...", "segmentationConfig": {"maxLengthChars": 10000}},
        "output_s3_uri": "s3://my-bucket/output/",
    }
    request: Final = config.transform_request(
        input="Long text content...",
        inference_params=inference_params,
        async_invoke_route=True,
        model_id="amazon.nova-2-multimodal-embeddings-v1:0",
        output_s3_uri="s3://my-bucket/output/",
    )
    assert "modelId" in request
    assert "modelInput" in request
    assert "outputDataConfig" in request
    model_input: Final = request["modelInput"]
    assert model_input["taskType"] == "SEGMENTED_EMBEDDING"
    assert "segmentedEmbeddingParams" in model_input
    params: Final = model_input["segmentedEmbeddingParams"]
    assert params["embeddingPurpose"] == "TEXT_RETRIEVAL"
    assert params["embeddingDimension"] == 3072
    assert params["text"]["segmentationConfig"]["maxLengthChars"] == 10000


def test_text_embedding_sync_request() -> None:
    """Test synchronous text embedding request transformation."""
    config: Final = AmazonNovaEmbeddingConfig()
    inference_params: Final = {
        "embeddingPurpose": "GENERIC_INDEX",
        "embedding_dimension": 1024,
        "truncation_mode": "END",
    }
    request: Final = config.transform_request(
        input="Hello, world!", inference_params=inference_params, async_invoke_route=False
    )
    assert request["schemaVersion"] == "nova-multimodal-embed-v1"
    assert request["taskType"] == "SINGLE_EMBEDDING"
    assert "singleEmbeddingParams" in request
    params: Final = request["singleEmbeddingParams"]
    assert params["embeddingPurpose"] == "GENERIC_INDEX"
    assert params["embeddingDimension"] == 1024
    assert params["text"]["truncationMode"] == "END"
    assert params["text"]["value"] == "Hello, world!"


def test_video_embedding_request() -> None:
    """Test video embedding request transformation."""
    config: Final = AmazonNovaEmbeddingConfig()
    inference_params: Final = {
        "embeddingPurpose": "VIDEO_RETRIEVAL",
        "embeddingDimension": 3072,
        "video": {
            "format": "mp4",
            "source": {"s3Location": {"uri": "s3://my-bucket/video.mp4"}},
            "embeddingMode": "AUDIO_VIDEO_COMBINED",
        },
    }
    request: Final = config.transform_request(
        input="s3://my-bucket/video.mp4", inference_params=inference_params, async_invoke_route=False
    )
    params: Final = request["singleEmbeddingParams"]
    assert params["embeddingPurpose"] == "VIDEO_RETRIEVAL"
    assert params["embeddingDimension"] == 3072
    assert params["video"]["format"] == "mp4"
    assert params["video"]["embeddingMode"] == "AUDIO_VIDEO_COMBINED"
    assert params["video"]["source"]["s3Location"]["uri"] == "s3://my-bucket/video.mp4"


def test_async_invoke_response() -> None:
    """Test async invoke response transformation."""
    config: Final = AmazonNovaEmbeddingConfig()
    response: Final = {"invocationArn": "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123"}
    result: Final = config.transform_async_invoke_response(response, model="amazon.nova-2-multimodal-embeddings-v1:0")
    assert result.model == "amazon.nova-2-multimodal-embeddings-v1:0"
    assert len(result.data) == 1
    assert result.data[0].embedding == []
    assert result.usage.total_tokens == 0
    assert hasattr(result, "_hidden_params")
    assert hasattr(result._hidden_params, "_invocation_arn")
    assert result._hidden_params._invocation_arn == "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123"


def test_image_embedding_response_with_image_count() -> None:
    """Test that Nova image embedding response populates image_count for cost tracking."""
    config: Final = AmazonNovaEmbeddingConfig()
    response_list: Final = [{"embeddings": [{"embeddingType": "IMAGE", "embedding": [0.1, 0.2, 0.3]}]}]
    batch_data: Final = [
        {
            "schemaVersion": "nova-multimodal-embed-v1",
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": 3072,
                "image": {"format": "jpeg", "source": {"bytes": "/9j/4AAQSkZJRg=="}},
            },
        }
    ]
    result: Final = config.transform_response(
        response_list=response_list, model="amazon.nova-2-multimodal-embeddings-v1:0", batch_data=batch_data
    )
    assert result.usage is not None
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.image_count == 1


def test_multiple_embeddings_response() -> None:
    """Test response with multiple embeddings."""
    config: Final = AmazonNovaEmbeddingConfig()
    response_list: Final = [
        {"embeddings": [{"embeddingType": "TEXT", "embedding": [0.1, 0.2, 0.3]}]},
        {"embeddings": [{"embeddingType": "TEXT", "embedding": [0.4, 0.5, 0.6]}]},
    ]
    result: Final = config.transform_response(response_list, model="amazon.nova-2-multimodal-embeddings-v1:0")
    assert len(result.data) == 2
    assert result.data[0].embedding == [0.1, 0.2, 0.3]
    assert result.data[1].embedding == [0.4, 0.5, 0.6]
    assert result.data[0].index == 0
    assert result.data[1].index == 1


def test_nova_embedding_backward_compat_no_batch_data() -> None:
    """Test that Nova transformer works without batch_data (backward compatibility)."""
    config: Final = AmazonNovaEmbeddingConfig()
    response_list: Final = [{"embeddings": [{"embeddingType": "TEXT", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]}]
    result: Final = config.transform_response(
        response_list=response_list, model="amazon.nova-2-multimodal-embeddings-v1:0"
    )
    assert result.usage is not None
    assert result.usage.total_tokens > 0
    assert result.usage.prompt_tokens_details is None


def test_text_embedding_response() -> None:
    """Test text embedding response transformation."""
    config: Final = AmazonNovaEmbeddingConfig()
    response_list: Final = [{"embeddings": [{"embeddingType": "TEXT", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]}]
    result: Final = config.transform_response(response_list, model="amazon.nova-2-multimodal-embeddings-v1:0")
    assert result.model == "amazon.nova-2-multimodal-embeddings-v1:0"
    assert len(result.data) == 1
    assert result.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert result.data[0].index == 0
    assert result.data[0].object == "embedding"
    assert result.usage.total_tokens > 0


def test_text_embedding_response_no_image_count() -> None:
    """Test that Nova text embedding response does not set image_count."""
    config: Final = AmazonNovaEmbeddingConfig()
    response_list: Final = [
        {"embeddings": [{"embeddingType": "TEXT", "embedding": [0.1, 0.2, 0.3], "truncatedCharLength": 20}]}
    ]
    batch_data: Final = [
        {
            "schemaVersion": "nova-multimodal-embed-v1",
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": 3072,
                "text": {"value": "hello world", "truncationMode": "END"},
            },
        }
    ]
    result: Final = config.transform_response(
        response_list=response_list, model="amazon.nova-2-multimodal-embeddings-v1:0", batch_data=batch_data
    )
    assert result.usage is not None
    assert result.usage.prompt_tokens_details is None


def test_video_embedding_response_separate_mode() -> None:
    """Test video embedding response with separate audio/video."""
    config: Final = AmazonNovaEmbeddingConfig()
    response_list: Final = [
        {
            "embeddings": [
                {"embeddingType": "VIDEO", "embedding": [0.1, 0.2, 0.3]},
                {"embeddingType": "AUDIO", "embedding": [0.4, 0.5, 0.6]},
            ]
        }
    ]
    result: Final = config.transform_response(response_list, model="amazon.nova-2-multimodal-embeddings-v1:0")
    assert len(result.data) == 2
    assert result.data[0].embedding == [0.1, 0.2, 0.3]
    assert result.data[1].embedding == [0.4, 0.5, 0.6]
