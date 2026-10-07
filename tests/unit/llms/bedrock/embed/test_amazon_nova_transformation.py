import pytest
from typing import Final
from litellm.llms.bedrock.embed.amazon_nova_transformation import AmazonNovaEmbeddingConfig


@pytest.mark.usefixtures("fake_provider_credentials")
def test_nova_in_model_name():
    """Test that models with 'nova' in the name are detected."""
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    # Test various Nova model name formats
    test_models = [
        "amazon.nova-2-multimodal-embeddings-v1:0",
        "us.amazon.nova-2-multimodal-embeddings-v1:0",
    ]

    for model in test_models:
        provider = BaseAWSLLM.get_bedrock_embedding_provider(model)
        assert provider is not None


@pytest.mark.usefixtures("fake_provider_credentials")
def test_nova_provider_detection():
    """Test that Nova provider is correctly detected."""
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    provider = BaseAWSLLM.get_bedrock_embedding_provider(
        "amazon.nova-2-multimodal-embeddings-v1:0"
    )

    # Should detect "amazon" as provider since "nova" is in the model name
    # but the provider detection looks at the first part before the dot
    assert provider in ["amazon", "nova"]


@pytest.mark.usefixtures("fake_provider_credentials")
def test_audio_embedding_request():
    """Test audio embedding request transformation."""
    config = AmazonNovaEmbeddingConfig()

    inference_params = {
        "embeddingPurpose": "AUDIO_RETRIEVAL",
        "embeddingDimension": 1024,
        "audio": {
            "format": "mp3",
            "source": {"s3Location": {"uri": "s3://my-bucket/audio.mp3"}},
        },
    }

    request = config.transform_request(
        input="s3://my-bucket/audio.mp3",
        inference_params=inference_params,
        async_invoke_route=False,
    )

    params = request["singleEmbeddingParams"]
    assert params["embeddingPurpose"] == "AUDIO_RETRIEVAL"
    assert params["embeddingDimension"] == 1024
    assert params["audio"]["format"] == "mp3"
    assert (
        params["audio"]["source"]["s3Location"]["uri"] == "s3://my-bucket/audio.mp3"
    )


@pytest.mark.usefixtures("fake_provider_credentials")
def test_data_url_audio_parsing():
    """Test that data URL audio files are properly parsed."""
    config = AmazonNovaEmbeddingConfig()

    audio_data_url = "data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAA"

    request = config.transform_request(
        input=audio_data_url,
        inference_params={},
        async_invoke_route=False,
    )

    params = request["singleEmbeddingParams"]
    assert "audio" in params
    assert params["audio"]["format"] == "mp3"
    assert params["audio"]["source"]["bytes"] == "SUQzBAAAAAAAI1RTU0UAAAA"


@pytest.mark.usefixtures("fake_provider_credentials")
def test_data_url_image_parsing():
    """Test that data URL images are properly parsed and transformed."""
    config = AmazonNovaEmbeddingConfig()

    # Test with JPEG image data URL
    jpeg_data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAASABIAAD"

    request = config.transform_request(
        input=jpeg_data_url,
        inference_params={"dimensions": 1024},
        async_invoke_route=False,
    )

    params = request["singleEmbeddingParams"]
    assert "image" in params
    assert params["image"]["format"] == "jpeg"
    assert "source" in params["image"]
    assert params["image"]["source"]["bytes"] == "/9j/4AAQSkZJRgABAQAASABIAAD"
    assert params["embeddingDimension"] == 1024
    assert params["embeddingPurpose"] == "GENERIC_INDEX"


@pytest.mark.usefixtures("fake_provider_credentials")
def test_data_url_jpg_format_conversion():
    """Test that jpg format is converted to jpeg."""
    config = AmazonNovaEmbeddingConfig()

    # Test with jpg (should be converted to jpeg)
    jpg_data_url = "data:image/jpg;base64,/9j/4AAQSkZJRg"

    request = config.transform_request(
        input=jpg_data_url,
        inference_params={},
        async_invoke_route=False,
    )

    params = request["singleEmbeddingParams"]
    assert params["image"]["format"] == "jpeg"  # Should be converted from jpg to jpeg


@pytest.mark.usefixtures("fake_provider_credentials")
def test_data_url_png_image_parsing():
    """Test that data URL PNG images are properly parsed."""
    config = AmazonNovaEmbeddingConfig()

    # Test with PNG image data URL
    png_data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"

    request = config.transform_request(
        input=png_data_url,
        inference_params={},
        async_invoke_route=False,
    )

    params = request["singleEmbeddingParams"]
    assert "image" in params
    assert params["image"]["format"] == "png"
    assert (
        params["image"]["source"]["bytes"]
        == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    )


@pytest.mark.usefixtures("fake_provider_credentials")
def test_data_url_video_parsing():
    """Test that data URL videos are properly parsed."""
    config = AmazonNovaEmbeddingConfig()

    video_data_url = "data:video/mp4;base64,AAAAIGZ0eXBpc29t"

    request = config.transform_request(
        input=video_data_url,
        inference_params={},
        async_invoke_route=False,
    )

    params = request["singleEmbeddingParams"]
    assert "video" in params
    assert params["video"]["format"] == "mp4"
    assert params["video"]["source"]["bytes"] == "AAAAIGZ0eXBpc29t"


@pytest.mark.usefixtures("fake_provider_credentials")
def test_default_embedding_dimension():
    """Test default embedding dimension is 3072."""
    config = AmazonNovaEmbeddingConfig()

    request = config.transform_request(
        input="Test text",
        inference_params={},
        async_invoke_route=False,
    )

    params = request["singleEmbeddingParams"]
    assert params["embeddingDimension"] == 3072


@pytest.mark.usefixtures("fake_provider_credentials")
def test_default_embedding_purpose():
    """Test default embedding purpose is GENERIC_INDEX."""
    config = AmazonNovaEmbeddingConfig()

    request = config.transform_request(
        input="Test text",
        inference_params={},
        async_invoke_route=False,
    )

    params = request["singleEmbeddingParams"]
    assert params["embeddingPurpose"] == "GENERIC_INDEX"


@pytest.mark.usefixtures("fake_provider_credentials")
def test_image_embedding_request():
    """Test image embedding request transformation."""
    config = AmazonNovaEmbeddingConfig()

    # Mock base64 image data
    image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    inference_params = {
        "embeddingPurpose": "IMAGE_RETRIEVAL",
        "embeddingDimension": 1024,
        "image": {
            "format": "png",
            "source": {"bytes": image_data},
            "detailLevel": "STANDARD_IMAGE",
        },
    }

    request = config.transform_request(
        input=image_data,
        inference_params=inference_params,
        async_invoke_route=False,
    )

    params = request["singleEmbeddingParams"]
    assert params["embeddingPurpose"] == "IMAGE_RETRIEVAL"
    assert params["embeddingDimension"] == 1024
    assert params["image"]["format"] == "png"
    assert params["image"]["detailLevel"] == "STANDARD_IMAGE"
    assert "source" in params["image"]
    assert "bytes" in params["image"]["source"]


@pytest.mark.usefixtures("fake_provider_credentials")
def test_text_embedding_async_request():
    """Test asynchronous text embedding request transformation."""
    config = AmazonNovaEmbeddingConfig()

    inference_params = {
        "embeddingPurpose": "TEXT_RETRIEVAL",
        "embeddingDimension": 3072,
        "text": {
            "value": "Long text content...",
            "segmentationConfig": {"maxLengthChars": 10000},
        },
        "output_s3_uri": "s3://my-bucket/output/",
    }

    request = config.transform_request(
        input="Long text content...",
        inference_params=inference_params,
        async_invoke_route=True,
        model_id="amazon.nova-2-multimodal-embeddings-v1:0",
        output_s3_uri="s3://my-bucket/output/",
    )

    assert "modelId" in request
    assert "modelInput" in request
    assert "outputDataConfig" in request

    model_input = request["modelInput"]
    assert model_input["taskType"] == "SEGMENTED_EMBEDDING"
    assert "segmentedEmbeddingParams" in model_input

    params = model_input["segmentedEmbeddingParams"]
    assert params["embeddingPurpose"] == "TEXT_RETRIEVAL"
    assert params["embeddingDimension"] == 3072
    assert params["text"]["segmentationConfig"]["maxLengthChars"] == 10000


@pytest.mark.usefixtures("fake_provider_credentials")
def test_text_embedding_sync_request():
    """Test synchronous text embedding request transformation."""
    config = AmazonNovaEmbeddingConfig()

    inference_params = {
        "embeddingPurpose": "GENERIC_INDEX",
        "embedding_dimension": 1024,
        "truncation_mode": "END",
    }

    request = config.transform_request(
        input="Hello, world!",
        inference_params=inference_params,
        async_invoke_route=False,
    )

    assert request["schemaVersion"] == "nova-multimodal-embed-v1"
    assert request["taskType"] == "SINGLE_EMBEDDING"
    assert "singleEmbeddingParams" in request

    params = request["singleEmbeddingParams"]
    assert params["embeddingPurpose"] == "GENERIC_INDEX"
    assert params["embeddingDimension"] == 1024
    assert params["text"]["truncationMode"] == "END"
    assert params["text"]["value"] == "Hello, world!"


@pytest.mark.usefixtures("fake_provider_credentials")
def test_video_embedding_request():
    """Test video embedding request transformation."""
    config = AmazonNovaEmbeddingConfig()

    inference_params = {
        "embeddingPurpose": "VIDEO_RETRIEVAL",
        "embeddingDimension": 3072,
        "video": {
            "format": "mp4",
            "source": {"s3Location": {"uri": "s3://my-bucket/video.mp4"}},
            "embeddingMode": "AUDIO_VIDEO_COMBINED",
        },
    }

    request = config.transform_request(
        input="s3://my-bucket/video.mp4",
        inference_params=inference_params,
        async_invoke_route=False,
    )

    params = request["singleEmbeddingParams"]
    assert params["embeddingPurpose"] == "VIDEO_RETRIEVAL"
    assert params["embeddingDimension"] == 3072
    assert params["video"]["format"] == "mp4"
    assert params["video"]["embeddingMode"] == "AUDIO_VIDEO_COMBINED"
    assert (
        params["video"]["source"]["s3Location"]["uri"] == "s3://my-bucket/video.mp4"
    )


@pytest.mark.usefixtures("fake_provider_credentials")
def test_async_invoke_response():
    """Test async invoke response transformation."""
    config = AmazonNovaEmbeddingConfig()

    response = {"invocationArn": "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123"}

    result = config.transform_async_invoke_response(response, model="amazon.nova-2-multimodal-embeddings-v1:0")

    assert result.model == "amazon.nova-2-multimodal-embeddings-v1:0"
    assert len(result.data) == 1
    assert result.data[0].embedding == []  # Empty for async jobs
    assert result.usage.total_tokens == 0
    assert hasattr(result, "_hidden_params")
    assert hasattr(result._hidden_params, "_invocation_arn")
    assert (
        result._hidden_params._invocation_arn
        == "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123"
    )


@pytest.mark.usefixtures("fake_provider_credentials")
def test_image_embedding_response_with_image_count():
    """Test that Nova image embedding response populates image_count for cost tracking."""
    config = AmazonNovaEmbeddingConfig()

    response_list = [
        {
            "embeddings": [
                {
                    "embeddingType": "IMAGE",
                    "embedding": [0.1, 0.2, 0.3],
                }
            ]
        }
    ]

    # Simulate batch_data with image in singleEmbeddingParams
    batch_data = [
        {
            "schemaVersion": "nova-multimodal-embed-v1",
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": 3072,
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": "/9j/4AAQSkZJRg=="},
                },
            },
        }
    ]

    result = config.transform_response(
        response_list=response_list,
        model="amazon.nova-2-multimodal-embeddings-v1:0",
        batch_data=batch_data,
    )

    assert result.usage is not None
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.image_count == 1


@pytest.mark.usefixtures("fake_provider_credentials")
def test_multiple_embeddings_response():
    """Test response with multiple embeddings."""
    config = AmazonNovaEmbeddingConfig()

    response_list = [
        {
            "embeddings": [
                {
                    "embeddingType": "TEXT",
                    "embedding": [0.1, 0.2, 0.3],
                }
            ]
        },
        {
            "embeddings": [
                {
                    "embeddingType": "TEXT",
                    "embedding": [0.4, 0.5, 0.6],
                }
            ]
        },
    ]

    result = config.transform_response(response_list, model="amazon.nova-2-multimodal-embeddings-v1:0")

    assert len(result.data) == 2
    assert result.data[0].embedding == [0.1, 0.2, 0.3]
    assert result.data[1].embedding == [0.4, 0.5, 0.6]
    assert result.data[0].index == 0
    assert result.data[1].index == 1


@pytest.mark.usefixtures("fake_provider_credentials")
def test_nova_embedding_backward_compat_no_batch_data():
    """Test that Nova transformer works without batch_data (backward compatibility)."""
    config = AmazonNovaEmbeddingConfig()

    response_list = [
        {
            "embeddings": [
                {
                    "embeddingType": "TEXT",
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                }
            ]
        }
    ]

    # Call without batch_data — should not break
    result = config.transform_response(
        response_list=response_list,
        model="amazon.nova-2-multimodal-embeddings-v1:0",
    )

    assert result.usage is not None
    assert result.usage.total_tokens > 0
    assert result.usage.prompt_tokens_details is None


@pytest.mark.usefixtures("fake_provider_credentials")
def test_text_embedding_response():
    """Test text embedding response transformation."""
    config = AmazonNovaEmbeddingConfig()

    response_list = [
        {
            "embeddings": [
                {
                    "embeddingType": "TEXT",
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                }
            ]
        }
    ]

    result = config.transform_response(response_list, model="amazon.nova-2-multimodal-embeddings-v1:0")

    assert result.model == "amazon.nova-2-multimodal-embeddings-v1:0"
    assert len(result.data) == 1
    assert result.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert result.data[0].index == 0
    assert result.data[0].object == "embedding"
    assert result.usage.total_tokens > 0


@pytest.mark.usefixtures("fake_provider_credentials")
def test_text_embedding_response_no_image_count():
    """Test that Nova text embedding response does not set image_count."""
    config = AmazonNovaEmbeddingConfig()

    response_list = [
        {
            "embeddings": [
                {
                    "embeddingType": "TEXT",
                    "embedding": [0.1, 0.2, 0.3],
                    "truncatedCharLength": 20,
                }
            ]
        }
    ]

    batch_data = [
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

    result = config.transform_response(
        response_list=response_list,
        model="amazon.nova-2-multimodal-embeddings-v1:0",
        batch_data=batch_data,
    )

    assert result.usage is not None
    assert result.usage.prompt_tokens_details is None


@pytest.mark.usefixtures("fake_provider_credentials")
def test_video_embedding_response_separate_mode():
    """Test video embedding response with separate audio/video."""
    config = AmazonNovaEmbeddingConfig()

    response_list = [
        {
            "embeddings": [
                {
                    "embeddingType": "VIDEO",
                    "embedding": [0.1, 0.2, 0.3],
                },
                {
                    "embeddingType": "AUDIO",
                    "embedding": [0.4, 0.5, 0.6],
                },
            ]
        }
    ]

    result = config.transform_response(response_list, model="amazon.nova-2-multimodal-embeddings-v1:0")

    assert len(result.data) == 2
    assert result.data[0].embedding == [0.1, 0.2, 0.3]
    assert result.data[1].embedding == [0.4, 0.5, 0.6]
