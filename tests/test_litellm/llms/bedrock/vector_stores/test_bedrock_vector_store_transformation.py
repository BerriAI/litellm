from unittest.mock import MagicMock

from litellm.llms.bedrock.vector_stores.transformation import BedrockVectorStoreConfig


def test_transform_search_request():
    """
    Test that BedrockVectorStoreConfig correctly transforms search vector store requests.
    
    Verifies that the transformation creates the proper URL endpoint and request body
    with the expected retrievalQuery structure.
    """
    config = BedrockVectorStoreConfig()
    mock_log = MagicMock()
    mock_log.model_call_details = {}

    url, body = config.transform_search_vector_store_request(
        vector_store_id="kb123",
        query="hello",
        vector_store_search_optional_params={},
        api_base="https://bedrock-agent-runtime.us-west-2.amazonaws.com/knowledgebases",
        litellm_logging_obj=mock_log,
        litellm_params={},
    )

    assert url.endswith("/kb123/retrieve")
    assert body["retrievalQuery"].get("text") == "hello"


def test_get_uri_from_location_s3():
    config = BedrockVectorStoreConfig()
    location = {
        "type": "S3",
        "s3Location": {"uri": "s3://my-bucket/docs/file.pdf"},
    }
    assert config._get_uri_from_location(location) == "s3://my-bucket/docs/file.pdf"


def test_get_uri_from_location_web():
    config = BedrockVectorStoreConfig()
    location = {
        "type": "WEB",
        "webLocation": {"url": "https://example.com/page"},
    }
    assert config._get_uri_from_location(location) == "https://example.com/page"


def test_get_uri_from_location_confluence():
    config = BedrockVectorStoreConfig()
    location = {
        "type": "CONFLUENCE",
        "confluenceLocation": {"url": "https://myorg.atlassian.net/wiki/spaces/PROJ/pages/123"},
    }
    assert config._get_uri_from_location(location) == "https://myorg.atlassian.net/wiki/spaces/PROJ/pages/123"


def test_get_uri_from_location_kendra():
    config = BedrockVectorStoreConfig()
    location = {
        "type": "KENDRA",
        "kendraDocumentLocation": {"uri": "kendra://index-id/doc-id"},
    }
    assert config._get_uri_from_location(location) == "kendra://index-id/doc-id"


def test_get_uri_from_location_salesforce():
    config = BedrockVectorStoreConfig()
    location = {
        "type": "SALESFORCE",
        "salesforceLocation": {"url": "https://myorg.salesforce.com/articles/example"},
    }
    assert config._get_uri_from_location(location) == "https://myorg.salesforce.com/articles/example"


def test_get_uri_from_location_sharepoint():
    config = BedrockVectorStoreConfig()
    location = {
        "type": "SHAREPOINT",
        "sharePointLocation": {"url": "https://myorg.sharepoint.com/sites/team/doc.docx"},
    }
    assert config._get_uri_from_location(location) == "https://myorg.sharepoint.com/sites/team/doc.docx"


def test_get_uri_from_location_custom():
    config = BedrockVectorStoreConfig()
    location = {
        "type": "CUSTOM",
        "customDocumentLocation": {"id": "custom-doc-id-abc123"},
    }
    assert config._get_uri_from_location(location) == "custom-doc-id-abc123"


def test_get_uri_from_location_unknown_returns_none():
    config = BedrockVectorStoreConfig()
    assert config._get_uri_from_location({}) is None
    assert config._get_uri_from_location({"type": "UNKNOWN"}) is None
    assert config._get_uri_from_location({"type": "S3"}) is None  # missing s3Location


def test_transform_response_uses_location_uri():
    """
    When x-amz-bedrock-kb-source-uri is absent from metadata, the URI should
    be resolved from location.s3Location.uri and used for filename/file_id.
    """
    config = BedrockVectorStoreConfig()
    mock_log = MagicMock()
    mock_log.model_call_details = {"query": "test query"}

    raw_response = {
        "retrievalResults": [
            {
                "content": {"text": "some content", "type": "TEXT"},
                "location": {
                    "s3Location": {"uri": "s3://my-company-bedrock-kb/docs/document.md"},
                    "type": "S3",
                },
                "metadata": {
                    "x-amz-bedrock-kb-source-file-modality": "TEXT",
                    "x-amz-bedrock-kb-chunk-id": "8befd6d7-d8d1-49f6-b01d-54cf4e77c01a",
                    "x-amz-bedrock-kb-data-source-id": "ABCDE12345",
                },
                "score": 0.506902021031646,
            }
        ]
    }

    mock_http_response = MagicMock()
    mock_http_response.json.return_value = raw_response
    mock_http_response.status_code = 200

    result = config.transform_search_vector_store_response(mock_http_response, mock_log)

    assert len(result["data"]) == 1
    item = result["data"][0]
    assert item["file_id"] == "s3://my-company-bedrock-kb/docs/document.md"
    assert item["filename"] == "document.md"
    # Synthesized URI must not leak into attributes
    assert "x-amz-bedrock-kb-source-uri" not in item["attributes"]


def test_transform_response_metadata_uri_takes_precedence():
    """
    When x-amz-bedrock-kb-source-uri is already in metadata, it must be used
    and the location field must be ignored.
    """
    config = BedrockVectorStoreConfig()
    mock_log = MagicMock()
    mock_log.model_call_details = {"query": "test query"}

    raw_response = {
        "retrievalResults": [
            {
                "content": {"text": "some content", "type": "TEXT"},
                "location": {
                    "s3Location": {"uri": "s3://example-bucket/location-path/file.pdf"},
                    "type": "S3",
                },
                "metadata": {
                    "x-amz-bedrock-kb-source-uri": "s3://example-bucket/metadata-path/other.pdf",
                },
                "score": 0.9,
            }
        ]
    }

    mock_http_response = MagicMock()
    mock_http_response.json.return_value = raw_response
    mock_http_response.status_code = 200

    result = config.transform_search_vector_store_response(mock_http_response, mock_log)

    item = result["data"][0]
    assert item["file_id"] == "s3://example-bucket/metadata-path/other.pdf"
    assert item["filename"] == "other.pdf"