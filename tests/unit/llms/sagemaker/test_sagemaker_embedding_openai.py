"""
Test cases for OpenAI-compatible SageMaker embedding endpoints and inference components

Covers the `sagemaker/openai/<endpoint>` route (request and response in the OpenAI
embeddings shape) and `model_id` forwarded to `invoke_endpoint` as `InferenceComponentName`
for every SageMaker embedding config.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from botocore.exceptions import ClientError, ConnectionClosedError

import litellm
from litellm import Router
from litellm.llms.sagemaker.common_utils import SagemakerError
from litellm.llms.sagemaker.completion.handler import SagemakerLLM
from litellm.llms.sagemaker.embedding.openai_transformation import (
    SagemakerOpenAIEmbeddingConfig,
)
from litellm.llms.sagemaker.embedding.transformation import SagemakerEmbeddingConfig
from litellm.llms.voyage.embedding.transformation import VoyageEmbeddingConfig
from litellm.types.utils import EmbeddingResponse

OPENAI_RESPONSE = {
    "object": "list",
    "data": [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
        {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
    ],
    "model": "served-embedding-model",
    "usage": {"prompt_tokens": 11, "total_tokens": 11},
}


def fake_sagemaker_session(response_body: object) -> tuple[MagicMock, MagicMock]:
    client = MagicMock()
    client.invoke_endpoint.return_value = {
        "Body": MagicMock(read=MagicMock(return_value=json.dumps(response_body).encode()))
    }
    session = MagicMock()
    session.client.return_value = client
    return session, client


def invoke_kwargs(client: MagicMock) -> dict:
    client.invoke_endpoint.assert_called_once()
    return client.invoke_endpoint.call_args.kwargs


class TestSagemakerEmbeddingFactoryOpenAIRoute:
    def test_openai_route_selects_openai_config(self):
        config = SagemakerEmbeddingConfig.get_model_config("openai/my-embed-endpoint")

        assert isinstance(config, SagemakerOpenAIEmbeddingConfig)
        assert SagemakerEmbeddingConfig().get_supported_openai_params("openai/my-embed-endpoint") == [
            "dimensions",
            "encoding_format",
            "user",
        ]

    def test_openai_route_wins_over_model_name_substrings(self):
        assert isinstance(
            SagemakerEmbeddingConfig.get_model_config("openai/voyage-endpoint"),
            SagemakerOpenAIEmbeddingConfig,
        )
        assert isinstance(
            SagemakerEmbeddingConfig.get_model_config("openai/cohere-endpoint"),
            SagemakerOpenAIEmbeddingConfig,
        )

    def test_plain_endpoint_names_keep_their_config(self):
        assert type(SagemakerEmbeddingConfig.get_model_config("my-embed-endpoint")) is SagemakerEmbeddingConfig
        assert isinstance(SagemakerEmbeddingConfig.get_model_config("voyage-endpoint"), VoyageEmbeddingConfig)

    def test_endpoint_name_strips_only_the_route(self):
        assert SagemakerEmbeddingConfig.get_endpoint_name("openai/my-embed-endpoint") == "my-embed-endpoint"
        assert SagemakerEmbeddingConfig.get_endpoint_name("my-embed-endpoint") == "my-embed-endpoint"
        assert SagemakerEmbeddingConfig.get_endpoint_name("my-openai-endpoint") == "my-openai-endpoint"


class TestSagemakerOpenAIEmbeddingConfig:
    def setup_method(self):
        self.config = SagemakerOpenAIEmbeddingConfig()

    def test_request_is_openai_shaped_with_extra_body_flattened(self):
        result = self.config.transform_embedding_request(
            model="my-embed-endpoint",
            input=["hello", "world"],
            optional_params={"truncate": True, "extra_body": {"model": "served-embedding-model"}},
            headers={},
        )

        assert result == {"input": ["hello", "world"], "truncate": True, "model": "served-embedding-model"}

    def test_request_without_extra_body_sends_input_only(self):
        result = self.config.transform_embedding_request(
            model="my-embed-endpoint", input=["hello"], optional_params={}, headers={}
        )

        assert result == {"input": ["hello"]}

    def test_openai_params_pass_through(self):
        mapped = self.config.map_openai_params(
            non_default_params={"dimensions": 256, "encoding_format": "float", "user": "u1", "stream": True},
            optional_params={},
            model="my-embed-endpoint",
            drop_params=False,
        )

        assert mapped == {"dimensions": 256, "encoding_format": "float", "user": "u1"}

    def test_response_reads_data_rows_and_usage(self):
        raw = httpx.Response(200, json=OPENAI_RESPONSE)

        result = self.config.transform_embedding_response(
            model="my-embed-endpoint",
            raw_response=raw,
            model_response=EmbeddingResponse(),
            logging_obj=MagicMock(),
        )

        assert result.object == "list"
        assert result.data == [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
            {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
        ]
        assert result.model == "served-embedding-model"
        assert result.usage is not None
        assert (result.usage.prompt_tokens, result.usage.total_tokens) == (11, 11)

    def test_response_without_usage_or_model_falls_back(self):
        raw = httpx.Response(200, json={"data": [{"embedding": [0.9]}]})

        result = self.config.transform_embedding_response(
            model="my-embed-endpoint",
            raw_response=raw,
            model_response=EmbeddingResponse(),
            logging_obj=MagicMock(),
        )

        assert result.data == [{"object": "embedding", "index": 0, "embedding": [0.9]}]
        assert result.model == "my-embed-endpoint"
        assert result.usage is not None
        assert (result.usage.prompt_tokens, result.usage.total_tokens) == (0, 0)

    def test_response_keeps_base64_embedding_strings(self):
        raw = httpx.Response(
            200, json={"data": [{"index": 0, "embedding": "AACAPwAAAEA="}], "usage": {"prompt_tokens": 2}}
        )

        result = self.config.transform_embedding_response(
            model="my-embed-endpoint",
            raw_response=raw,
            model_response=EmbeddingResponse(),
            logging_obj=MagicMock(),
        )

        assert result.data == [{"object": "embedding", "index": 0, "embedding": "AACAPwAAAEA="}]
        assert result.usage is not None
        assert (result.usage.prompt_tokens, result.usage.total_tokens) == (2, 2)

    @pytest.mark.parametrize(
        "body",
        [
            {"embedding": [[0.1, 0.2]]},
            [[0.1, 0.2]],
            {"data": [{"index": 0}]},
            {"data": "not-a-list"},
        ],
    )
    def test_response_rejects_non_openai_shapes(self, body):
        raw = httpx.Response(200, json=body)

        with pytest.raises(SagemakerError) as exc_info:
            self.config.transform_embedding_response(
                model="my-embed-endpoint",
                raw_response=raw,
                model_response=EmbeddingResponse(),
                logging_obj=MagicMock(),
            )

        assert exc_info.value.status_code == 500
        assert "'data'" in exc_info.value.message


class TestSagemakerEmbeddingInferenceComponents:
    def setup_method(self):
        self.sagemaker_llm = SagemakerLLM()

    def run_embedding(self, model: str, optional_params: dict, response_body: object) -> tuple[EmbeddingResponse, dict]:
        session, client = fake_sagemaker_session(response_body)
        with patch("boto3.Session", return_value=session):
            result = self.sagemaker_llm.embedding(
                model=model,
                input=["hello", "good morning"],
                model_response=EmbeddingResponse(),
                print_verbose=print,
                encoding=None,
                logging_obj=MagicMock(),
                optional_params={"aws_region_name": "us-east-1", **optional_params},
            )
        return result, invoke_kwargs(client)

    def test_openai_route_forwards_component_and_keeps_it_out_of_the_body(self):
        result, kwargs = self.run_embedding(
            model="openai/my-embed-endpoint",
            optional_params={"model_id": "my-component", "extra_body": {"model": "served-embedding-model"}},
            response_body=OPENAI_RESPONSE,
        )

        assert kwargs["EndpointName"] == "my-embed-endpoint"
        assert kwargs["InferenceComponentName"] == "my-component"
        assert json.loads(kwargs["Body"]) == {"input": ["hello", "good morning"], "model": "served-embedding-model"}
        assert [row["embedding"] for row in result.data] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert result.usage is not None
        assert result.usage.prompt_tokens == 11

    def test_hf_endpoint_forwards_component_and_keeps_its_body_shape(self):
        result, kwargs = self.run_embedding(
            model="my-embed-endpoint",
            optional_params={"model_id": "my-component", "normalize": True},
            response_body={"embedding": [[0.1, 0.2], [0.3, 0.4]]},
        )

        assert kwargs["EndpointName"] == "my-embed-endpoint"
        assert kwargs["InferenceComponentName"] == "my-component"
        assert json.loads(kwargs["Body"]) == {"inputs": ["hello", "good morning"], "normalize": True}
        assert [row["embedding"] for row in result.data] == [[0.1, 0.2], [0.3, 0.4]]

    def test_no_component_without_model_id(self):
        _, kwargs = self.run_embedding(
            model="my-embed-endpoint",
            optional_params={},
            response_body={"embedding": [[0.1, 0.2], [0.3, 0.4]]},
        )

        assert "InferenceComponentName" not in kwargs
        assert json.loads(kwargs["Body"]) == {"inputs": ["hello", "good morning"]}

    @pytest.mark.parametrize(
        ("sagemaker_message", "expected_hint"),
        [
            (
                "Inference Component Name header is not allowed for endpoints to which you dont plan to deploy"
                " inference components. Please remove the Inference Component Name header and try again.",
                "remove `model_id` from this deployment, the endpoint has no inference components",
            ),
            (
                "Inference Component Name header is required for this endpoint.",
                "pass in via `litellm.embedding(..., model_id={InferenceComponentName})`",
            ),
            ('Received server error (500) from model with message "boom".', ""),
        ],
    )
    def test_component_errors_keep_status_and_name_model_id(self, sagemaker_message: str, expected_hint: str):
        session, client = fake_sagemaker_session(OPENAI_RESPONSE)
        client.invoke_endpoint.side_effect = ClientError(
            {
                "Error": {"Code": "ValidationError", "Message": sagemaker_message},
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            "InvokeEndpoint",
        )
        with patch("boto3.Session", return_value=session), pytest.raises(SagemakerError) as raised:
            self.sagemaker_llm.embedding(
                model="my-embed-endpoint",
                input=["hello"],
                model_response=EmbeddingResponse(),
                print_verbose=print,
                encoding=None,
                logging_obj=MagicMock(),
                optional_params={"aws_region_name": "us-east-1", "model_id": "stray-component"},
            )

        assert raised.value.status_code == 400
        assert raised.value.message == (
            sagemaker_message + "\n " + expected_hint if expected_hint else sagemaker_message
        )

    def test_null_error_message_surfaces_the_error_code(self):
        session, client = fake_sagemaker_session(OPENAI_RESPONSE)
        client.invoke_endpoint.side_effect = ClientError(
            {
                "Error": {"Code": "InternalFailure", "Message": None},
                "ResponseMetadata": {"HTTPStatusCode": 500},
            },
            "InvokeEndpoint",
        )
        with patch("boto3.Session", return_value=session), pytest.raises(SagemakerError) as raised:
            self.sagemaker_llm.embedding(
                model="openai/my-embed-endpoint",
                input=["hello"],
                model_response=EmbeddingResponse(),
                print_verbose=print,
                encoding=None,
                logging_obj=MagicMock(),
                optional_params={"aws_region_name": "us-east-1", "model_id": "x" * 2048},
            )

        assert raised.value.status_code == 500
        assert "InternalFailure" in raised.value.message

    def test_dropped_connection_surfaces_as_a_server_error(self):
        session, client = fake_sagemaker_session(OPENAI_RESPONSE)
        client.invoke_endpoint.side_effect = ConnectionClosedError(endpoint_url="https://runtime.sagemaker.test/x")
        with patch("boto3.Session", return_value=session), pytest.raises(SagemakerError) as raised:
            self.sagemaker_llm.embedding(
                model="openai/my-embed-endpoint",
                input=["hello"],
                model_response=EmbeddingResponse(),
                print_verbose=print,
                encoding=None,
                logging_obj=MagicMock(),
                optional_params={"aws_region_name": "us-east-1", "model_id": "my-component"},
            )

        assert raised.value.status_code == 500
        assert "Connection was closed" in raised.value.message

    def test_non_string_model_id_is_rejected_before_the_call(self):
        session, client = fake_sagemaker_session(OPENAI_RESPONSE)
        with patch("boto3.Session", return_value=session), pytest.raises(SagemakerError) as raised:
            self.sagemaker_llm.embedding(
                model="openai/my-embed-endpoint",
                input=["hello"],
                model_response=EmbeddingResponse(),
                print_verbose=print,
                encoding=None,
                logging_obj=MagicMock(),
                optional_params={"aws_region_name": "us-east-1", "model_id": ["my-component"]},
            )

        assert raised.value.status_code == 400
        assert "model_id" in raised.value.message
        client.invoke_endpoint.assert_not_called()


SDK_KWARGS = {
    "model": "sagemaker/openai/my-embed-endpoint",
    "input": ["hello", "good morning"],
    "model_id": "my-component",
    "extra_body": {"model": "served-embedding-model"},
    "aws_region_name": "us-east-1",
    "aws_access_key_id": "test-key",
    "aws_secret_access_key": "test-secret",
}

EXPECTED_BODY = {"input": ["hello", "good morning"], "model": "served-embedding-model"}


class TestSagemakerOpenAIEmbeddingEndToEnd:
    def test_sdk_sync(self):
        session, client = fake_sagemaker_session(OPENAI_RESPONSE)
        with patch("boto3.Session", return_value=session):
            result = litellm.embedding(**SDK_KWARGS)

        kwargs = invoke_kwargs(client)
        assert (kwargs["EndpointName"], kwargs["InferenceComponentName"]) == ("my-embed-endpoint", "my-component")
        assert json.loads(kwargs["Body"]) == EXPECTED_BODY
        assert len(result.data) == 2
        assert result.usage is not None
        assert result.usage.total_tokens == 11

    def test_sdk_async(self):
        session, client = fake_sagemaker_session(OPENAI_RESPONSE)
        with patch("boto3.Session", return_value=session):
            result = asyncio.run(litellm.aembedding(**SDK_KWARGS))

        kwargs = invoke_kwargs(client)
        assert (kwargs["EndpointName"], kwargs["InferenceComponentName"]) == ("my-embed-endpoint", "my-component")
        assert json.loads(kwargs["Body"]) == EXPECTED_BODY
        assert len(result.data) == 2

    def test_router_deployment_with_model_id(self):
        router = Router(
            model_list=[
                {
                    "model_name": "my-embeddings",
                    "litellm_params": {
                        "model": "sagemaker/openai/my-embed-endpoint",
                        "model_id": "my-component",
                        "extra_body": {"model": "served-embedding-model"},
                        "aws_region_name": "us-east-1",
                        "aws_access_key_id": "test-key",
                        "aws_secret_access_key": "test-secret",
                    },
                    "model_info": {"mode": "embedding"},
                }
            ]
        )
        session, client = fake_sagemaker_session(OPENAI_RESPONSE)
        with patch("boto3.Session", return_value=session):
            result = asyncio.run(router.aembedding(model="my-embeddings", input=["hello", "good morning"]))

        kwargs = invoke_kwargs(client)
        assert (kwargs["EndpointName"], kwargs["InferenceComponentName"]) == ("my-embed-endpoint", "my-component")
        assert json.loads(kwargs["Body"]) == EXPECTED_BODY
        assert [row["index"] for row in result.data] == [0, 1]
