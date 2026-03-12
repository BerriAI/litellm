import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from litellm.llms.sagemaker.completion.handler import SagemakerLLM
from litellm.types.utils import EmbeddingResponse


def test_embedding_uses_load_credentials_for_role_assumption():
    llm = SagemakerLLM()
    assumed_credentials = SimpleNamespace(
        access_key="assumed-access-key",
        secret_key="assumed-secret-key",
        token="assumed-session-token",
    )
    mocked_sagemaker_client = MagicMock()
    mocked_sagemaker_client.invoke_endpoint.return_value = {
        "Body": MagicMock(
            read=MagicMock(return_value=json.dumps([[0.1, 0.2, 0.3]]).encode("utf-8"))
        )
    }
    optional_params = {
        "aws_role_name": "arn:aws:iam::123456789012:role/CrossAccountRole",
        "aws_session_name": "litellm-session",
        "aws_region_name": "us-east-1",
    }

    mocked_boto3_client_factory = MagicMock(return_value=mocked_sagemaker_client)
    mocked_boto3_module = ModuleType("boto3")
    mocked_boto3_module.client = mocked_boto3_client_factory
    mocked_botocore_module = ModuleType("botocore")
    mocked_botocore_credentials_module = ModuleType("botocore.credentials")

    class MockCredentials:
        pass

    mocked_botocore_credentials_module.Credentials = MockCredentials
    mocked_botocore_module.credentials = mocked_botocore_credentials_module

    with patch.object(
        SagemakerLLM, "get_credentials", return_value=assumed_credentials
    ) as mock_get_credentials, patch.dict(
        sys.modules,
        {
            "boto3": mocked_boto3_module,
            "botocore": mocked_botocore_module,
            "botocore.credentials": mocked_botocore_credentials_module,
        },
    ):
        response = llm.embedding(
            model="sentence-transformers-model",
            input=["hello world"],
            model_response=EmbeddingResponse(),
            print_verbose=lambda *_: None,
            encoding=None,
            logging_obj=MagicMock(),
            optional_params=optional_params,
            litellm_params={},
        )

    assert isinstance(response, EmbeddingResponse)
    assert len(response.data or []) == 1

    get_credentials_kwargs = mock_get_credentials.call_args.kwargs
    assert (
        get_credentials_kwargs["aws_role_name"]
        == "arn:aws:iam::123456789012:role/CrossAccountRole"
    )
    assert get_credentials_kwargs["aws_session_name"] == "litellm-session"

    mocked_boto3_client_factory.assert_called_once_with(
        service_name="sagemaker-runtime",
        aws_access_key_id="assumed-access-key",
        aws_secret_access_key="assumed-secret-key",
        aws_session_token="assumed-session-token",
        region_name="us-east-1",
    )

    assert "aws_role_name" not in optional_params
    assert "aws_session_name" not in optional_params
