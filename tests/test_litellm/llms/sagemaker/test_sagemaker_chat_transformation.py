import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))
from litellm.llms.sagemaker.chat.transformation import SagemakerChatConfig


class TestSagemakerChatSignRequest:
    def setup_method(self):
        self.config = SagemakerChatConfig()

    @patch.object(SagemakerChatConfig, "_sign_request")
    def test_sign_request_injects_model_id_header(self, mock_sign_request):
        """
        Test that model_id in optional_params is injected as the
        X-Amzn-SageMaker-Inference-Component header before signing.
        """
        mock_sign_request.return_value = ({"Authorization": "signed"}, b'{}')

        headers = {"Content-Type": "application/json"}
        optional_params = {"model_id": "my-inference-component"}

        self.config.sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data={"messages": []},
            api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/my-endpoint/invocations",
            model="my-endpoint",
        )

        # Verify _sign_request was called with the inference component header
        call_kwargs = mock_sign_request.call_args
        signed_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert signed_headers["X-Amzn-SageMaker-Inference-Component"] == "my-inference-component"

    @patch.object(SagemakerChatConfig, "_sign_request")
    def test_sign_request_no_model_id_no_header(self, mock_sign_request):
        """
        Test that when model_id is not provided, the inference component
        header is not added.
        """
        mock_sign_request.return_value = ({"Authorization": "signed"}, b'{}')

        headers = {"Content-Type": "application/json"}
        optional_params = {}

        self.config.sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data={"messages": []},
            api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/my-endpoint/invocations",
            model="my-endpoint",
        )

        call_kwargs = mock_sign_request.call_args
        signed_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert "X-Amzn-SageMaker-Inference-Component" not in signed_headers

    @patch.object(SagemakerChatConfig, "_sign_request")
    def test_sign_request_model_id_with_none_headers(self, mock_sign_request):
        """
        Test that model_id injection works even when headers is initially None.
        """
        mock_sign_request.return_value = ({"Authorization": "signed"}, b'{}')

        optional_params = {"model_id": "component-abc"}

        self.config.sign_request(
            headers=None,
            optional_params=optional_params,
            request_data={"messages": []},
            api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/my-endpoint/invocations",
            model="my-endpoint",
        )

        call_kwargs = mock_sign_request.call_args
        signed_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert signed_headers is not None
        assert signed_headers["X-Amzn-SageMaker-Inference-Component"] == "component-abc"
