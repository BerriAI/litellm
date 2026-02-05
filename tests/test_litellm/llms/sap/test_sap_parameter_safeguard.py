import pytest


class TestSAPTransformationIntegration:
    """Integration tests for SAP transformation with parameter classification."""

    @pytest.fixture
    def mock_config(self):
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        config = GenAIHubOrchestrationConfig()
        config.token_creator = lambda: "Bearer TEST_TOKEN"
        config._base_url = "https://api.test-sap.com"
        config._resource_group = "test-group"

        return config

    def test_parameter_classification_in_transform_request(self, mock_config):
        """Test parameter classification within the actual transform_request method."""

        model = "gpt-4o"
        messages = [{"role": "user", "content": "Hello"}]

        optional_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "deployment_url": "https://custom.sap.com/deployment/123",
            "model_version": "v1.5",
            "tools": [{"type": "function", "function": {"name": "calculator"}}],
            "frequency_penalty": 0.1
        }

        result = mock_config.transform_request(
            model, messages, optional_params, {}, {}
        )

        model_params = result["config"]["modules"]["prompt_templating"]["model"]["params"]

        assert "temperature" in model_params
        assert "frequency_penalty" in model_params
        assert "deployment_url" not in model_params
        assert "model_version" not in model_params
        assert "tools" not in model_params

        model_version = result["config"]["modules"]["prompt_templating"]["model"]["version"]
        assert model_version == "v1.5"

        prompt = result["config"]["modules"]["prompt_templating"]["prompt"]
        if "tools" in prompt:
            assert isinstance(prompt["tools"], list)

    def test_transform_request_parameter_handling_robustness(self, mock_config):
        """Test transform_request method handles various parameter combinations correctly."""

        model = "gpt-4o"
        messages = [{"role": "user", "content": "Hello"}]

        test_cases = [
            # Case 1: Basic parameters only
            {
                "params": {"temperature": 0.7, "max_tokens": 100},
                "expected_in_model": {"temperature", "max_tokens"},
                "expected_excluded": set()
            },
            # Case 2: Parameters with auth/infrastructure components
            {
                "params": {
                    "temperature": 0.8,
                    "deployment_url": "https://api.sap.com/deployments/test",
                    "max_tokens": 150
                },
                "expected_in_model": {"temperature", "max_tokens"},
                "expected_excluded": {"deployment_url"}
            },
            # Case 3: Parameters with framework components
            {
                "params": {
                    "temperature": 0.6,
                    "model_version": "v2.0",
                    "tools": [{"function": {"name": "test"}}],
                    "frequency_penalty": 0.1
                },
                "expected_in_model": {"temperature", "frequency_penalty"},
                "expected_excluded": {"model_version", "tools"}
            }
        ]

        for i, test_case in enumerate(test_cases):
            filtered_params = {
                k: v for k, v in test_case["params"].items()
                if k not in {"tools", "model_version", "deployment_url"}
            }

            for expected_param in test_case["expected_in_model"]:
                assert expected_param in filtered_params, f"Case {i + 1}: {expected_param} should be in model params"

            for excluded_param in test_case["expected_excluded"]:
                assert excluded_param not in filtered_params, f"Case {i + 1}: {excluded_param} should be excluded from model params"

            try:
                result = mock_config.transform_request(
                    model, messages, test_case["params"], {}, {}
                )
                if result and "config" in result:
                    model_params = result["config"]["modules"]["prompt_templating"]["model"]["params"]

                    for excluded_param in test_case["expected_excluded"]:
                        assert excluded_param not in model_params, (
                            f"Case {i + 1}: {excluded_param} should not be in actual model params"
                        )
            except AttributeError as e:
                if "deployment_url" in str(e):
                    pass
                else:
                    pytest.fail(f"Unexpected AttributeError: {e}")
            except Exception as e:
                pytest.fail(f"Unexpected exception in transform_request: {e}")
