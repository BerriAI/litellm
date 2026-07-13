import warnings
import pytest
from pydantic import ValidationError


class TestGeminiThinkingNormalization:
    """Regression tests for list-shaped reasoning_content from Gemini models.

    AI Core returns Gemini thinking as:
      message.reasoning_content = [{"thought": "...", "signature": "..."}]

    ModelResponse.reasoning_content is typed Optional[str], so model_validate
    would crash without the normalization step in transform_response.
    """

    @pytest.fixture
    def normalize(self):
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        return GenAIHubOrchestrationConfig._normalize_gemini_thinking

    def test_list_reasoning_content_converted_to_thinking_blocks(self, normalize):
        raw = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "42",
                        "reasoning_content": [
                            {"thought": "let me think", "signature": "sig-abc"},
                        ],
                    }
                }
            ]
        }
        out = normalize(raw)
        msg = out["choices"][0]["message"]
        assert msg["reasoning_content"] == "let me think"
        assert msg["thinking_blocks"] == [
            {"type": "thinking", "thinking": "let me think", "signature": "sig-abc"}
        ]

    def test_multiple_thought_blocks_joined(self, normalize):
        raw = {
            "choices": [
                {
                    "message": {
                        "reasoning_content": [
                            {"thought": "first", "signature": "s1"},
                            {"thought": "second", "signature": "s2"},
                        ],
                        "content": "done",
                    }
                }
            ]
        }
        out = normalize(raw)
        msg = out["choices"][0]["message"]
        assert msg["reasoning_content"] == "first\nsecond"
        assert len(msg["thinking_blocks"]) == 2

    def test_string_reasoning_content_untouched(self, normalize):
        raw = {
            "choices": [
                {"message": {"reasoning_content": "plain string", "content": "hi"}}
            ]
        }
        out = normalize(raw)
        assert out["choices"][0]["message"]["reasoning_content"] == "plain string"
        assert "thinking_blocks" not in out["choices"][0]["message"]

    def test_no_reasoning_content_untouched(self, normalize):
        raw = {"choices": [{"message": {"content": "hi"}}]}
        out = normalize(raw)
        assert "thinking_blocks" not in out["choices"][0]["message"]
        assert "reasoning_content" not in out["choices"][0]["message"]

    def test_model_validate_succeeds_after_normalization(self, normalize):
        from litellm.types.utils import ModelResponse

        raw = {
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "gemini-2.5-pro",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": [
                            {"thought": "thinking...", "signature": "abc"}
                        ],
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "fn", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        normalized = normalize(raw)
        response = ModelResponse.model_validate(normalized)
        assert response.choices[0].message.reasoning_content == "thinking..."
        assert response.choices[0].message.thinking_blocks[0]["thinking"] == "thinking..."
        assert response.choices[0].message.thinking_blocks[0]["signature"] == "abc"

    def test_thought_signature_preserved_on_tool_call_response(self, normalize):
        raw = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": [
                            {"thought": "I need to call a tool", "signature": "sig-xyz"}
                        ],
                        "tool_calls": [
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q":"x"}'},
                            }
                        ],
                    }
                }
            ]
        }
        out = normalize(raw)
        msg = out["choices"][0]["message"]
        assert msg["thinking_blocks"][0]["signature"] == "sig-xyz"
        assert msg["reasoning_content"] == "I need to call a tool"


class TestReasoningParamSupport:
    @pytest.fixture
    def mock_config(self):
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        config = GenAIHubOrchestrationConfig()
        config.token_creator = lambda: "Bearer TEST_TOKEN"
        config._base_url = "https://api.test-sap.com"
        config._resource_group = "test-group"
        return config

    def test_reasoning_params_included_for_o_series(self, mock_config):
        params = mock_config.get_supported_openai_params("o4-mini")
        assert "reasoning_effort" in params
        assert "thinking" in params

    def test_reasoning_params_included_for_anthropic_claude4(self, mock_config):
        params = mock_config.get_supported_openai_params("anthropic--claude-4.5-sonnet")
        assert "reasoning_effort" in params
        assert "thinking" in params

    def test_reasoning_params_included_for_anthropic_claude37(self, mock_config):
        params = mock_config.get_supported_openai_params(
            "anthropic--claude-3-7-sonnet-20250219"
        )
        assert "reasoning_effort" in params
        assert "thinking" in params

    def test_reasoning_params_excluded_for_anthropic_claude3(self, mock_config):
        # claude-3-haiku does not support extended thinking
        params = mock_config.get_supported_openai_params("anthropic--claude-3-haiku")
        assert "reasoning_effort" not in params
        assert "thinking" not in params

    def test_cohere_reasoning_model_supports_thinking_only(self, mock_config):
        # Cohere accepts thinking but not reasoning_effort
        params = mock_config.get_supported_openai_params("cohere--command-a-reasoning")
        assert "thinking" in params
        assert "reasoning_effort" not in params

    def test_cohere_non_reasoning_model_excluded(self, mock_config):
        params = mock_config.get_supported_openai_params("cohere-reranker")
        assert "reasoning_effort" not in params
        assert "thinking" not in params

    def test_reasoning_params_excluded_for_gpt_model(self, mock_config):
        params = mock_config.get_supported_openai_params("gpt-4o")
        assert "reasoning_effort" not in params
        assert "thinking" not in params

    def test_reasoning_params_excluded_for_model_starting_with_o_but_not_o_series(
        self, mock_config
    ):
        params = mock_config.get_supported_openai_params("oceanai-model")
        assert "reasoning_effort" not in params
        assert "thinking" not in params

    def test_reasoning_params_excluded_for_mistralai(self, mock_config):
        params = mock_config.get_supported_openai_params(
            "mistralai--mistral-large-instruct"
        )
        assert "reasoning_effort" not in params
        assert "thinking" not in params

    def test_reasoning_effort_reaches_model_params(self, mock_config):
        result = mock_config.transform_request(
            model="o4-mini",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"reasoning_effort": "low"},
            litellm_params={},
            headers={},
        )
        model_params = result["config"]["modules"]["prompt_templating"]["model"][
            "params"
        ]
        assert model_params["reasoning_effort"] == "low"

    def test_thinking_reaches_model_params(self, mock_config):
        thinking = {"type": "enabled", "budget_tokens": 8000}
        result = mock_config.transform_request(
            model="anthropic--claude-3-7-sonnet-20250219",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"thinking": thinking},
            litellm_params={},
            headers={},
        )
        model_params = result["config"]["modules"]["prompt_templating"]["model"][
            "params"
        ]
        assert model_params["thinking"] == thinking


class TestSAPTransformationIntegration:
    """Integration tests for SAP transformation."""

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
            "frequency_penalty": 0.1,
        }

        result = mock_config.transform_request(model, messages, optional_params, {}, {})

        model_params = result["config"]["modules"]["prompt_templating"]["model"][
            "params"
        ]

        assert "temperature" in model_params
        assert "frequency_penalty" in model_params
        assert "deployment_url" not in model_params
        assert "model_version" not in model_params
        assert "tools" not in model_params

        model_version = result["config"]["modules"]["prompt_templating"]["model"][
            "version"
        ]
        assert model_version == "v1.5"

        prompt = result["config"]["modules"]["prompt_templating"]["prompt"]
        if "tools" in prompt:
            assert isinstance(prompt["tools"], list)
            for tool in prompt["tools"]:
                assert (
                    tool["function"]["parameters"]["type"] == "object"
                ), "SAP API requires parameters.type == 'object'"
                assert "properties" in tool["function"]["parameters"]

    def test_transform_request_parameter_handling_robustness(self, mock_config):
        """Test transform_request method handles various parameter combinations correctly."""

        model = "gpt-4o"
        messages = [{"role": "user", "content": "Hello"}]

        test_cases = [
            # Case 1: Basic parameters only
            {
                "params": {"temperature": 0.7, "max_tokens": 100},
                "expected_in_model": {"temperature", "max_tokens"},
                "expected_excluded": set(),
            },
            # Case 2: Parameters with auth/infrastructure components
            {
                "params": {
                    "temperature": 0.8,
                    "deployment_url": "https://api.sap.com/deployments/test",
                    "max_tokens": 150,
                },
                "expected_in_model": {"temperature", "max_tokens"},
                "expected_excluded": {"deployment_url"},
            },
            # Case 3: Parameters with framework components
            {
                "params": {
                    "temperature": 0.6,
                    "model_version": "v2.0",
                    "tools": [{"function": {"name": "test"}}],
                    "frequency_penalty": 0.1,
                },
                "expected_in_model": {"temperature", "frequency_penalty"},
                "expected_excluded": {"model_version", "tools"},
            },
        ]

        for i, test_case in enumerate(test_cases):
            filtered_params = {
                k: v
                for k, v in test_case["params"].items()
                if k not in {"tools", "model_version", "deployment_url"}
            }

            for expected_param in test_case["expected_in_model"]:
                assert (
                    expected_param in filtered_params
                ), f"Case {i + 1}: {expected_param} should be in model params"

            for excluded_param in test_case["expected_excluded"]:
                assert (
                    excluded_param not in filtered_params
                ), f"Case {i + 1}: {excluded_param} should be excluded from model params"

            result = mock_config.transform_request(
                model, messages, test_case["params"], {}, {}
            )
            if result and "config" in result:
                model_params = result["config"]["modules"]["prompt_templating"][
                    "model"
                ]["params"]

                for excluded_param in test_case["expected_excluded"]:
                    assert (
                        excluded_param not in model_params
                    ), f"Case {i + 1}: {excluded_param} should not be in actual model params"

    def test_config_transform_with_response_format_json_object(self, mock_config):
        expected_dict = {
            "config": {
                "modules": {
                    "prompt_templating": {
                        "prompt": {
                            "template": [
                                {
                                    "role": "user",
                                    "content": "First man on the moon, answer in json",
                                }
                            ],
                            "response_format": {"type": "json_object"},
                        },
                        "model": {"name": "gpt-4o", "params": {}, "version": "latest"},
                    }
                },
            }
        }
        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "First man on the moon, answer in json"}
            ],
            optional_params={
                "response_format": {"type": "json_object"},
                "deployment_url": "shouldn't be in results",
            },
            litellm_params={},
            headers={},
        )
        assert config == expected_dict

    def test_config_transform_with_response_format_json_schema(self, mock_config):

        expected_response_format = {
            "type": "json_schema",
            "json_schema": {
                "description": "Schema for person information",
                "name": "person_info",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The person's full name",
                        },
                        "age": {
                            "type": "integer",
                            "description": "The person's age in years",
                        },
                        "occupation": {
                            "type": "string",
                            "description": "The person's job title",
                        },
                    },
                    "required": ["name", "age", "occupation"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "First man on the moon, answer in json"}
            ],
            optional_params={
                "response_format": expected_response_format,
                "deployment_url": "shouldn't be in results",
            },
            litellm_params={},
            headers={},
        )
        assert (
            config["config"]["modules"]["prompt_templating"]["prompt"][
                "response_format"
            ]
            == expected_response_format
        )
        assert (
            len(config["config"]["modules"]["prompt_templating"]["model"]["params"])
            == 0
        )

    def test_config_transform_with_stream(self, mock_config):
        expected_dict = {
            "config": {
                "modules": {
                    "prompt_templating": {
                        "prompt": {
                            "template": [
                                {"role": "user", "content": "Hello, how are you?"}
                            ]
                        },
                        "model": {
                            "name": "anthropic--claude-4-sonnet",
                            "params": {},
                            "version": "latest",
                        },
                    }
                },
                "stream": {"chunk_size": 10},
            }
        }
        config = mock_config.transform_request(
            model="anthropic--claude-4-sonnet",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            optional_params={
                "stream": True,
                "stream_options": {"chunk_size": 10},
                "model_version": "latest",
                "deployment_url": "shouldn't be in results",
            },
            litellm_params={},
            headers={},
        )

        assert config == expected_dict

    def test_sap_placeholder_defaults(self, mock_config):
        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello. Answer {{ ?user_query }}"}],
            optional_params={
                "deployment_url": "shouldn't be in results",
                "placeholder_defaults": {"user_query": "default value"},
            },
            litellm_params={},
            headers={},
        )

        assert config["config"]["modules"]["prompt_templating"]["prompt"][
            "defaults"
        ] == {"user_query": "default value"}
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

    def test_sap_placeholder_values(self, mock_config):
        placeholder_values = {"user_query": "Some text"}
        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello. Answer {{ ?user_query }}"}],
            optional_params={
                "deployment_url": "shouldn't be in results",
                "placeholder_values": placeholder_values,
            },
            litellm_params={},
            headers={},
        )

        assert config["placeholder_values"] == placeholder_values
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

    def test_sap_grounding(self, mock_config):
        grounding_config = {
            "type": "document_grounding_service",
            "config": {
                "filters": [
                    {
                        "id": "s3-docs",
                        "data_repository_type": "vector",
                        "search_config": {"max_chunk_count": 2},
                        "data_repositories": ["123456890-test"],
                    }
                ],
                "placeholders": {
                    "input": ["user_query"],
                    "output": "grounding_response",
                },
                "metadata_params": [
                    "source",
                    "webUrl",
                    "title",
                    "mimeType",
                    "fileSuffix",
                ],
            },
        }
        placeholder_values = {"user_query": "Some text"}
        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "Hello. Answer {{ ?user_query }} using context: {{ ?grounding_response }}",
                }
            ],
            optional_params={
                "deployment_url": "shouldn't be in results",
                "grounding": grounding_config,
                "placeholder_values": placeholder_values,
            },
            litellm_params={},
            headers={},
        )
        assert config["placeholder_values"] == placeholder_values
        modules = config["config"]["modules"]
        assert modules["grounding"]["type"] == "document_grounding_service"
        assert (
            modules["grounding"]["config"]["placeholders"]["output"]
            == "grounding_response"
        )
        assert (
            modules["grounding"]["config"]["filters"][0]["data_repository_type"]
            == "vector"
        )
        assert modules["prompt_templating"]["model"]["params"] == {}

    def test_grounding_search_config_rejects_both_count_fields(self, mock_config):
        with pytest.raises(ValidationError):
            mock_config.transform_request(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                optional_params={
                    "grounding": {
                        "type": "document_grounding_service",
                        "config": {
                            "filters": [
                                {
                                    "data_repository_type": "vector",
                                    "search_config": {
                                        "max_chunk_count": 2,
                                        "max_document_count": 5,
                                    },
                                }
                            ],
                            "placeholders": {"input": ["q"], "output": "r"},
                        },
                    }
                },
                litellm_params={},
                headers={},
            )

    def test_sap_filtering(self, mock_config):
        filtering_config_azure = {
            "input": {
                "filters": [
                    {
                        "type": "azure_content_safety",
                        "config": {
                            "hate": 0,
                            "sexual": 0,
                            "violence": 0,
                            "self_harm": 0,
                        },
                    }
                ]
            },
            "output": {
                "filters": [
                    {
                        "type": "azure_content_safety",
                        "config": {
                            "hate": 0,
                            "sexual": 0,
                            "violence": 0,
                            "self_harm": 0,
                        },
                    }
                ]
            },
        }
        filtering_config_llama = {
            "input": {
                "filters": [
                    {
                        "type": "llama_guard_3_8b",
                        "config": {"hate": True, "elections": True},
                    }
                ]
            },
            "output": {
                "filters": [
                    {
                        "type": "llama_guard_3_8b",
                        "config": {"hate": True, "elections": True},
                    }
                ]
            },
        }
        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello."}],
            optional_params={
                "deployment_url": "shouldn't be in results",
                "filtering": filtering_config_azure,
            },
            litellm_params={},
            headers={},
        )
        assert config["config"]["modules"]["filtering"] == filtering_config_azure
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello."}],
            optional_params={
                "deployment_url": "shouldn't be in results",
                "filtering": filtering_config_llama,
            },
            litellm_params={},
            headers={},
        )
        assert config["config"]["modules"]["filtering"] == filtering_config_llama
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

    def test_filtering_config_requires_at_least_one_property(self, mock_config):
        with pytest.raises(ValidationError) as exc_info:
            mock_config.transform_request(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={"filtering": {}},
                litellm_params={},
                headers={},
            )

        assert (
            "For using SAP Filtering Module you must provide at least one property"
            in str(exc_info.value)
        )

    def test_sap_masking(self, mock_config):
        masking_config = {
            "providers": [
                {
                    "type": "sap_data_privacy_integration",
                    "method": "anonymization",
                    "entities": [
                        {"type": "profile-address"},
                        {"type": "profile-email"},
                        {"type": "profile-phone"},
                        {"type": "profile-person"},
                        {"type": "profile-location"},
                    ],
                }
            ]
        }

        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello."}],
            optional_params={
                "deployment_url": "shouldn't be in results",
                "masking": masking_config,
            },
            litellm_params={},
            headers={},
        )
        assert config["config"]["modules"]["masking"] == masking_config
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

    def test_masking_config_requires_exactly_one_provider_list(self, mock_config):
        masking_config = {
            "providers": [
                {
                    "type": "sap_data_privacy_integration",
                    "method": "anonymization",
                    "entities": [
                        {"type": "profile-address"},
                        {"type": "profile-email"},
                        {"type": "profile-phone"},
                        {"type": "profile-person"},
                        {"type": "profile-location"},
                    ],
                }
            ],
            "masking_providers": [
                {
                    "type": "sap_data_privacy_integration",
                    "method": "anonymization",
                    "entities": [{"type": "profile-address"}],
                }
            ],
        }
        with pytest.raises(ValidationError) as exc_info:
            mock_config.transform_request(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={"masking": masking_config},
                litellm_params={},
                headers={},
            )

        assert "must set exactly one of: 'providers' or 'masking_providers'" in str(
            exc_info.value
        )

    def test_masking_providers_deprecated_emits_warning(self, mock_config):
        masking_config = {
            "masking_providers": [
                {
                    "type": "sap_data_privacy_integration",
                    "method": "anonymization",
                    "entities": [{"type": "profile-address"}],
                }
            ]
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mock_config.transform_request(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                optional_params={"masking": masking_config},
                litellm_params={},
                headers={},
            )
        assert any(
            issubclass(warning.category, DeprecationWarning)
            and "masking_providers" in str(warning.message)
            for warning in w
        ), "Expected DeprecationWarning for 'masking_providers'"

    def test_sap_translation(self, mock_config):
        translation_config = {
            "input": {
                "type": "sap_document_translation",
                "config": {"source_language": "en-US", "target_language": "de-DE"},
            },
            "output": {
                "type": "sap_document_translation",
                "config": {"source_language": "de-DE", "target_language": "fr-FR"},
            },
        }

        config = mock_config.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello."}],
            optional_params={
                "deployment_url": "shouldn't be in results",
                "translation": translation_config,
            },
            litellm_params={},
            headers={},
        )
        assert config["config"]["modules"]["translation"] == translation_config
        assert config["config"]["modules"]["prompt_templating"]["model"]["params"] == {}

    def test_translation_config_requires_at_least_one_property(self, mock_config):
        with pytest.raises(ValidationError) as exc_info:
            mock_config.transform_request(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={"translation": {}},
                litellm_params={},
                headers={},
            )

        assert (
            "TranslationModuleConfig requires at least one of 'input' or 'output'"
            in str(exc_info.value)
        )

    def test_sap_multiple_modules(self, mock_config):
        translation_config = {
            "input": {
                "type": "sap_document_translation",
                "config": {"source_language": "en-US", "target_language": "de-DE"},
            },
            "output": {
                "type": "sap_document_translation",
                "config": {"source_language": "de-DE", "target_language": "fr-FR"},
            },
        }
        for model in ["sap/gpt-5", "gpt-5"]:
            config = mock_config.transform_request(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello."}],
                optional_params={
                    "deployment_url": "shouldn't be in results",
                    "fallback_sap_modules": [
                        {
                            "model": model,
                            "messages": [{"role": "user", "content": "Hello world!"}],
                            "translation": translation_config,
                        }
                    ],
                },
                litellm_params={},
                headers={},
            )
            assert "translation" not in config["config"]["modules"][0]
            translation = config["config"]["modules"][1]["translation"]
            assert translation["input"]["config"]["source_language"] == "en-US"
            assert translation["input"]["config"]["target_language"] == "de-DE"
            assert translation["output"]["config"]["target_language"] == "fr-FR"
            assert (
                config["config"]["modules"][1]["prompt_templating"]["model"]["name"]
                == "gpt-5"
            )
            assert (
                config["config"]["modules"][0]["prompt_templating"]["model"]["name"]
                == "gpt-4o"
            )
            assert (
                config["config"]["modules"][0]["prompt_templating"]["model"]["params"]
                == {}
            )
            assert (
                config["config"]["modules"][1]["prompt_templating"]["prompt"][
                    "template"
                ][0]["content"]
                == "Hello world!"
            )
            assert (
                config["config"]["modules"][0]["prompt_templating"]["prompt"][
                    "template"
                ][0]["content"]
                == "Hello."
            )
            assert (
                config["config"]["modules"][1]["translation"]["input"]["type"]
                == "sap_document_translation"
            )
