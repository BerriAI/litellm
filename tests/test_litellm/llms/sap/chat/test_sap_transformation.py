import warnings
import pytest
from pydantic import ValidationError


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

class TestGeminiReasoningNormalization:
    """Unit tests for _normalize_gemini_reasoning.

    Verifies that list-shaped reasoning_content from Gemini is coerced to str
    before model_validate is called, without touching other shapes.
    """

    def _make_final_result(self, reasoning_content):
        """Helper: build a minimal final_result dict with given reasoning_content."""
        result = {
            "id": "test-id",
            "object": "chat.completion",
            "model": "gemini-2.0-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        if reasoning_content is not None:
            result["choices"][0]["message"]["reasoning_content"] = reasoning_content
        return result

    def test_list_shaped_is_joined_to_string(self):
        """Gemini list of thought dicts is joined into a newline-separated string."""
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        final_result = self._make_final_result(
            [
                {"thought": "First thought.", "signature": "sig1"},
                {"thought": "Second thought.", "signature": "sig2"},
            ]
        )
        GenAIHubOrchestrationConfig._normalize_gemini_reasoning(final_result)
        assert (
            final_result["choices"][0]["message"]["reasoning_content"]
            == "First thought.\n\nSecond thought."
        )

    def test_string_reasoning_content_is_untouched(self):
        """A reasoning_content that is already a str is left as-is."""
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        final_result = self._make_final_result("already a string")
        GenAIHubOrchestrationConfig._normalize_gemini_reasoning(final_result)
        assert final_result["choices"][0]["message"]["reasoning_content"] == "already a string"

    def test_missing_reasoning_content_is_untouched(self):
        """A message without reasoning_content is left unchanged."""
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        final_result = self._make_final_result(None)
        # reasoning_content key is absent — _make_final_result(None) does not add it
        assert "reasoning_content" not in final_result["choices"][0]["message"]
        GenAIHubOrchestrationConfig._normalize_gemini_reasoning(final_result)
        assert "reasoning_content" not in final_result["choices"][0]["message"]

    def test_empty_list_becomes_none(self):
        """An empty list collapses to None so model_validate sees no reasoning."""
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        final_result = self._make_final_result([])
        GenAIHubOrchestrationConfig._normalize_gemini_reasoning(final_result)
        assert final_result["choices"][0]["message"]["reasoning_content"] is None

    def test_model_validate_succeeds_after_normalization(self):
        """model_validate no longer raises after normalisation."""
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig
        from litellm.types.utils import ModelResponse

        final_result = self._make_final_result(
            [{"thought": "Thinking hard.", "signature": "abc"}]
        )
        GenAIHubOrchestrationConfig._normalize_gemini_reasoning(final_result)
        response = ModelResponse.model_validate(final_result)
        assert response.choices[0].message.reasoning_content == "Thinking hard."


class TestReasoningCapability:
    """Unit tests for reasoning_effort / thinking parameter routing.

    Verifies that capable models expose the params and that they land in
    model.params, while non-capable models have them silently dropped.
    """

    def _transform(self, model: str, **kwargs) -> dict:
        """Run transform_request and return the parsed body."""
        from unittest.mock import MagicMock
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        cfg = GenAIHubOrchestrationConfig()
        logging_obj = MagicMock()
        return cfg.transform_request(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            optional_params=dict(kwargs),
            litellm_params={},
            headers={},
        )

    def _model_params(self, body: dict) -> dict:
        return body["config"]["modules"]["prompt_templating"]["model"]["params"]

    # --- get_supported_openai_params ---

    def test_reasoning_params_exposed_for_claude_3_7(self):
        """reasoning_effort and thinking appear for claude-3-7 models."""
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        cfg = GenAIHubOrchestrationConfig()
        params = cfg.get_supported_openai_params("anthropic--claude-3-7-sonnet")
        assert "reasoning_effort" in params
        assert "thinking" in params

    def test_reasoning_params_exposed_for_claude_4(self):
        """reasoning_effort and thinking appear for claude-4 models."""
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        cfg = GenAIHubOrchestrationConfig()
        params = cfg.get_supported_openai_params("anthropic--claude-4-opus")
        assert "reasoning_effort" in params
        assert "thinking" in params

    def test_reasoning_params_absent_for_gpt4o(self):
        """reasoning_effort and thinking are not exposed for gpt-4o."""
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        cfg = GenAIHubOrchestrationConfig()
        params = cfg.get_supported_openai_params("gpt-4o")
        assert "reasoning_effort" not in params
        assert "thinking" not in params

    # --- transform_request model.params ---

    def test_reasoning_effort_lands_in_model_params_for_o3(self):
        """reasoning_effort is forwarded into model.params for o-series models."""
        body = self._transform("o3", reasoning_effort="high")
        assert self._model_params(body).get("reasoning_effort") == "high"

    def test_thinking_lands_in_model_params_for_claude_3_7(self):
        """thinking dict is forwarded into model.params for claude-3-7."""
        thinking = {"type": "enabled", "budget_tokens": 8000}
        body = self._transform("anthropic--claude-3-7-sonnet", thinking=thinking)
        assert self._model_params(body).get("thinking") == thinking

    def test_reasoning_effort_dropped_for_non_capable_model(self):
        """reasoning_effort is silently dropped for models that don't support it."""
        body = self._transform("gpt-4o", reasoning_effort="high")
        assert "reasoning_effort" not in self._model_params(body)


class TestCacheControl:
    """Unit tests for cache_control preservation on message content parts.

    Verifies that TextContent with extra fields (cache_control) survives
    pydantic validation and reaches the SAP payload intact.
    """

    def _transform(self, model: str, messages: list) -> dict:
        from unittest.mock import MagicMock
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

        cfg = GenAIHubOrchestrationConfig()
        return cfg.transform_request(
            model=model,
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

    def _template(self, body: dict) -> list:
        return body["config"]["modules"]["prompt_templating"]["prompt"]["template"]

    def test_cache_control_preserved_on_text_content(self):
        """cache_control on a TextContent part survives validation and appears in payload."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
        body = self._transform("anthropic--claude-3-5-sonnet", messages)
        template = self._template(body)
        content = template[0]["content"]
        assert isinstance(content, list)
        assert content[0].get("cache_control") == {"type": "ephemeral"}

    def test_plain_text_content_unaffected(self):
        """Text content without cache_control still serialises cleanly."""
        messages = [{"role": "user", "content": "Hello"}]
        body = self._transform("anthropic--claude-3-5-sonnet", messages)
        template = self._template(body)
        assert template[0]["content"] == "Hello"

    def test_cache_control_preserved_on_multiple_parts(self):
        """cache_control is preserved on each part independently."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part A", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "Part B"},
                ],
            }
        ]
        body = self._transform("anthropic--claude-3-5-sonnet", messages)
        content = self._template(body)[0]["content"]
        assert content[0].get("cache_control") == {"type": "ephemeral"}
        assert "cache_control" not in content[1]

class TestModelVersionAdvertisement:
    """model_version is advertised in get_supported_openai_params and lands in
    model.version (not model.params) in the serialised request body.
    """

    def _cfg(self):
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig
        cfg = GenAIHubOrchestrationConfig()
        return cfg

    def _transform(self, model: str, **kwargs) -> dict:
        cfg = self._cfg()
        return cfg.transform_request(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            optional_params=dict(kwargs),
            litellm_params={},
            headers={},
        )

    def test_model_version_advertised_for_all_models(self):
        for model in ("gpt-4o", "anthropic--claude-4-sonnet", "gemini-1.5-pro"):
            params = self._cfg().get_supported_openai_params(model)
            assert "model_version" in params, f"model_version missing for {model}"

    def test_model_version_lands_in_model_version_field(self):
        body = self._transform("gpt-4o", model_version="1.2.3")
        pt = body["config"]["modules"]["prompt_templating"]
        assert pt["model"]["version"] == "1.2.3"

    def test_model_version_absent_from_model_params(self):
        body = self._transform("gpt-4o", model_version="1.2.3")
        pt = body["config"]["modules"]["prompt_templating"]
        assert "model_version" not in pt["model"]["params"]

    def test_model_version_defaults_to_latest(self):
        body = self._transform("gpt-4o")
        pt = body["config"]["modules"]["prompt_templating"]
        assert pt["model"]["version"] == "latest"


class TestToolChoiceForwarding:
    """tool_choice is no longer silently dropped.

    When tools are present it must appear in the prompt template dict.
    Without tools it is omitted (there is nothing to choose from).
    """

    _TOOL = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }

    def _transform(self, model: str, **kwargs) -> dict:
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig
        cfg = GenAIHubOrchestrationConfig()
        return cfg.transform_request(
            model=model,
            messages=[{"role": "user", "content": "What is the weather?"}],
            optional_params=dict(kwargs),
            litellm_params={},
            headers={},
        )

    def _prompt(self, body: dict) -> dict:
        return body["config"]["modules"]["prompt_templating"]["prompt"]

    def test_tool_choice_required_forwarded_with_tools(self):
        body = self._transform("gpt-4o", tools=[self._TOOL], tool_choice="required")
        prompt = self._prompt(body)
        assert prompt.get("tool_choice") == "required"

    def test_tool_choice_auto_forwarded_with_tools(self):
        body = self._transform("gpt-4o", tools=[self._TOOL], tool_choice="auto")
        assert self._prompt(body).get("tool_choice") == "auto"

    def test_tool_choice_dict_forwarded_with_tools(self):
        tc = {"type": "function", "function": {"name": "get_weather"}}
        body = self._transform("gpt-4o", tools=[self._TOOL], tool_choice=tc)
        assert self._prompt(body).get("tool_choice") == tc

    def test_tool_choice_absent_without_tools(self):
        """tool_choice without tools must not appear in the payload."""
        body = self._transform("gpt-4o", tool_choice="required")
        assert "tool_choice" not in self._prompt(body)

    def test_tool_choice_not_in_model_params(self):
        """tool_choice must never bleed into model.params."""
        body = self._transform("gpt-4o", tools=[self._TOOL], tool_choice="required")
        pt = body["config"]["modules"]["prompt_templating"]
        assert "tool_choice" not in pt["model"]["params"]


class TestTimeoutAndMaxRetries:
    """timeout and max_retries land in model-level sibling fields,
    not inside model.params.
    """

    def _transform(self, model: str, **kwargs) -> dict:
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig
        cfg = GenAIHubOrchestrationConfig()
        return cfg.transform_request(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            optional_params=dict(kwargs),
            litellm_params={},
            headers={},
        )

    def _model(self, body: dict) -> dict:
        return body["config"]["modules"]["prompt_templating"]["model"]

    def test_timeout_and_max_retries_advertised(self):
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig
        cfg = GenAIHubOrchestrationConfig()
        for model in ("gpt-4o", "anthropic--claude-4-sonnet"):
            params = cfg.get_supported_openai_params(model)
            assert "timeout" in params, f"timeout missing for {model}"
            assert "max_retries" in params, f"max_retries missing for {model}"

    def test_timeout_lands_at_model_level(self):
        body = self._transform("gpt-4o", timeout=120)
        model = self._model(body)
        assert model.get("timeout") == 120
        assert "timeout" not in model.get("params", {})

    def test_max_retries_lands_at_model_level(self):
        body = self._transform("gpt-4o", max_retries=3)
        model = self._model(body)
        assert model.get("max_retries") == 3
        assert "max_retries" not in model.get("params", {})

    def test_timeout_and_max_retries_together(self):
        body = self._transform("gpt-4o", timeout=60, max_retries=2)
        model = self._model(body)
        assert model.get("timeout") == 60
        assert model.get("max_retries") == 2
        assert "timeout" not in model.get("params", {})
        assert "max_retries" not in model.get("params", {})

    def test_absent_when_not_passed(self):
        """Neither key appears in the serialised body when not supplied."""
        body = self._transform("gpt-4o", temperature=0.7)
        model = self._model(body)
        assert "timeout" not in model
        assert "max_retries" not in model


class TestUserForwarding:
    """user param is advertised and lands in model.params (correct per SDK v2)."""

    def _cfg(self):
        from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig
        return GenAIHubOrchestrationConfig()

    def _transform(self, **kwargs) -> dict:
        cfg = self._cfg()
        return cfg.transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            optional_params=dict(kwargs),
            litellm_params={},
            headers={},
        )

    def test_user_advertised(self):
        params = self._cfg().get_supported_openai_params("gpt-4o")
        assert "user" in params

    def test_user_lands_in_model_params(self):
        body = self._transform(user="uid-abc123")
        pt = body["config"]["modules"]["prompt_templating"]
        assert pt["model"]["params"].get("user") == "uid-abc123"
