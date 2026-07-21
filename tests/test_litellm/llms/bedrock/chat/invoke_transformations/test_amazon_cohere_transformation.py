"""Regression tests for AmazonCohereConfig.map_openai_params.

Cohere on Bedrock (the classic `cohere.command-*` models) uses Cohere's
Generate API, not the Chat API that ``CohereChatConfig``/``CohereV2ChatConfig``
otherwise target -- and the Generate API genuinely supports ``num_generations``
(see https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html).

``AmazonCohereConfig`` delegates most of its param mapping to
``CohereChatConfig.map_openai_params`` for convenience, but must NOT inherit
the Chat-API-specific ``n`` guard added for issue #34111 -- that guard would
incorrectly raise/drop a parameter that Bedrock's Cohere models actually
support. These tests pin the original, correct Bedrock behavior.
"""

from litellm.llms.bedrock.chat.invoke_transformations.amazon_cohere_transformation import (
    AmazonCohereConfig,
)


class TestAmazonCohereConfigNHandling:
    def setup_method(self):
        self.config = AmazonCohereConfig()
        self.model = "cohere.command-text-v14"

    def test_n_greater_than_1_maps_to_num_generations_not_dropped(self):
        """Bedrock's Cohere Generate API genuinely supports num_generations;
        this must keep working exactly as before issue #34111's fix."""
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.6, "n": 2},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["num_generations"] == 2

    def test_n_greater_than_1_does_not_raise(self):
        """The Chat-API-specific UnsupportedParamsError must not apply here."""
        # Should not raise, unlike CohereChatConfig/CohereV2ChatConfig with n>1.
        result = self.config.map_openai_params(
            non_default_params={"n": 5},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["num_generations"] == 5

    def test_n_equal_1_still_maps_to_num_generations(self):
        """Unlike the Chat API path, n=1 is not special-cased here -- Bedrock's
        Generate API accepts num_generations=1 as a normal, explicit value."""
        result = self.config.map_openai_params(
            non_default_params={"n": 1},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["num_generations"] == 1

    def test_n_absent_does_not_set_num_generations(self):
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.6},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert "num_generations" not in result
        assert result["temperature"] == 0.6

    def test_other_params_still_mapped_via_cohere_chat_config(self):
        """Everything except n should still go through the shared mapping."""
        result = self.config.map_openai_params(
            non_default_params={
                "temperature": 0.6,
                "top_p": 1.0,
                "stop": ["END"],
                "n": 3,
            },
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["temperature"] == 0.6
        assert result["p"] == 1.0
        assert result["stop_sequences"] == ["END"]
        assert result["num_generations"] == 3
