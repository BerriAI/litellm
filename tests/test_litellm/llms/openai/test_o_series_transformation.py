import pytest

from litellm.llms.openai.chat.o_series_transformation import OpenAIOSeriesConfig

@pytest.mark.parametrize(
    "model_name,expected",
    [
        # Valid O-series models
        ("o1", True),
        ("o3", True),
        ("o4-mini", True),
        ("o3-mini", True),
        # Valid O-series models with provider prefix
        ("openai/o1", True),
        ("openai/o3", True),
        ("openai/o4-mini", True),
        ("openai/o3-mini", True),
        # Non-O-series models
        ("gpt-4", False),
        ("gpt-3.5-turbo", False),
        ("claude-3-opus", False),
        # Non-O-series models with provider prefix
        ("openai/gpt-4", False),
        ("openai/gpt-3.5-turbo", False),
        ("anthropic/claude-3-opus", False),
        # Edge cases
        ("o", False),  # Too short
        ("o5", False),  # Not a valid O-series model
        ("o1-", False),  # Invalid suffix
        ("o3_", False),  # Invalid suffix
    ],
)
def test_is_model_o_series_model(model_name: str, expected: bool):
    """
    Test that is_model_o_series_model correctly identifies O-series models.

    Args:
        model_name: The model name to test
        expected: The expected result (True if it should be identified as an O-series model)
    """
    config = OpenAIOSeriesConfig()
    assert (
        config.is_model_o_series_model(model_name) == expected
    ), f"Expected {model_name} to be {'an O-series model' if expected else 'not an O-series model'}"
