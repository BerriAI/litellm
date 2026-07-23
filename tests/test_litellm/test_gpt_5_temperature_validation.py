import pytest
import litellm


def test_gpt_5_5_temperature_validation():
    """Verify that calling gpt-5.5 with temperature != 1 raises UnsupportedParamsError locally."""
    # Test temperature=0.0 raises UnsupportedParamsError
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config = litellm.OpenAIGPT5Config()
        config.map_openai_params(
            non_default_params={"temperature": 0.0},
            optional_params={},
            model="gpt-5.5",
            drop_params=False,
        )

    # Test temperature=0.5 raises UnsupportedParamsError
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config = litellm.OpenAIGPT5Config()
        config.map_openai_params(
            non_default_params={"temperature": 0.5},
            optional_params={},
            model="gpt-5.5",
            drop_params=False,
        )

    # Test temperature=1.0 is valid and passes through
    config = litellm.OpenAIGPT5Config()
    optional_params = {}
    config.map_openai_params(
        non_default_params={"temperature": 1.0},
        optional_params=optional_params,
        model="gpt-5.5",
        drop_params=False,
    )
    assert optional_params.get("temperature") == 1.0

    # Test drop_params=True drops invalid temperature without error
    config = litellm.OpenAIGPT5Config()
    optional_params = {}
    config.map_openai_params(
        non_default_params={"temperature": 0.0},
        optional_params=optional_params,
        model="gpt-5.5",
        drop_params=True,
    )
    assert "temperature" not in optional_params


def test_gpt_5_6_temperature_validation():
    """Verify that calling gpt-5.6 variants with temperature != 1 raises UnsupportedParamsError locally."""
    for model in ("gpt-5.6", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "azure/gpt-5.6-sol"):
        with pytest.raises(litellm.utils.UnsupportedParamsError):
            config = litellm.OpenAIGPT5Config()
            config.map_openai_params(
                non_default_params={"temperature": 0.0},
                optional_params={},
                model=model,
                drop_params=False,
            )
