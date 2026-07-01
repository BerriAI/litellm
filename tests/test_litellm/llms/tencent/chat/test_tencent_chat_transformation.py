from unittest.mock import patch

from litellm.llms.tencent.chat.transformation import TencentChatConfig


def test_supported_openai_params_includes_thinking_and_reasoning_effort():
    config = TencentChatConfig()

    with patch(
        "litellm.llms.tencent.chat.transformation.supports_reasoning",
        return_value=True,
    ):
        params = config.get_supported_openai_params(model="tencent/deepseek-v4-pro")

    assert "thinking" in params
    assert "reasoning_effort" in params
    assert "stream" in params
    assert "temperature" in params


def test_supported_openai_params_excludes_thinking_without_reasoning_support():
    config = TencentChatConfig()

    with patch(
        "litellm.llms.tencent.chat.transformation.supports_reasoning",
        return_value=False,
    ):
        params = config.get_supported_openai_params(model="tencent/non-reasoning-model")

    assert "thinking" not in params
    assert "reasoning_effort" not in params
    assert "stream" in params


def test_map_openai_params_passes_thinking_dict_through():
    config = TencentChatConfig()
    with patch(
        "litellm.llms.tencent.chat.transformation.supports_reasoning",
        return_value=True,
    ):
        result = config.map_openai_params(
            non_default_params={"thinking": {"type": "enabled", "budget_tokens": 1024}},
            optional_params={},
            model="tencent/deepseek-v4-pro",
            drop_params=False,
        )

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_map_openai_params_converts_reasoning_effort_to_thinking():
    config = TencentChatConfig()
    with patch(
        "litellm.llms.tencent.chat.transformation.supports_reasoning",
        return_value=True,
    ):
        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "medium"},
            optional_params={},
            model="tencent/deepseek-v4-pro",
            drop_params=False,
        )

    assert result["thinking"] == {"type": "enabled"}


def test_map_openai_params_drops_none_reasoning_effort():
    config = TencentChatConfig()
    with patch(
        "litellm.llms.tencent.chat.transformation.supports_reasoning",
        return_value=True,
    ):
        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="tencent/deepseek-v4-pro",
            drop_params=False,
        )

    assert "thinking" not in result
    assert "reasoning_effort" not in result


def test_map_openai_params_thinking_priority_over_reasoning_effort():
    config = TencentChatConfig()
    with patch(
        "litellm.llms.tencent.chat.transformation.supports_reasoning",
        return_value=True,
    ):
        result = config.map_openai_params(
            non_default_params={
                "thinking": {"type": "enabled", "budget_tokens": 2048},
                "reasoning_effort": "high",
            },
            optional_params={},
            model="tencent/deepseek-v4-pro",
            drop_params=False,
        )

    assert result["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_map_openai_params_extracts_thinking_and_effort_from_optional_params():
    config = TencentChatConfig()
    result = config.map_openai_params(
        non_default_params={},
        optional_params={"thinking": {"type": "enabled"}, "reasoning_effort": "medium"},
        model="tencent/deepseek-v4-pro",
        drop_params=False,
    )

    assert "thinking" in result
    assert "reasoning_effort" not in result


def test_get_complete_url_default():
    config = TencentChatConfig()

    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="tencent/deepseek-v4-pro",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://tokenhub-intl.tencentcloudmaas.com/v1/chat/completions"


def test_get_complete_url_strips_trailing_slash():
    config = TencentChatConfig()

    url = config.get_complete_url(
        api_base="https://tokenhub-intl.tencentcloudmaas.com/v1/",
        api_key=None,
        model="tencent/deepseek-v4-pro",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://tokenhub-intl.tencentcloudmaas.com/v1/chat/completions"


def test_get_complete_url_custom_base_preserves_v1():
    config = TencentChatConfig()

    url = config.get_complete_url(
        api_base="https://tokenhub.tencentcloudmaas.com/v1",
        api_key=None,
        model="tencent/deepseek-v4-pro",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://tokenhub.tencentcloudmaas.com/v1/chat/completions"


def test_get_complete_url_adds_v1_to_custom_base():
    config = TencentChatConfig()

    url = config.get_complete_url(
        api_base="https://tokenhub.tencentcloudmaas.com",
        api_key=None,
        model="tencent/deepseek-v4-pro",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://tokenhub.tencentcloudmaas.com/v1/chat/completions"


def test_get_complete_url_does_not_append_to_full_url():
    config = TencentChatConfig()

    url = config.get_complete_url(
        api_base="https://tokenhub.tencentcloudmaas.com/v1/chat/completions",
        api_key=None,
        model="tencent/deepseek-v4-pro",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://tokenhub.tencentcloudmaas.com/v1/chat/completions"
