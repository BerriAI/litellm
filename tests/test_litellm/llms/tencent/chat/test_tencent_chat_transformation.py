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

    assert "thinking" not in result
    assert result["extra_body"]["thinking"] == {"type": "enabled", "budget_tokens": 1024}


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

    assert "thinking" not in result
    assert result["extra_body"]["thinking"] == {"type": "enabled"}


def test_map_openai_params_none_reasoning_effort_disables_thinking():
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
    assert result["extra_body"]["thinking"] == {"type": "disabled"}
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

    assert "thinking" not in result
    assert result["extra_body"]["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_map_openai_params_extracts_thinking_and_effort_from_optional_params():
    config = TencentChatConfig()
    result = config.map_openai_params(
        non_default_params={},
        optional_params={"thinking": {"type": "enabled"}, "reasoning_effort": "medium"},
        model="tencent/deepseek-v4-pro",
        drop_params=False,
    )

    assert "thinking" not in result
    assert result["extra_body"]["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in result


def test_map_openai_params_merges_into_existing_extra_body():
    config = TencentChatConfig()
    result = config.map_openai_params(
        non_default_params={},
        optional_params={
            "thinking": {"type": "enabled"},
            "extra_body": {"custom_flag": True},
        },
        model="tencent/deepseek-v4-pro",
        drop_params=False,
    )

    assert result["extra_body"] == {"custom_flag": True, "thinking": {"type": "enabled"}}


def test_transform_request_never_passes_thinking_as_top_level_kwarg():
    """
    Regression test: tencent routes through the OpenAI SDK's
    chat.completions.create(**data), which raises TypeError on unknown kwargs.
    `thinking` must be nested inside extra_body, never top-level.
    """
    config = TencentChatConfig()
    optional_params = config.map_openai_params(
        non_default_params={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        optional_params={},
        model="tencent/deepseek-v4-pro",
        drop_params=False,
    )

    data = config.transform_request(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "thinking" not in data
    assert data["extra_body"]["thinking"] == {"type": "enabled", "budget_tokens": 1024}


class TestMinimaxThinkingCoercion:
    """
    MiniMax models on TokenHub only accept thinking.type "adaptive"/"disabled" —
    "enabled" returns a 400. Ref: https://www.tencentcloud.com/document/product/1300/82345
    """

    def test_reasoning_effort_maps_to_adaptive_for_minimax(self):
        config = TencentChatConfig()
        with patch(
            "litellm.llms.tencent.chat.transformation.supports_reasoning",
            return_value=True,
        ):
            result = config.map_openai_params(
                non_default_params={"reasoning_effort": "medium"},
                optional_params={},
                model="tencent/minimax-m3",
                drop_params=False,
            )

        assert result["extra_body"]["thinking"] == {"type": "adaptive"}

    def test_explicit_enabled_thinking_coerced_to_adaptive_for_minimax(self):
        config = TencentChatConfig()
        with patch(
            "litellm.llms.tencent.chat.transformation.supports_reasoning",
            return_value=True,
        ):
            result = config.map_openai_params(
                non_default_params={"thinking": {"type": "enabled", "budget_tokens": 4096}},
                optional_params={},
                model="minimax-m3",
                drop_params=False,
            )

        assert result["extra_body"]["thinking"] == {"type": "adaptive", "budget_tokens": 4096}

    def test_disabled_thinking_kept_for_minimax(self):
        config = TencentChatConfig()
        with patch(
            "litellm.llms.tencent.chat.transformation.supports_reasoning",
            return_value=True,
        ):
            result = config.map_openai_params(
                non_default_params={"thinking": {"type": "disabled"}},
                optional_params={},
                model="tencent/minimax-m3",
                drop_params=False,
            )

        assert result["extra_body"]["thinking"] == {"type": "disabled"}

    def test_none_reasoning_effort_disables_thinking_for_minimax(self):
        config = TencentChatConfig()
        with patch(
            "litellm.llms.tencent.chat.transformation.supports_reasoning",
            return_value=True,
        ):
            result = config.map_openai_params(
                non_default_params={"reasoning_effort": "none"},
                optional_params={},
                model="tencent/minimax-m3",
                drop_params=False,
            )

        assert result["extra_body"]["thinking"] == {"type": "disabled"}

    def test_non_minimax_model_keeps_enabled(self):
        config = TencentChatConfig()
        with patch(
            "litellm.llms.tencent.chat.transformation.supports_reasoning",
            return_value=True,
        ):
            result = config.map_openai_params(
                non_default_params={"reasoning_effort": "high"},
                optional_params={},
                model="tencent/kimi-k3",
                drop_params=False,
            )

        assert result["extra_body"]["thinking"] == {"type": "enabled"}


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


def test_provider_info_falls_back_to_default_base():
    config = TencentChatConfig()

    with patch("litellm.llms.tencent.chat.transformation.get_secret_str", return_value=None):
        api_base, api_key = config._get_openai_compatible_provider_info(api_base=None, api_key="sk-arg")

    assert api_base == "https://tokenhub-intl.tencentcloudmaas.com/v1"
    assert api_key == "sk-arg"


def test_provider_info_reads_env_secrets():
    config = TencentChatConfig()

    secrets = {"TENCENT_API_BASE": "https://env.tencent/v1", "TENCENT_API_KEY": "sk-env"}
    with patch(
        "litellm.llms.tencent.chat.transformation.get_secret_str",
        side_effect=lambda key: secrets.get(key),
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(api_base=None, api_key=None)

    assert api_base == "https://env.tencent/v1"
    assert api_key == "sk-env"
