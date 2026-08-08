from litellm.llms.byteplus.chat.transformation import BytePlusChatConfig


class TestBytePlusChatConfig:
    def test_get_supported_openai_params(self):
        config = BytePlusChatConfig()
        params = config.get_supported_openai_params("byteplus/seed-2-0-lite-260228")
        assert "max_completion_tokens" in params
        assert "thinking" in params
        assert "stream" in params

    def test_map_openai_params_max_completion_tokens(self):
        config = BytePlusChatConfig()
        non_default = {"max_completion_tokens": 100}
        optional = {}
        res = config.map_openai_params(
            non_default_params=non_default,
            optional_params=optional,
            model="byteplus/seed-2-0-lite-260228",
            drop_params=False,
        )
        assert res.get("max_tokens") == 100

    def test_map_openai_params_thinking(self):
        config = BytePlusChatConfig()
        non_default = {"thinking": {"type": "disabled"}}
        optional = {}
        res = config.map_openai_params(
            non_default_params=non_default,
            optional_params=optional,
            model="byteplus/seed-2-0-lite-260228",
            drop_params=False,
        )
        assert res.get("extra_body", {}).get("thinking") == {"type": "disabled"}
