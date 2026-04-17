from litellm.integrations.langfuse.langfuse_handler import LangFuseHandler


class TestLangFuseHandlerDynamicCredentials:
    def test_dynamic_credentials_none_returns_false(self):
        """_dynamic_langfuse_credentials_are_passed should return False when
        standard_callback_dynamic_params is None (env-var-only config)."""
        result = LangFuseHandler._dynamic_langfuse_credentials_are_passed(None)
        assert result is False

    def test_dynamic_credentials_empty_dict_returns_false(self):
        """_dynamic_langfuse_credentials_are_passed should return False when
        standard_callback_dynamic_params is an empty dict."""
        result = LangFuseHandler._dynamic_langfuse_credentials_are_passed({})
        assert result is False

    def test_dynamic_credentials_with_langfuse_host_returns_true(self):
        result = LangFuseHandler._dynamic_langfuse_credentials_are_passed(
            {"langfuse_host": "http://localhost:3000"}
        )
        assert result is True

    def test_dynamic_credentials_with_public_key_returns_true(self):
        result = LangFuseHandler._dynamic_langfuse_credentials_are_passed(
            {"langfuse_public_key": "pk-lf-test"}
        )
        assert result is True

    def test_dynamic_credentials_with_secret_key_returns_true(self):
        result = LangFuseHandler._dynamic_langfuse_credentials_are_passed(
            {"langfuse_secret_key": "sk-lf-test"}
        )
        assert result is True

    def test_dynamic_credentials_with_secret_returns_true(self):
        result = LangFuseHandler._dynamic_langfuse_credentials_are_passed(
            {"langfuse_secret": "sk-lf-test"}
        )
        assert result is True
