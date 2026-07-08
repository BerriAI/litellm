"""
Direct unit tests for Router._model_group_has_max_input_tokens() and
Router._get_messages_for_pre_call_check().

These are exercised end-to-end via test_pre_call_checks_optimization.py and
tests/test_litellm/router_utils/pre_call_checks/test_responses_api_context_window_check.py,
but the repo's router_code_coverage.py check only scans files whose *filename*
contains "router", so this file calls both helpers directly to satisfy it.
"""

from unittest.mock import patch

from litellm import Router


def _make_router(max_input_tokens=None):
    model_info = {"id": "test-1"}
    if max_input_tokens is not None:
        model_info["max_input_tokens"] = max_input_tokens
    return Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "gpt-5-mini", "api_key": "sk-test"},
                "model_info": model_info,
            },
        ],
        set_verbose=False,
        enable_pre_call_checks=True,
    )


class TestRouterModelGroupHasMaxInputTokens:
    def test_true_when_deployment_has_max_input_tokens(self):
        router = _make_router(max_input_tokens=100)
        assert router._model_group_has_max_input_tokens(model="gpt-5-mini") is True

    def test_false_when_no_deployment_resolves_max_input_tokens(self):
        router = _make_router()
        # max_input_tokens can also be resolved via litellm's model cost map
        # (get_router_model_info), not just a static model_info override -
        # mock it directly so this test isn't coupled to gpt-5-mini's cost-map entry.
        with patch.object(router, "get_router_model_info", return_value={}):
            assert router._model_group_has_max_input_tokens(model="gpt-5-mini") is False


class TestRouterGetMessagesForPreCallCheck:
    def test_returns_messages_when_present(self):
        router = _make_router(max_input_tokens=100)
        messages = [{"role": "user", "content": "hi"}]
        result = router._get_messages_for_pre_call_check(model="gpt-5-mini", kwargs={"messages": messages})
        assert result is messages

    def test_returns_none_when_not_a_responses_api_call(self):
        router = _make_router(max_input_tokens=100)
        result = router._get_messages_for_pre_call_check(
            model="gpt-5-mini", kwargs={"input": "hi", "_responses_api_pre_call_check": False}
        )
        assert result is None

    def test_returns_none_when_no_deployment_resolves_max_input_tokens(self):
        router = _make_router()
        with patch.object(router, "get_router_model_info", return_value={}):
            result = router._get_messages_for_pre_call_check(
                model="gpt-5-mini", kwargs={"input": "hi", "_responses_api_pre_call_check": True}
            )
        assert result is None

    def test_converts_input_to_messages_for_responses_api_call(self):
        router = _make_router(max_input_tokens=100)
        result = router._get_messages_for_pre_call_check(
            model="gpt-5-mini", kwargs={"input": "hello there", "_responses_api_pre_call_check": True}
        )
        assert result is not None
        assert any("hello there" in str(m.get("content", "")) for m in result)
