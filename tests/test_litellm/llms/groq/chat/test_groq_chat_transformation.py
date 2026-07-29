import logging
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.groq.chat.transformation import GroqChatConfig
from litellm.utils import get_optional_params

WEB_SEARCH_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
)

COMPOUND_MODELS = ("compound", "compound-mini", "groq/compound", "groq/compound-mini")


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


class TestGroqWebSearchOptions:
    @pytest.mark.parametrize("model", WEB_SEARCH_MODELS + COMPOUND_MODELS)
    def test_supported_on_search_capable_models(self, model: str):
        assert "web_search_options" in GroqChatConfig().get_supported_openai_params(model)

    def test_not_supported_on_other_models(self):
        assert "web_search_options" not in GroqChatConfig().get_supported_openai_params("llama-3.3-70b-versatile")

    @pytest.mark.parametrize("web_search_options", [{"search_context_size": "high"}, {}])
    def test_translates_to_browser_search_tool(self, web_search_options: dict):
        optional_params = get_optional_params(
            model="openai/gpt-oss-20b",
            custom_llm_provider="groq",
            web_search_options=web_search_options,
        )
        assert optional_params["tools"] == [{"type": "browser_search"}]
        assert "web_search_options" not in optional_params

    def test_no_duplicate_browser_search_tool(self):
        optional_params = get_optional_params(
            model="openai/gpt-oss-20b",
            custom_llm_provider="groq",
            web_search_options={"search_context_size": "high"},
            tools=[{"type": "browser_search"}],
        )
        assert optional_params["tools"] == [{"type": "browser_search"}]

    def test_caller_function_tools_preserved(self):
        function_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        optional_params = get_optional_params(
            model="openai/gpt-oss-20b",
            custom_llm_provider="groq",
            web_search_options={},
            tools=[function_tool],
        )
        assert optional_params["tools"] == [function_tool, {"type": "browser_search"}]

    def test_unsupported_model_drops_param_with_drop_params(self):
        optional_params = get_optional_params(
            model="llama-3.3-70b-versatile",
            custom_llm_provider="groq",
            web_search_options={"search_context_size": "high"},
            drop_params=True,
        )
        assert "web_search_options" not in optional_params
        assert "tools" not in optional_params

    def test_unsupported_model_raises_without_drop_params(self):
        with pytest.raises(litellm.UnsupportedParamsError):
            get_optional_params(
                model="llama-3.3-70b-versatile",
                custom_llm_provider="groq",
                web_search_options={"search_context_size": "high"},
                drop_params=False,
            )

    @pytest.mark.parametrize("model", COMPOUND_MODELS)
    def test_compound_injects_no_tool(self, model: str):
        optional_params = get_optional_params(
            model=model,
            custom_llm_provider="groq",
            web_search_options={"search_context_size": "high"},
        )
        assert "web_search_options" not in optional_params
        assert "tools" not in optional_params

    def test_ignored_fields_logged_as_info(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.INFO, logger="LiteLLM"):
            get_optional_params(
                model="openai/gpt-oss-20b",
                custom_llm_provider="groq",
                web_search_options={"search_context_size": "high", "user_location": {"type": "approximate"}},
            )
        ignored_fields_records = tuple(
            record
            for record in caplog.records
            if "search_context_size" in record.message and "user_location" in record.message
        )
        assert len(ignored_fields_records) == 1
        assert ignored_fields_records[0].levelno == logging.INFO
        assert "enabled" in ignored_fields_records[0].message

    def test_empty_options_log_nothing(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.INFO, logger="LiteLLM"):
            get_optional_params(
                model="openai/gpt-oss-20b",
                custom_llm_provider="groq",
                web_search_options={},
            )
        assert not [record for record in caplog.records if "web_search_options" in record.message]


def _searched_groq_response(executed_tools: list | None) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-oss-20b",
        "service_tier": "auto",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Top headline: example",
                    **({"executed_tools": executed_tools} if executed_tools is not None else {}),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }


EXECUTED_TOOLS_THREE_SEARCHES_TWO_OPENS = [
    {"name": "browser.search", "type": "browser_search"},
    {"name": "browser.open", "type": "function"},
    {"name": "browser.search", "type": "browser_search"},
    {"type": "browser_search"},
    {"name": "browser.open", "type": "browser_search"},
    {"name": "browser.find", "type": "browser.find"},
]

EXECUTED_TOOLS_OPENS_ONLY = [
    {"name": "browser.open", "type": "browser.open"},
    {"name": "browser.open", "type": "function"},
    {"name": "browser.find", "type": "browser.find"},
]


def _groq_completion_with_mocked_response(response_json: dict) -> litellm.ModelResponse:
    client = HTTPHandler()
    fake_response = httpx.Response(
        status_code=200,
        json=response_json,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    with patch.object(client, "post", return_value=fake_response):
        return litellm.completion(
            model="groq/openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "hi"}],
            web_search_options={"search_context_size": "high"},
            api_key="fake-key",
            client=client,
        )


class TestGroqWebSearchUsageSignal:
    @pytest.mark.parametrize(
        "executed_tools, expected_searches, expected_opens",
        [
            (EXECUTED_TOOLS_THREE_SEARCHES_TWO_OPENS, 3, 2),
            (EXECUTED_TOOLS_OPENS_ONLY, 0, 2),
        ],
    )
    def test_counts_actions_into_usage(self, executed_tools: list, expected_searches: int, expected_opens: int):
        response = _groq_completion_with_mocked_response(_searched_groq_response(executed_tools))
        assert response.usage.server_tool_use.web_search_requests == expected_searches
        assert response.usage.server_tool_use.browser_open_requests == expected_opens

    def test_no_signal_without_executed_tools(self):
        response = _groq_completion_with_mocked_response(_searched_groq_response(None))
        assert getattr(response.usage, "server_tool_use", None) is None

    @pytest.mark.usefixtures("local_model_cost_map")
    @pytest.mark.parametrize(
        "executed_tools, expected_cost",
        [
            (EXECUTED_TOOLS_THREE_SEARCHES_TWO_OPENS, 3 * 0.005 + 2 * 0.001),
            (EXECUTED_TOOLS_OPENS_ONLY, 2 * 0.001),
        ],
    )
    def test_response_billed_per_action(self, executed_tools: list, expected_cost: float):
        response = _groq_completion_with_mocked_response(_searched_groq_response(executed_tools))
        assert StandardBuiltInToolCostTracking.response_object_includes_web_search_call(
            response_object=response, usage=response.usage
        )
        cost = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
            model="groq/openai/gpt-oss-20b",
            response_object=response,
            usage=response.usage,
            custom_llm_provider="groq",
            standard_built_in_tools_params={"web_search_options": {"search_context_size": "high"}},
        )
        assert cost == pytest.approx(expected_cost)


class TestGroqWebSearchCost:
    @pytest.mark.usefixtures("local_model_cost_map")
    @pytest.mark.parametrize("model", WEB_SEARCH_MODELS)
    @pytest.mark.parametrize("search_context_size", ["low", "medium", "high"])
    def test_browser_search_priced_per_search(self, model: str, search_context_size: str):
        cost = StandardBuiltInToolCostTracking.get_cost_for_web_search(
            web_search_options={"search_context_size": search_context_size},
            model_info=litellm.get_model_info(model=model, custom_llm_provider="groq"),
        )
        assert cost == 0.005
