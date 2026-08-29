"""
Apodex Responses API transformation.

The core models expose a stateless subset of /v1/responses while the Deep
Research tiers keep server-side state, so the parameter contract is keyed off
the model rather than applied provider-wide.
"""

import gzip
from types import SimpleNamespace

import httpx
import pytest

import litellm
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

CORE_MODEL = "apodex/apodex-1.1"
CORE_MINI_MODEL = "apodex/apodex-1.1-mini"
DEEP_RESEARCH_MODEL = "apodex/apodex-1-1-deep-research"


@pytest.fixture(autouse=True)
def _apodex_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APODEX_API_KEY", "sk-apodex-test")
    monkeypatch.delenv("APODEX_API_BASE", raising=False)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setattr(litellm, "drop_params", False)
    yield


_SENTINEL = "apodex-request-captured"


def _capture(**kwargs) -> dict:
    """Run litellm.responses() and return the request it would have sent.

    Validation errors raised before the request is built propagate to the caller.
    """
    captured: dict = {}

    class CapturingHandler(HTTPHandler):
        def post(self, *args, **post_kwargs):
            captured.update(url=post_kwargs.get("url"), body=post_kwargs.get("json"))
            raise RuntimeError(_SENTINEL)

    try:
        litellm.responses(client=CapturingHandler(), **kwargs)
    except Exception as exc:
        if _SENTINEL not in str(exc):
            raise
    assert captured, "no request was sent"
    return captured


def _responses_config(model: str):
    return ProviderConfigManager.get_provider_responses_api_config(model=model, provider=LlmProviders.APODEX)


class TestConfigSelection:
    def test_python_config_is_used_for_every_apodex_model(self):
        for model in ("apodex-1.1", "apodex-1.1-mini", "apodex-1-1-deep-research"):
            config = _responses_config(model)
            assert type(config).__name__ == "ApodexResponsesConfig"

    def test_auth_uses_the_apodex_key(self):
        config = _responses_config("apodex-1.1")
        assert config.validate_environment(headers={}, model="apodex-1.1", litellm_params=None) == {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-apodex-test",
        }

    def test_auth_does_not_fall_back_to_an_openai_key(self, monkeypatch: pytest.MonkeyPatch):
        """The inherited OpenAI config would forward OPENAI_API_KEY to Apodex."""
        monkeypatch.delenv("APODEX_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-leak")

        config = _responses_config("apodex-1.1")
        assert config.validate_environment(headers={}, model="apodex-1.1", litellm_params=None) == {}

    def test_request_targets_the_apodex_responses_url(self):
        assert _capture(model=CORE_MODEL, input="hi")["url"] == "https://api.apodex.ai/v1/responses"

    def test_polling_without_model_resolution_targets_apodex(self):
        config = _responses_config("apodex-1-1-deep-research")
        assert config.get_complete_url(api_base=None, litellm_params={}) == "https://api.apodex.ai/v1/responses"

    def test_polling_honours_an_explicit_api_base(self):
        config = _responses_config("apodex-1-1-deep-research")
        assert (
            config.get_complete_url(api_base="https://gateway.apodex.test/v1/", litellm_params={})
            == "https://gateway.apodex.test/v1/responses"
        )

    def test_request_honours_an_api_base_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APODEX_API_BASE", "https://env.apodex.test/v1")
        assert _capture(model=CORE_MODEL, input="hi")["url"] == "https://env.apodex.test/v1/responses"


class TestStreamDefault:
    def test_non_streaming_pins_stream_false(self):
        captured = _capture(model=DEEP_RESEARCH_MODEL, input="hi")
        assert captured["url"] == "https://api.apodex.ai/v1/responses"
        assert captured["body"]["stream"] is False

    @pytest.mark.asyncio
    async def test_streaming_sends_stream_true(self):
        captured: dict = {}

        class CapturingHandler(AsyncHTTPHandler):
            async def post(self, *args, **kwargs):
                captured.update(body=kwargs.get("json"))
                raise RuntimeError("captured")

        with pytest.raises(Exception, match="captured"):
            await litellm.aresponses(model=DEEP_RESEARCH_MODEL, input="hi", stream=True, client=CapturingHandler())

        assert captured["body"]["stream"] is True


class TestCoreModelStatelessSubset:
    """Apodex core models reject anything that would persist state on their side."""

    @pytest.mark.parametrize("model", [CORE_MODEL, CORE_MINI_MODEL])
    def test_store_is_pinned_false(self, model: str):
        captured = _capture(model=model, input="hi")
        assert captured["body"]["store"] is False

    def test_store_true_raises(self):
        with pytest.raises(litellm.UnsupportedParamsError, match="store=True"):
            _capture(model=CORE_MODEL, input="hi", store=True)

    def test_background_raises(self):
        with pytest.raises(litellm.UnsupportedParamsError, match="background"):
            _capture(model=CORE_MODEL, input="hi", background=True)

    def test_previous_response_id_raises(self):
        with pytest.raises(litellm.UnsupportedParamsError, match="previous_response_id"):
            _capture(model=CORE_MODEL, input="hi", previous_response_id="resp_1")

    def test_drop_params_strips_all_three(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(litellm, "drop_params", True)
        captured = _capture(
            model=CORE_MODEL,
            input="hi",
            store=True,
            background=True,
            previous_response_id="resp_1",
        )

        assert captured["body"]["store"] is False
        assert "background" not in captured["body"]
        assert "previous_response_id" not in captured["body"]

    def test_stateful_params_are_not_advertised(self):
        supported = _responses_config("apodex-1.1").get_supported_openai_params("apodex-1.1")
        assert "background" not in supported
        assert "previous_response_id" not in supported
        assert "max_output_tokens" in supported

    def test_max_output_tokens_still_passes_through(self):
        captured = _capture(model=CORE_MODEL, input="hi", max_output_tokens=512)
        assert captured["body"]["max_output_tokens"] == 512


class TestDeepResearchKeepsState:
    """The agent tiers survive client disconnects, so none of this may be stripped."""

    def test_background_passes_through(self):
        captured = _capture(model=DEEP_RESEARCH_MODEL, input="hi", background=True)
        assert captured["body"]["background"] is True

    def test_store_and_previous_response_id_pass_through(self):
        captured = _capture(model=DEEP_RESEARCH_MODEL, input="hi", store=True, previous_response_id="resp_1")
        assert captured["body"]["store"] is True
        assert captured["body"]["previous_response_id"] == "resp_1"

    def test_store_is_not_pinned_when_unset(self):
        captured = _capture(model=DEEP_RESEARCH_MODEL, input="hi")
        assert "store" not in captured["body"]

    def test_stateful_params_are_advertised(self):
        supported = _responses_config("apodex-1-1-deep-research").get_supported_openai_params(
            "apodex-1-1-deep-research"
        )
        assert "background" in supported
        assert "previous_response_id" in supported

    def test_minimal_cancel_response_is_normalized(self):
        config = _responses_config("apodex-1-1-deep-research")
        response = config.transform_cancel_response_api_response(
            raw_response=httpx.Response(
                200,
                json={"id": "resp_1", "object": "response", "status": "cancelled"},
            ),
            logging_obj=None,
        )

        assert response.id == "resp_1"
        assert response.status == "cancelled"
        assert response.output == []
        assert response.created_at > 0

    def test_cancel_response_survives_a_compressed_upstream_response(self):
        """The body is rebuilt, so the original framing headers must not follow it.

        httpx decodes on read, so carrying Content-Encoding over from the compressed
        upstream response makes it try to gunzip the plain JSON replacement.
        """
        body = b'{"id": "resp_1", "object": "response", "status": "cancelled"}'
        # As it arrives off the wire: httpx decodes the body but leaves the header in place
        upstream = httpx.Response(
            200,
            headers={
                "content-encoding": "gzip",
                "x-ratelimit-remaining-requests": "42",
            },
            content=gzip.compress(body),
        )
        assert upstream.content == body

        config = _responses_config("apodex-1-1-deep-research")
        response = config.transform_cancel_response_api_response(
            raw_response=upstream,
            logging_obj=None,
        )

        assert response.id == "resp_1"
        assert response.status == "cancelled"
        assert response._hidden_params["headers"]["x-ratelimit-remaining-requests"] == "42"

    def test_output_text_delta_is_normalized(self):
        """A Deep Research stream never emits response.output_text.delta of its own.

        Payload shape captured from a live stream; the extra swarm.data keys ride
        along untouched and must not affect the mapping.
        """
        config = _responses_config("apodex-1-1-deep-research")
        event = config.transform_streaming_response(
            model="apodex-1-1-deep-research",
            parsed_chunk={
                "type": "response.swarm.llm_delta",
                "created_at": 1786873219.3543231,
                "response_id": "w_c4b77c96",
                "sequence_number": 12,
                "swarm": {
                    "agent_id": "reporter",
                    "data": {
                        "channel": "output_text",
                        "delta": "Hello there, friend!",
                        "delta_index": 0,
                        "call_id": "llm_fce8e965",
                        "turn": 1,
                    },
                },
            },
            logging_obj=None,
        )

        assert event.type == "response.output_text.delta"
        assert event.item_id == "msg_w_c4b77c96"
        assert event.delta == "Hello there, friend!"
        assert event.sequence_number == 12
        assert event.content_index == 0

    def test_reasoning_delta_becomes_a_reasoning_summary_delta(self):
        """`response.reasoning_summary_text.delta` is what LiteLLM already translates
        into an Anthropic `thinking_delta`, which is the route Deep Research takes on
        /v1/messages. It also keeps a separate item id from the answer text."""
        config = _responses_config("apodex-1-1-deep-research")
        event = config.transform_streaming_response(
            model="apodex-1-1-deep-research",
            parsed_chunk={
                "type": "response.swarm.llm_delta",
                "response_id": "w_c4b77c96",
                "sequence_number": 7,
                "swarm": {
                    "agent_id": "stateful_react",
                    "data": {"channel": "reasoning", "delta": "The user wants", "delta_index": 0},
                },
            },
            logging_obj=None,
        )

        assert event.type == "response.reasoning_summary_text.delta"
        assert event.item_id == "rs_w_c4b77c96"
        assert event.delta == "The user wants"
        assert event.summary_index == 0
        assert not hasattr(event, "content_index")

    def test_reasoning_and_answer_form_valid_anthropic_blocks(self):
        config = _responses_config("apodex-1-1-deep-research")
        reasoning_event = config.transform_streaming_response(
            model="apodex-1-1-deep-research",
            parsed_chunk={
                "type": "response.swarm.llm_delta",
                "response_id": "w_c4b77c96",
                "sequence_number": 7,
                "swarm": {"data": {"channel": "reasoning", "delta": "The user wants"}},
            },
            logging_obj=None,
        )
        answer_event = config.transform_streaming_response(
            model="apodex-1-1-deep-research",
            parsed_chunk={
                "type": "response.swarm.llm_delta",
                "response_id": "w_c4b77c96",
                "sequence_number": 8,
                "swarm": {"data": {"channel": "output_text", "delta": "Hello there, friend!"}},
            },
            logging_obj=None,
        )
        wrapper = AnthropicResponsesStreamWrapper(responses_stream=None, model="apodex-1-1-deep-research")
        for event in (
            reasoning_event,
            answer_event,
            {"type": "response.completed", "response": SimpleNamespace(status="completed", output=[], usage=None)},
        ):
            wrapper._process_event(event)

        chunks = list(wrapper._chunk_queue)
        assert [(chunk["type"], chunk.get("index")) for chunk in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
            ("content_block_start", 1),
            ("content_block_delta", 1),
            ("content_block_stop", 1),
            ("message_delta", None),
            ("message_stop", None),
        ]
        assert chunks[0]["content_block"]["type"] == "thinking"
        assert chunks[1]["delta"] == {"type": "thinking_delta", "thinking": "The user wants"}
        assert chunks[3]["content_block"]["type"] == "text"
        assert chunks[4]["delta"] == {"type": "text_delta", "text": "Hello there, friend!"}

    @pytest.mark.parametrize(
        "channel",
        (None, "tool_output", []),
        ids=("no-channel", "unknown-channel", "non-string-channel"),
    )
    def test_intermediate_agent_deltas_are_not_claimed(self, channel):
        """The worker agent streams a draft answer on a channel-less delta.

        Live capture: those four deltas spell "Hello, friend! How are you?" while the
        reporter's `output_text` is the "Hello there, friend!" that lands in
        response.completed. Claiming them would splice the draft into the answer.
        """
        data = {"delta": "Hello,"} if channel is None else {"channel": channel, "delta": "Hello,"}
        config = _responses_config("apodex-1-1-deep-research")
        event = config.transform_streaming_response(
            model="apodex-1-1-deep-research",
            parsed_chunk={
                "type": "response.swarm.llm_delta",
                "response_id": "w_c4b77c96",
                "sequence_number": 7,
                "swarm": {"agent_id": "stateful_react", "data": data},
            },
            logging_obj=None,
        )

        assert event.type == "response.swarm.llm_delta"

    def test_non_string_delta_is_not_claimed(self):
        config = _responses_config("apodex-1-1-deep-research")
        assert (
            config._map_swarm_delta(
                {
                    "type": "response.swarm.llm_delta",
                    "swarm": {"data": {"channel": "output_text", "delta": None}},
                }
            )
            is None
        )

    @pytest.mark.parametrize(
        "event_type",
        (
            "response.swarm.run_started",
            "response.swarm.run_finished",
            "response.swarm.injection_window",
            "response.swarm.llm_attempt_started",
            "response.swarm.llm_attempt_finished",
        ),
    )
    def test_swarm_lifecycle_events_pass_through(self, event_type: str):
        """LiteLLM cannot drop a chunk from the stream, so these stay as GenericEvent
        rather than being silently swallowed; `run_finished` carries the final content."""
        config = _responses_config("apodex-1-1-deep-research")
        event = config.transform_streaming_response(
            model="apodex-1-1-deep-research",
            parsed_chunk={
                "type": event_type,
                "response_id": "w_c4b77c96",
                "sequence_number": 4,
                "swarm": {"agent_id": "reporter", "data": {"status": "success"}},
            },
            logging_obj=None,
        )

        assert event.type == event_type

    def test_documented_events_pass_through_untouched(self):
        config = _responses_config("apodex-1-1-deep-research")
        event = config.transform_streaming_response(
            model="apodex-1-1-deep-research",
            parsed_chunk={"type": "response.in_progress", "sequence_number": 2},
            logging_obj=None,
        )

        assert event.type == "response.in_progress"

    def test_non_json_cancel_body_raises_the_provider_error(self):
        """The gateway answers a timed-out cancel with an HTML 504, not the JSON envelope."""
        config = _responses_config("apodex-1-1-deep-research")
        with pytest.raises(Exception, match="gateway timeout"):
            config.transform_cancel_response_api_response(
                raw_response=httpx.Response(504, content=b"<html>gateway timeout</html>"),
                logging_obj=None,
            )
