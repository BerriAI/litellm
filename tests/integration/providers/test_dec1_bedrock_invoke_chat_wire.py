"""`/v1/chat/completions` on Bedrock InvokeModel deployments: per-provider request bodies, answers and headers.

InvokeModel (`POST /model/<id>/invoke`) has no common schema: each model family on Bedrock takes its own
body. The proxy picks the family from the model id (`get_bedrock_invoke_provider`,
`litellm/llms/bedrock/base_aws_llm.py:561`) and builds the body in
`litellm/llms/bedrock/chat/invoke_transformations/base_invoke_transformation.py:208`:
Mistral and Llama 2 take a rendered `prompt` string, Titan text takes `inputText` with
`textGenerationConfig`, Cohere Command R takes `message` plus `chat_history`, legacy Cohere Command takes
`prompt`, Anthropic Claude takes the Anthropic Messages body with `anthropic_version: bedrock-2023-05-31`,
and Nova takes a Converse-like body. Usage comes from the `x-amzn-bedrock-input-token-count` and
`x-amzn-bedrock-output-token-count` response headers when the body has none. Guardrails and the Nova
service tier travel as `X-Amzn-Bedrock-*` headers, not in the body.

A model goes to InvokeModel when it is not in the Converse list and its id names an Invoke provider, or
when the deployment uses the explicit `bedrock/invoke/` route. A Rust port of these transforms must
produce the same bytes, since Bedrock rejects any body field it does not know.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_bedrock_support import (
    CLAUDE_SONNET_45,
    COHERE_COMMAND_R,
    COHERE_COMMAND_TEXT,
    LLAMA2_13B,
    MISTRAL_7B,
    NOVA_PRO,
    TITAN_TEXT,
    TOKEN,
    WEATHER_TOOL,
    bedrock,
    body_of,
    converse_json,
    error_message,
    json_reply,
    marker,
    path_of,
)
from pydantic import JsonValue

USAGE_HEADERS: Final = {"x-amzn-bedrock-input-token-count": "13", "x-amzn-bedrock-output-token-count": "5"}


def _mistral_reply(_: Request) -> Reply:
    return json_reply({"outputs": [{"text": "bonjour", "stop_reason": "length"}]}, headers=USAGE_HEADERS)


def _claude_message(content: list[JsonValue], stop_reason: str = "end_turn") -> dict[str, JsonValue]:
    return {
        "id": "msg_bdrk_dec1",
        "type": "message",
        "role": "assistant",
        "model": CLAUDE_SONNET_45,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 6},
    }


def test_a_bare_mistral_7b_id_posts_to_invoke_with_the_mistral_instruct_prompt(gateway: Gateway) -> None:
    """`bedrock/mistral.mistral-7b-instruct-v0:2` is not a Converse model, so a chat call goes to InvokeModel.

    Bedrock must receive one `POST /model/mistral.mistral-7b-instruct-v0%3A2/invoke` with a bearer key and the
    body `{"prompt": "<s>[INST] <prompt> [/INST]\\n", "max_tokens": 50, "temperature": 0.3, "top_p": 0.9,
    "stop": ["END"]}` (`mistral_instruct_pt`, `factory.py:386`; `AmazonMistralConfig.map_openai_params`).
    """
    prompt: Final = marker()
    with bedrock(gateway, _mistral_reply, model=f"bedrock/{MISTRAL_7B}") as rig:
        response: Final = rig.chat(prompt, max_tokens=50, temperature=0.3, top_p=0.9, stop=["END"])
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.target == "/model/mistral.mistral-7b-instruct-v0%3A2/invoke", sent.target
        assert sent.headers["authorization"] == f"Bearer {TOKEN}"
        assert body_of(sent) == {
            "prompt": f"<s>[INST] {prompt} [/INST]\n",
            "max_tokens": 50,
            "temperature": 0.3,
            "top_p": 0.9,
            "stop": ["END"],
        }


def test_a_mistral_system_and_assistant_history_render_into_one_instruct_prompt(gateway: Gateway) -> None:
    """A system message, a user turn, an assistant turn and a second user turn go to Mistral on InvokeModel.

    `custom_prompt` concatenates each role's pre and post strings, so the `prompt` must be exactly
    `<s>[INST] \\nBe brief. [/INST]\\n[INST] one [/INST]\\n two</s> [INST] <second> [/INST]\\n`.
    """
    second: Final = marker()
    with bedrock(gateway, _mistral_reply, model=f"bedrock/{MISTRAL_7B}") as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                    {"role": "user", "content": second},
                ]
            }
        )
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["prompt"] == (
            f"<s>[INST] \nBe brief. [/INST]\n[INST] one [/INST]\n two</s> [INST] {second} [/INST]\n"
        )


def test_a_mistral_invoke_answer_reports_text_finish_reason_and_header_usage(gateway: Gateway) -> None:
    """Bedrock answers `{"outputs": [{"text": "bonjour", "stop_reason": "length"}]}` with token count headers 13 and 5.

    The caller must get `content: "bonjour"`, `finish_reason: "length"`, and `usage` 13 prompt and 5
    completion tokens read from `x-amzn-bedrock-input-token-count` and `x-amzn-bedrock-output-token-count`
    (`base_invoke_transformation.py:428`), not a local token estimate.
    """
    with bedrock(gateway, _mistral_reply, model=f"bedrock/{MISTRAL_7B}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["choices"][0]["message"]["content"] == "bonjour"
        assert body["choices"][0]["finish_reason"] == "length"
        assert (body["usage"]["prompt_tokens"], body["usage"]["completion_tokens"]) == (13, 5)


def test_a_mistral_invoke_call_is_priced_from_the_cost_map_row_of_the_model(gateway: Gateway) -> None:
    """With no deployment prices, the `mistral.mistral-7b-instruct-v0:2` cost map row (1.5e-07 in, 2e-07 out) applies.

    For 13 prompt and 5 completion tokens the `x-litellm-response-cost` header must be
    13 * 1.5e-07 + 5 * 2e-07 = 2.95e-06.
    """
    with bedrock(gateway, _mistral_reply, model=f"bedrock/{MISTRAL_7B}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(13 * 1.5e-07 + 5 * 2e-07)


def test_llama_2_chat_gets_the_llama_2_prompt_and_max_gen_len(gateway: Gateway) -> None:
    """`bedrock/meta.llama2-13b-chat-v1` goes to InvokeModel with a rendered Llama 2 chat prompt.

    Bedrock must receive `{"prompt": "<s>[INST] <prompt> [/INST]\\n", "max_gen_len": 40, "temperature": 0.2}`:
    `max_tokens` is renamed to Llama's `max_gen_len` (`amazon_llama_transformation.py:74`), and the
    `generation` field of the answer becomes the caller's content.
    """
    prompt: Final = marker()

    def respond(_: Request) -> Reply:
        return json_reply(
            {
                "generation": "hello from llama",
                "prompt_token_count": 9,
                "generation_token_count": 4,
                "stop_reason": "stop",
            }
        )

    with bedrock(gateway, respond, model=f"bedrock/{LLAMA2_13B}") as rig:
        response: Final = rig.chat(prompt, max_tokens=40, temperature=0.2)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{LLAMA2_13B}/invoke"
        assert body_of(sent) == {"prompt": f"<s>[INST] {prompt} [/INST]\n", "max_gen_len": 40, "temperature": 0.2}
        assert response.json()["choices"][0]["message"]["content"] == "hello from llama"


def test_titan_text_gets_input_text_and_text_generation_config(gateway: Gateway) -> None:
    """`bedrock/amazon.titan-text-express-v1` goes to InvokeModel with Titan's own body.

    Bedrock must receive `{"inputText": "\\n\\nUser: <prompt>\\n\\nBot: ", "textGenerationConfig":
    {"maxTokenCount": 30, "temperature": 0.1}}` (`amazon_titan_pt`, `factory.py:3836`), and
    `results[0].outputText` must become the caller's content.
    """
    prompt: Final = marker()

    def respond(_: Request) -> Reply:
        return json_reply(
            {
                "inputTextTokenCount": 8,
                "results": [{"tokenCount": 3, "outputText": "titan says hi", "completionReason": "FINISH"}],
            }
        )

    with bedrock(gateway, respond, model=f"bedrock/{TITAN_TEXT}") as rig:
        response: Final = rig.chat(prompt, max_tokens=30, temperature=0.1)
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post()) == {
            "inputText": f"\n\nUser: {prompt}\n\nBot: ",
            "textGenerationConfig": {"maxTokenCount": 30, "temperature": 0.1},
        }
        assert response.json()["choices"][0]["message"]["content"] == "titan says hi"


def test_cohere_command_r_gets_message_and_an_empty_chat_history(gateway: Gateway) -> None:
    """`bedrock/cohere.command-r-v1:0` goes to InvokeModel in the Cohere chat shape.

    Every message with content is joined into `message` (`cohere_message_pt`, `factory.py:3821`) and the
    tool results list becomes `chat_history`, so a single user turn must be sent as
    `{"message": <prompt>, "chat_history": [], "max_tokens": 20}`; Cohere's `text` answer becomes the content.
    """
    prompt: Final = marker()

    def respond(_: Request) -> Reply:
        return json_reply({"text": "command r answer", "finish_reason": "COMPLETE"})

    with bedrock(gateway, respond, model=f"bedrock/{COHERE_COMMAND_R}") as rig:
        response: Final = rig.chat(prompt, max_tokens=20)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{COHERE_COMMAND_R}/invoke"
        body: Final = body_of(sent)
        assert body["message"] == prompt
        assert body["chat_history"] == []
        assert body["max_tokens"] == 20
        assert response.json()["choices"][0]["message"]["content"] == "command r answer"


def test_legacy_cohere_command_gets_a_prompt_and_maps_generations_back(gateway: Gateway) -> None:
    """`bedrock/cohere.command-text-v14` is the pre Command R family: the body is `{"prompt": <prompt>, "max_tokens": 20}`.

    Its answer `generations[0]` (`text`, `finish_reason: COMPLETE`) must reach the caller as the content
    and `finish_reason: stop` (`core_helpers.py:443`).
    """
    prompt: Final = marker()

    def respond(_: Request) -> Reply:
        return json_reply({"generations": [{"text": "legacy answer", "finish_reason": "COMPLETE"}]})

    with bedrock(gateway, respond, model=f"bedrock/{COHERE_COMMAND_TEXT}") as rig:
        response: Final = rig.chat(prompt, max_tokens=20)
        assert response.status_code == 200, response.text
        body: Final = body_of(rig.one_post())
        assert body["prompt"] == prompt
        assert body["max_tokens"] == 20
        choice: Final = response.json()["choices"][0]
        assert choice["message"]["content"] == "legacy answer"
        assert choice["finish_reason"] == "stop"


def test_claude_on_the_invoke_route_gets_the_bedrock_anthropic_body(gateway: Gateway) -> None:
    """`bedrock/invoke/anthropic.claude-sonnet-4-5-20250929-v1:0` sends a chat call to InvokeModel, not Converse.

    Bedrock must receive `POST /model/<id>/invoke` with an Anthropic Messages body that carries
    `anthropic_version: "bedrock-2023-05-31"`, `max_tokens: 64`, the system text in `system`, and no `model`
    or `stream` key (`_build_bedrock_anthropic_request_base`, `anthropic_claude_transformation.py:335`).
    """

    def respond(_: Request) -> Reply:
        return json_reply(_claude_message([{"type": "text", "text": "invoke pong"}]))

    with bedrock(gateway, respond, model=f"bedrock/invoke/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat_body(
            {
                "max_tokens": 64,
                "messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": marker()}],
            }
        )
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{CLAUDE_SONNET_45}/invoke"
        body: Final = body_of(sent)
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert body["max_tokens"] == 64
        assert "Be terse." in json.dumps(body["system"])
        assert "model" not in body and "stream" not in body
        assert response.json()["choices"][0]["message"]["content"] == "invoke pong"
        assert (response.json()["usage"]["prompt_tokens"], response.json()["usage"]["completion_tokens"]) == (12, 6)


def test_claude_invoke_tools_are_sent_in_anthropic_form_and_tool_use_comes_back_as_tool_calls(gateway: Gateway) -> None:
    """An OpenAI function tool on the Claude Invoke route is sent as an Anthropic tool (`name`, `input_schema`).

    When Bedrock answers a `tool_use` block (`toolu_dec1`, `get_weather`, `{"city": "Oslo"}`) with
    `stop_reason: tool_use`, the caller must get one OpenAI tool call with that id, name and JSON
    arguments, and `finish_reason: tool_calls`.
    """

    def respond(_: Request) -> Reply:
        return json_reply(
            _claude_message(
                [{"type": "tool_use", "id": "toolu_dec1", "name": "get_weather", "input": {"city": "Oslo"}}],
                stop_reason="tool_use",
            )
        )

    with bedrock(gateway, respond, model=f"bedrock/invoke/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL], max_tokens=64)
        assert response.status_code == 200, response.text
        tool: Final = body_of(rig.one_post())["tools"][0]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert tool["name"] == "get_weather"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert tool["input_schema"]["properties"] == {"city": {"type": "string"}}  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        choice: Final = response.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        call: Final = choice["message"]["tool_calls"][0]
        assert (call["id"], call["function"]["name"]) == ("toolu_dec1", "get_weather")
        assert json.loads(call["function"]["arguments"]) == {"city": "Oslo"}


def test_claude_invoke_thinking_blocks_come_back_as_reasoning_content(gateway: Gateway) -> None:
    """Bedrock's Anthropic answer on the Invoke route starts with a `thinking` block (`thinking`, `signature`).

    The caller must get `reasoning_content` equal to the thinking text and a `thinking_blocks` entry
    carrying the same `signature`, exactly as on first-party Anthropic.
    """

    def respond(_: Request) -> Reply:
        return json_reply(
            _claude_message(
                [
                    {"type": "thinking", "thinking": "count the letters", "signature": "sig-invoke"},
                    {"type": "text", "text": "three"},
                ]
            )
        )

    with bedrock(gateway, respond, model=f"bedrock/invoke/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat(max_tokens=64)
        assert response.status_code == 200, response.text
        message: Final = response.json()["choices"][0]["message"]
        assert message["content"] == "three"
        assert message["reasoning_content"] == "count the letters"
        assert message["thinking_blocks"][0]["signature"] == "sig-invoke"


def test_nova_on_the_invoke_route_gets_a_converse_like_body_without_request_metadata(gateway: Gateway) -> None:
    """`bedrock/invoke/amazon.nova-pro-v1:0` reuses the Converse transform and then filters to Nova's Invoke fields.

    Bedrock must receive `messages: [{"role": "user", "content": [{"text": <prompt>}]}]` at
    `/model/amazon.nova-pro-v1:0/invoke`, and a caller `requestMetadata` must not appear in the body,
    because Nova InvokeModel rejects it (`amazon_nova_transformation.py:120`).
    """
    prompt: Final = marker()

    def respond(_: Request) -> Reply:
        return json_reply(converse_json(content=[{"text": "nova invoke"}]))

    with bedrock(gateway, respond, model=f"bedrock/invoke/{NOVA_PRO}") as rig:
        response: Final = rig.chat(prompt, requestMetadata={"team": "dec1"})
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{NOVA_PRO}/invoke"
        body: Final = body_of(sent)
        assert body["messages"] == [{"role": "user", "content": [{"text": prompt}]}]
        assert "requestMetadata" not in body
        assert response.json()["choices"][0]["message"]["content"] == "nova invoke"


def test_nova_invoke_service_tier_travels_as_the_service_tier_header(gateway: Gateway) -> None:
    """On Nova InvokeModel the tier is a header: `service_tier: "priority"` must arrive as `X-Amzn-Bedrock-Service-Tier: priority`.

    (`_bedrock_invoke_service_tier_headers`, `base_invoke_transformation.py:87`). UNVERIFIED that the body
    no longer carries `serviceTier` after `_filter_allowed_fields`.
    """

    def respond(_: Request) -> Reply:
        return json_reply(converse_json())

    with bedrock(gateway, respond, model=f"bedrock/invoke/{NOVA_PRO}") as rig:
        response: Final = rig.chat(service_tier="priority")
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.headers["x-amzn-bedrock-service-tier"] == "priority"
        assert "serviceTier" not in body_of(sent)


def test_invoke_guardrail_config_becomes_bedrock_guardrail_headers_not_body_fields(gateway: Gateway) -> None:
    """A Claude Invoke deployment with `guardrailConfig` `{"guardrailIdentifier": "gr-inv", "guardrailVersion": "3", "trace": "enabled"}`.

    InvokeModel takes guardrails as headers, so Bedrock must see `X-Amzn-Bedrock-GuardrailIdentifier: gr-inv`,
    `X-Amzn-Bedrock-GuardrailVersion: 3` and `X-Amzn-Bedrock-Trace: ENABLED` (upper-cased,
    `bedrock_invoke_guardrail_headers`, `base_invoke_transformation.py:95`) and no `guardrailConfig` in the body.
    """

    def respond(_: Request) -> Reply:
        return json_reply(_claude_message([{"type": "text", "text": "ok"}]))

    guardrail: Final[dict[str, JsonValue]] = {
        "guardrailIdentifier": "gr-inv",
        "guardrailVersion": "3",
        "trace": "enabled",
    }
    with bedrock(gateway, respond, model=f"bedrock/invoke/{CLAUDE_SONNET_45}", guardrailConfig=guardrail) as rig:
        response: Final = rig.chat(max_tokens=32)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.headers["x-amzn-bedrock-guardrailidentifier"] == "gr-inv"
        assert sent.headers["x-amzn-bedrock-guardrailversion"] == "3"
        assert sent.headers["x-amzn-bedrock-trace"] == "ENABLED"
        assert "guardrailConfig" not in body_of(sent)


def test_invoke_guardrail_config_without_an_identifier_is_a_400_before_bedrock(gateway: Gateway) -> None:
    """A Claude Invoke deployment whose `guardrailConfig` is `{"guardrailVersion": "2"}` cannot build the headers.

    The caller must get HTTP 400 whose message contains `is missing 'guardrailIdentifier'`, and no request
    reaches Bedrock.
    """

    def respond(_: Request) -> Reply:
        return json_reply(_claude_message([{"type": "text", "text": "never"}]))

    with bedrock(
        gateway, respond, model=f"bedrock/invoke/{CLAUDE_SONNET_45}", guardrailConfig={"guardrailVersion": "2"}
    ) as rig:
        response: Final = rig.chat(max_tokens=32)
        assert response.status_code == 400, response.text
        assert "is missing 'guardrailIdentifier'" in error_message(response), response.text
        assert rig.posts() == ()


def test_an_invoke_answer_bedrock_cannot_parse_is_a_422_style_bad_request(gateway: Gateway) -> None:
    """Bedrock answers HTTP 200 with a Mistral body missing `outputs` and `choices` (`{"unexpected": true}`).

    `transform_response` raises `BedrockError(status_code=422, "Error processing=...")`
    (`base_invoke_transformation.py:402`), which `_map_bedrock_exception` maps to `BadRequestError`, so the
    caller must get HTTP 400 naming the unparsed body, never an empty 200.
    UNVERIFIED: the outer status; 422 maps to BadRequestError in `exception_mapping_utils.py:1127`.
    """

    def respond(_: Request) -> Reply:
        return json_reply({"unexpected": True})

    with bedrock(gateway, respond, model=f"bedrock/{MISTRAL_7B}") as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 400, response.text
        assert "unexpected" in error_message(response), response.text
