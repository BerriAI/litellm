"""Bedrock model families that render their own prompt or speak OpenAI: Llama 3/4, DeepSeek, Qwen, gpt-oss.

InvokeModel has no common schema. Llama 3 and Llama 4 instruct models take one `prompt` string rendered
with the Llama 3 chat template, which `prompt_factory` (`litellm/litellm_core_utils/prompt_templates/
factory.py:6363`) builds through `hf_chat_template("meta-llama/Meta-Llama-3-8B-Instruct", ...)`. That
helper would download `tokenizer_config.json` from Hugging Face, but the template and its `bos_token`
are baked into `litellm.known_tokenizer_config` (`litellm/constants.py:1607`), so the proxy renders it
offline; the same holds for DeepSeek R1 (`deepseek-r1/deepseek-r1-7b-instruct`, `constants.py:1615`).
These tests own the only peer the proxy can reach, so a regression that fetches a tokenizer at request
time fails them instead of silently depending on the network.

Imported models carry their family as a path prefix (`bedrock/qwen3/<arn>`, `bedrock/deepseek_r1/<arn>`,
`bedrock/openai/<arn>`): the prefix picks the config (`common_utils.py:2300`) and is stripped from the
URL, where the ARN must stay one encoded path segment. Bare `deepseek.v3`, `qwen.qwen3-*`,
`openai.gpt-oss-*` and Llama 3/4 ids are in the Converse list and must go to `/converse`, priced from
their own cost map rows. A Rust port of the transforms has to produce the same prompt bytes, since the
model reads the template literally.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_bedrock3_support import (
    DEEPSEEK_ASSISTANT,
    DEEPSEEK_BOS,
    DEEPSEEK_USER,
    DEEPSEEK_V3,
    GPT_OSS_120B,
    IMPORTED_ARN,
    LLAMA3_8B,
    LLAMA3_BOS,
    LLAMA4_MAVERICK,
    QWEN3_32B,
    quoted,
    response_cost,
    unique_text,
)
from integration.providers.dec1_bedrock_support import (
    bedrock,
    body_of,
    converse_reply,
    json_reply,
    path_of,
)


def _llama_reply(_: Request) -> Reply:
    return json_reply(
        {"generation": "llama says hi", "prompt_token_count": 9, "generation_token_count": 4, "stop_reason": "stop"}
    )


def _llama3_prompt(*turns: tuple[str, str]) -> str:
    rendered: Final = "".join(f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>" for role, text in turns)
    return f"{LLAMA3_BOS}{rendered}<|start_header_id|>assistant<|end_header_id|>\n\n"


def _openai_chat_reply(_: Request) -> Reply:
    return json_reply(
        {
            "id": "chatcmpl-imported-dec1",
            "object": "chat.completion",
            "created": 1763373600,
            "model": "imported",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "imported answer"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        }
    )


# ------------------------------------------------------------------ Llama 3 and Llama 4 on InvokeModel


def test_a_bare_llama_3_id_goes_to_converse_not_invoke(gateway: Gateway) -> None:
    """`bedrock/meta.llama3-8b-instruct-v1:0` is in `BEDROCK_CONVERSE_MODELS` (`litellm/constants.py:1551`).

    Bedrock must receive one `POST /model/meta.llama3-8b-instruct-v1%3A0/converse`, so no prompt template is
    rendered for a bare Llama 3 id.
    """
    with bedrock(gateway, lambda _: converse_reply(), model=f"bedrock/{LLAMA3_8B}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert rig.one_post().target == f"/model/{quoted(LLAMA3_8B)}/converse"


def test_llama_3_on_invoke_gets_the_llama_3_chat_template_prompt(gateway: Gateway) -> None:
    """`bedrock/invoke/meta.llama3-8b-instruct-v1:0` with one user turn.

    Bedrock must receive `POST /model/meta.llama3-8b-instruct-v1:0/invoke` (decoded path) whose `prompt` is
    `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n<text><|eot_id|>` followed by the open
    `<|start_header_id|>assistant<|end_header_id|>\\n\\n` header, rendered from the baked-in template.
    """
    text: Final = unique_text()
    with bedrock(gateway, _llama_reply, model=f"bedrock/invoke/{LLAMA3_8B}") as rig:
        response: Final = rig.chat(text)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{LLAMA3_8B}/invoke"
        assert body_of(sent)["prompt"] == _llama3_prompt(("user", text))


def test_llama_3_template_keeps_the_system_turn_as_its_own_header(gateway: Gateway) -> None:
    """A system message, a user turn, an assistant turn and a second user turn on Llama 3 invoke.

    The template renders every role under its own header in order (the system role is supported, so it is
    not folded into a user turn), with only the first turn carrying `<|begin_of_text|>`:
    `system`, `user`, `assistant`, `user`, then the open `assistant` header.
    """
    first: Final = unique_text()
    second: Final = unique_text()
    with bedrock(gateway, _llama_reply, model=f"bedrock/invoke/{LLAMA3_8B}") as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": first},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": second},
                ]
            }
        )
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["prompt"] == _llama3_prompt(
            ("system", "Be brief."), ("user", first), ("assistant", "ok"), ("user", second)
        )


def test_llama_3_template_trims_surrounding_whitespace_of_each_turn(gateway: Gateway) -> None:
    """The user content is sent as `"  <text>\\n  "`.

    The template pipes each content through `| trim` before `<|eot_id|>`, so Bedrock must see the bare
    text between the header and `<|eot_id|>`.
    """
    text: Final = unique_text()
    with bedrock(gateway, _llama_reply, model=f"bedrock/invoke/{LLAMA3_8B}") as rig:
        response: Final = rig.chat(f"  {text}\n  ")
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["prompt"] == _llama3_prompt(("user", text))


def test_llama_4_on_invoke_uses_the_same_llama_3_template(gateway: Gateway) -> None:
    """`bedrock/invoke/meta.llama4-maverick-17b-instruct-v1:0` with one user turn.

    `prompt_factory` matches `llama4` and `instruct` to the same Llama 3 template, so the `prompt` must be
    byte-identical to the Llama 3 rendering, and the path must name the Llama 4 id.
    """
    text: Final = unique_text()
    with bedrock(gateway, _llama_reply, model=f"bedrock/invoke/{LLAMA4_MAVERICK}") as rig:
        response: Final = rig.chat(text)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{LLAMA4_MAVERICK}/invoke"
        assert body_of(sent)["prompt"] == _llama3_prompt(("user", text))


def test_llama_invoke_maps_max_tokens_and_sampling_and_reads_generation_back(gateway: Gateway) -> None:
    """`max_tokens: 33`, `temperature: 0.4`, `top_p: 0.9` on Llama 3 invoke.

    Llama takes `max_gen_len` (`amazon_llama_transformation.py:74`), so the body must hold `max_gen_len: 33`,
    `temperature: 0.4`, `top_p: 0.9` and no `max_tokens`; the answer's `generation` becomes the content.
    """
    with bedrock(gateway, _llama_reply, model=f"bedrock/invoke/{LLAMA3_8B}") as rig:
        response: Final = rig.chat(max_tokens=33, temperature=0.4, top_p=0.9)
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert (sent["max_gen_len"], sent["temperature"], sent["top_p"]) == (33, 0.4, 0.9), sent
        assert "max_tokens" not in sent, sent
        assert response.json()["choices"][0]["message"]["content"] == "llama says hi"


def test_a_llama_invoke_call_is_priced_from_the_llama_cost_map_row(gateway: Gateway) -> None:
    """Bedrock reports `x-amzn-bedrock-input-token-count: 9` and `x-amzn-bedrock-output-token-count: 4`.

    `meta.llama3-8b-instruct-v1:0` costs 3e-07 in and 6e-07 out per token in the cost map, so the cost header
    must be 9 * 3e-07 + 4 * 6e-07 = 5.1e-06.
    UNVERIFIED: that the header counts win over the body counts on the Llama path.
    """

    def respond(_: Request) -> Reply:
        return json_reply(
            {"generation": "priced", "prompt_token_count": 9, "generation_token_count": 4, "stop_reason": "stop"},
            headers={"x-amzn-bedrock-input-token-count": "9", "x-amzn-bedrock-output-token-count": "4"},
        )

    with bedrock(gateway, respond, model=f"bedrock/invoke/{LLAMA3_8B}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(9 * 3e-07 + 4 * 6e-07)


# ------------------------------------------------------------------ DeepSeek R1 imported on InvokeModel


def test_an_imported_deepseek_r1_gets_the_deepseek_prompt_at_the_encoded_arn_path(gateway: Gateway) -> None:
    """`bedrock/deepseek_r1/arn:aws:bedrock:us-east-1:<acct>:imported-model/dec1imp7k2m` with one user turn.

    The `deepseek_r1/` spec is stripped and the ARN encoded (`base_aws_llm.py:632`), so the target must be
    `/model/<encoded ARN>/invoke`; the body `prompt` must be the DeepSeek R1 template output
    `<bos><User><text><Assistant><think>\\n` with the full-width DeepSeek special tokens.
    """
    text: Final = unique_text()

    def respond(_: Request) -> Reply:
        return json_reply(
            {"generation": "done", "prompt_token_count": 5, "generation_token_count": 1, "stop_reason": "stop"}
        )

    with bedrock(gateway, respond, model=f"bedrock/deepseek_r1/{IMPORTED_ARN}") as rig:
        response: Final = rig.chat(text)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.target == f"/model/{quoted(IMPORTED_ARN)}/invoke", sent.target
        assert body_of(sent)["prompt"] == f"{DEEPSEEK_BOS}{DEEPSEEK_USER}{text}{DEEPSEEK_ASSISTANT}<think>\n"


def test_an_imported_deepseek_r1_answer_is_split_at_the_closing_think_tag(gateway: Gateway) -> None:
    """The prompt ends in `<think>`, and Bedrock answers `generation: "weighing options</think>Final answer"`.

    `AmazonDeepSeekR1Config.transform_response` prepends `<think>` and splits with `parse_content_for_reasoning`,
    so the caller's `message.content` must be `Final answer` with the reasoning kept out of it.
    UNVERIFIED where the reasoning surfaces (`provider_specific_fields.reasoning_content` per
    `amazon_deepseek_transformation.py:67`, possibly lifted to `message.reasoning_content`).
    """

    def respond(_: Request) -> Reply:
        return json_reply(
            {
                "generation": "weighing options</think>Final answer",
                "prompt_token_count": 5,
                "generation_token_count": 6,
                "stop_reason": "stop",
            }
        )

    with bedrock(gateway, respond, model=f"bedrock/deepseek_r1/{IMPORTED_ARN}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        message: Final = response.json()["choices"][0]["message"]
        assert message["content"] == "Final answer", message
        assert "weighing options" not in message["content"]


def test_an_imported_deepseek_r1_maps_max_tokens_to_max_gen_len(gateway: Gateway) -> None:
    """DeepSeek R1 reuses the Llama parameter map (`AmazonDeepSeekR1Config(AmazonLlamaConfig)`).

    `max_tokens: 21` must reach Bedrock as `max_gen_len: 21` beside the `prompt`, with no `max_tokens` key.
    """

    def respond(_: Request) -> Reply:
        return json_reply(
            {"generation": "ok", "prompt_token_count": 5, "generation_token_count": 1, "stop_reason": "stop"}
        )

    with bedrock(gateway, respond, model=f"bedrock/deepseek_r1/{IMPORTED_ARN}") as rig:
        response: Final = rig.chat(max_tokens=21)
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert sent["max_gen_len"] == 21 and "max_tokens" not in sent, sent


# ------------------------------------------------------------------ Qwen 3 imported on InvokeModel


def test_an_imported_qwen3_gets_the_chatml_prompt_and_max_gen_len(gateway: Gateway) -> None:
    """`bedrock/qwen3/<imported ARN>` with a system and a user turn, `max_tokens: 50`, `temperature: 0.3`.

    `AmazonQwen3Config.transform_request` (`amazon_qwen3_transformation.py:89`) must send exactly
    `{"prompt": "<|im_start|>system\\nS<|im_end|>\\n<|im_start|>user\\nU<|im_end|>\\n<|im_start|>assistant\\n",
    "max_gen_len": 50, "temperature": 0.3}` to `/model/<encoded ARN>/invoke`.
    """
    text: Final = unique_text()

    def respond(_: Request) -> Reply:
        return json_reply({"generation": "hola"})

    with bedrock(gateway, respond, model=f"bedrock/qwen3/{IMPORTED_ARN}") as rig:
        response: Final = rig.chat_body(
            {
                "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": text}],
                "max_tokens": 50,
                "temperature": 0.3,
            }
        )
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.target == f"/model/{quoted(IMPORTED_ARN)}/invoke", sent.target
        assert body_of(sent) == {
            "prompt": f"<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n",
            "max_gen_len": 50,
            "temperature": 0.3,
        }


def test_an_imported_qwen3_answer_strips_the_chatml_wrapper_and_reads_usage(gateway: Gateway) -> None:
    """Bedrock answers `generation: "<|im_start|>assistant\\nhola<|im_end|>"` with `usage` 8, 2, 10.

    The caller must get `content: "hola"` (prefix and suffix removed, `amazon_qwen3_transformation.py:192`),
    `finish_reason: stop`, and `usage` 8 prompt, 2 completion, 10 total tokens.
    """

    def respond(_: Request) -> Reply:
        return json_reply(
            {
                "generation": "<|im_start|>assistant\nhola<|im_end|>",
                "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
            }
        )

    with bedrock(gateway, respond, model=f"bedrock/qwen3/{IMPORTED_ARN}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["choices"][0]["message"]["content"] == "hola"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert (body["usage"]["prompt_tokens"], body["usage"]["completion_tokens"], body["usage"]["total_tokens"]) == (
            8,
            2,
            10,
        )


def test_an_imported_qwen3_renders_an_image_part_as_the_vision_placeholder(gateway: Gateway) -> None:
    """A user turn with a text part `look` and an `image_url` part.

    Qwen 3 has no image field on InvokeModel, so the prompt must hold
    `<|im_start|>user\\nlook<|vision_start|><|image_pad|><|vision_end|><|im_end|>`.
    """

    def respond(_: Request) -> Reply:
        return json_reply({"generation": "seen"})

    with bedrock(gateway, respond, model=f"bedrock/qwen3/{IMPORTED_ARN}") as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "look"},
                            {"type": "image_url", "image_url": {"url": "https://example.com/dec1.png"}},
                        ],
                    }
                ]
            }
        )
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["prompt"] == (
            "<|im_start|>user\nlook<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
        )


# ------------------------------------------------------------------ OpenAI-format imported models and gpt-oss


def test_an_imported_openai_model_gets_an_openai_chat_body_at_the_arn_invoke_path(gateway: Gateway) -> None:
    """`bedrock/openai/<imported ARN>` is the `openai` route (`AmazonBedrockOpenAIConfig`).

    Bedrock must receive one POST whose decoded path is `/model/<ARN>/invoke` with the ARN's `/` encoded as
    `%2F` (`encode_bedrock_runtime_modelid_arn`), and an OpenAI Chat Completions body: `model` is the bare ARN
    and `messages` is the caller's list unchanged.
    """
    text: Final = unique_text()
    with bedrock(gateway, _openai_chat_reply, model=f"bedrock/openai/{IMPORTED_ARN}") as rig:
        response: Final = rig.chat(text)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{IMPORTED_ARN}/invoke"
        assert "imported-model%2F" in sent.target, sent.target
        body: Final = body_of(sent)
        assert body["model"] == IMPORTED_ARN
        assert body["messages"] == [{"role": "user", "content": text}]


def test_an_imported_openai_model_answer_is_returned_as_the_chat_completion(gateway: Gateway) -> None:
    """The imported model answers an OpenAI chat completion `imported answer` with usage 8, 2, 10.

    The caller must get that content, `finish_reason: stop` and `usage.total_tokens: 10`.
    """
    with bedrock(gateway, _openai_chat_reply, model=f"bedrock/openai/{IMPORTED_ARN}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["choices"][0]["message"]["content"] == "imported answer"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["total_tokens"] == 10


def test_an_imported_openai_model_streams_through_invoke_with_response_stream(gateway: Gateway) -> None:
    """`stream: true` on the imported OpenAI model.

    `get_complete_url` picks `invoke-with-response-stream` when streaming
    (`amazon_openai_transformation.py:96`), so the decoded path must be `/model/<ARN>/invoke-with-response-stream`.
    Only the request is pinned; the stream body shape on this route is UNVERIFIED.
    """
    with bedrock(gateway, _openai_chat_reply, model=f"bedrock/openai/{IMPORTED_ARN}") as rig:
        rig.chat(stream=True)
        assert path_of(rig.one_post()) == f"/model/{IMPORTED_ARN}/invoke-with-response-stream"


def test_bare_gpt_oss_goes_to_converse_and_is_priced_from_its_row(gateway: Gateway) -> None:
    """`bedrock/openai.gpt-oss-120b-1:0` is in the Converse list and below the GPT 5.6 native default.

    Bedrock must receive `POST /model/openai.gpt-oss-120b-1%3A0/converse`, and 11 in, 7 out tokens at the
    row's 1.5e-07 and 6e-07 give a cost header of 5.85e-06.
    """
    with bedrock(gateway, lambda _: converse_reply(), model=f"bedrock/{GPT_OSS_120B}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert rig.one_post().target == f"/model/{quoted(GPT_OSS_120B)}/converse"
        assert response_cost(response) == pytest.approx(11 * 1.5e-07 + 7 * 6e-07)


def test_gpt_oss_on_the_invoke_route_sends_an_openai_body(gateway: Gateway) -> None:
    """`bedrock/invoke/openai.gpt-oss-120b-1:0`: the invoke provider is `openai` (the id's first dotted segment).

    `AmazonInvokeConfig.transform_request` hands provider `openai` to `AmazonBedrockOpenAIConfig`, so the body
    sent to `/model/openai.gpt-oss-120b-1:0/invoke` must be OpenAI shaped: a `messages` list and no `prompt`.
    """
    text: Final = unique_text()
    with bedrock(gateway, _openai_chat_reply, model=f"bedrock/invoke/{GPT_OSS_120B}") as rig:
        rig.chat(text)
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{GPT_OSS_120B}/invoke"
        body: Final = body_of(sent)
        assert body["messages"] == [{"role": "user", "content": text}], body
        assert "prompt" not in body, body


def test_gpt_oss_on_the_invoke_route_returns_the_openai_answer(gateway: Gateway) -> None:
    """Bedrock answers the gpt-oss invoke call with an OpenAI chat completion `imported answer`.

    The caller should get that content with status 200.
    """
    pytest.skip(
        "BUG: provider config for bedrock/invoke/openai.* is AmazonInvokeConfig, whose transform_response has no "
        "openai branch and reads Titan `results[0]`, so an OpenAI-shaped answer becomes a 422 "
        "(base_invoke_transformation.py:405, common_utils.py:2359)"
    )
    with bedrock(gateway, _openai_chat_reply, model=f"bedrock/invoke/{GPT_OSS_120B}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "imported answer"


@pytest.mark.parametrize(
    ("model_id", "input_rate", "output_rate"),
    [(QWEN3_32B, 1.5e-07, 6e-07), (DEEPSEEK_V3, 5.8e-07, 1.68e-06)],
    ids=["qwen3_32b", "deepseek_v3"],
)
def test_bare_qwen_and_deepseek_ids_go_to_converse_priced_from_their_rows(
    gateway: Gateway, model_id: str, input_rate: float, output_rate: float
) -> None:
    """`bedrock/qwen.qwen3-32b-v1:0` and `bedrock/deepseek.v3-v1:0` are in the Converse list (`constants.py:1516`).

    Each must POST once to `/model/<encoded id>/converse` and be priced at its own cost map row:
    11 input and 7 output tokens at the listed per-token rates.
    """
    with bedrock(gateway, lambda _: converse_reply(), model=f"bedrock/{model_id}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert rig.one_post().target == f"/model/{quoted(model_id)}/converse"
        assert response_cost(response) == pytest.approx(11 * input_rate + 7 * output_rate)


def test_converse_reasoning_content_from_deepseek_v3_comes_back_as_reasoning_content(gateway: Gateway) -> None:
    """DeepSeek V3 on Converse answers a `reasoningContent.reasoningText` block then a text block.

    The caller must get `message.content: "answer"` and `message.reasoning_content: "thought"`, the same
    mapping Claude reasoning gets on Converse.
    """

    def respond(_: Request) -> Reply:
        return converse_reply(
            content=[{"reasoningContent": {"reasoningText": {"text": "thought"}}}, {"text": "answer"}]
        )

    with bedrock(gateway, respond, model=f"bedrock/{DEEPSEEK_V3}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        message: Final = response.json()["choices"][0]["message"]
        assert message["content"] == "answer"
        assert message["reasoning_content"] == "thought"
