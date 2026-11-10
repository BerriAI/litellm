import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Final
from unittest.mock import ANY

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import RejectionTestCase, TranslationTestCase
from integration.translation.responses.bases.anthropic import CLAUDE_HAIKU_5_5_TEST_CASE, CLAUDE_SONNET_5_5_TEST_CASE
from integration.translation.runner import assert_rejection, assert_translation
from pydantic import JsonValue

CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0: Final = (
    "CAQSowkKEAgSGAI4AUIIdGhpbmtpbmcSDOwQCAqgczGjZGBQDhoMT4bVPdH6VMFI48MwIjC029tj1Uerfhj93na2igkejc43fgJVSxspenJp"
    "9C9WIR+Pw8eVIU2u2oLaCRh9ZaAqwAgXjHbSSi1epawZB+b2WVRclEY69ZZaFJOtC+iA5VBnzI70bhzfYwUL2Bh9vuETb9UyiGNuXlEjTY73"
    "AUNWyLn1JGS1y4ZqwoN5u8TZV9zR7nnPoXTIdAlp8vWz0uyE2UgAlqyNX8XNEq2+CsK9R2RFskyXhd8FRSQ49QPj8e+Dcv0OLLpQrrPIa7Nv"
    "aG7A0E/5D/VEU94iC538mQ/Eh2NcwBS+gqCCGbeLhHMvmSZqDUp4s4kGMhBPILcAEx1f1FE1ZCsUSKTPMR91XJsarHkdOjgpgBx2yiNr7EAa"
    "qP8DEDTMIUZaM4rHuJfCIE2Gu/KOvBaCV5b8gl4HhvONjxfnRqoDkwB79hhB5kARXXcoYPigyJAebXTUKogtdfrLlqJ18KqFiq+gwLk2IP5v"
    "l78q3TNiDz4hWoMFdXeaffH2++6FscuZU3im68D2aSWzfUvJoG/5kqFvMKsMUMQ7JTRq8bwdy0Pzc38peH9hgy9OwYjShmmp/lX+8tH4Mpu9"
    "4ScwOx0oCKdlAJyFBd4I9kOJ03SM2bZI/NkWTZ11LrtLNqxUpdzESIuaPqcwpBlMQu09oKUthvL7rs5U4WhOD7V+l92UFphiR+hI2Wc+d7Ys"
    "kh5PgGpiIGxq5QpMGUOwgy/+/KyZcHc3EDAD4oY/BVpqc4zjjOdFOc6KvtKLn1lqZc47+XL2HxI0Kh7qPqCue3yArxkn16Jk+bdinaJPfVQI"
    "rKxIav0gWowtmxcjFOVYXRDzhlaqOfnGZOLDOwJ8TpESRBxKWJ3fYrUjLZjY6X9MdPWQQ88h/DsQijwqXNQFYs/hMOMqEYcW7oIMmbQXv/KT"
    "05nX5aMMo0AFD3Iych6bVwNC0b0S87wSolWLBBhnhYyEUJfAW6p28ARtLdkJlqrsOSupzMkKxTxjXafEbM0GcObF9FgaeDh/3gpG9ahFex8W"
    "MaBqIakbogrEeQrrt21DPgeb6lWsEMj6lEupfmY5a3Ch2fsKYIFmvvQ4JN9b5C0K7H3GFfH4qTclqTEegX4Gzm7tjuwQsv9INVilYVUkpMgo"
    "7JD52wVVtqSJqGgQ8xKLWXR1M5oMjGCE2bRhbgbVfMarvhJW17FJvS/7Yugcgxu/n/dhmo6pxFHlSujiPOBODr8fWPGJKwhpGqKdi+BUQyGF"
    "b41hPjRZEuyYSO7OdmqblZ4MlkHFLt7CfCvQQZePBvNyDniiCuwZBRgO3zlRO5VKVsom3KkzZfG+tsfzKwMMrNeN1sQ7udJsp2giBRN1wBu6"
    "LBYfs8ASv8Y1OEmFVew5/aByf7oGaH74DlxLi/E0cquY37LiUWl1iTx4SS13wv0NpMaU03X6ahmsEpdHbR2E2I8sBkn8F7b5ydnWpp9OT8Tg"
    "9tWMVfprteYjcVkJoKjkp4M7QG4jKSb3/2eofN44aPeX3KfkpbjkMONZkQ7A8z4o1ezDw2swOGVGUBgB"
)

CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_1: Final = (
    "CAQSmgcKEQgSGAI4AUIJbmFycmF0aW9uEgytCDiV0k0xcr3T9oYaDBoLSK23c8n6+DLIhCIwebnBjlb04xHa2geO2nvLWXbdL/mEVCXz+MjE"
    "oDSqFLe0Ivh+hn+mIUtg83a3B7NDKrYGXA1HllbZ1yM4570ouFk4bKliGKHWdi0coYw1u9YGlThcas5S7s1hL2+R1hqj0QBP/M/hDMFNbf/1"
    "T/idB+7QKbXFsYtonKoTlaQdwhXJehGZfIPNXFM7TYE8hB1KFnJpkJGLM2YU5j42Y1HL0kTh8lNvTdsl2GfF4elrtPAJjeAHEME2b4WfGEbp"
    "gJbPFJ6qPWOaVPJbYa7ZpdlOcGYviY9uiXDrXdUNkJx+fQgZ78pX5MIZPUeCkiGIDmxQJtuJC/961w9DEutBOy2KuaQHkuWDZqRCeFAzgW9s"
    "hmvT+jo5SxPDtn5E7KTt8DoDoClb6VNEGd+5BX9SSIfl39XyWKsrT7fs/dXDzicl4wUYbD527ZLWM6kJd3PpWcV6He3ltvELyXjYjCGv79+D"
    "pM87AwmmpkuYiuTyXrwdDBRfHSf00eoLyFzW/FZNYanwcbTTrReiW1ehhGivaLe1v0ppO2NLSHli2u1+xpdR8bK8Nn8J4q+QnCwON70fuqf+"
    "3sQLrspyOO7PlrpV2Ybq8Sng0dC0mBAAv7CpHeR1IaQPC936cfpL7bbB75/m15Pqy2R8A257/g3zZIA53oxwW2B3y/UNzL5m6XSzpmLaYmRM"
    "z4ahp50+rytQTCqjG0mr27g2vWerU8U+qqjyzTKrT795/CkVDQ2koodup6jLsfwx1W6bq5aJRVYUEY1ZJKywZylilmWrfTz8MZ4TajBx/yUu"
    "5X12jCdjzTOLVdGeOgD5a2lsxZzgU/Y9RgyzeT6RHKNnbjVUXaJa0g5K0MclCfVWFPhuIAwn2Gya/o6eF1k4Pebzr8aIOU8nY3YxXwN9hytO"
    "YCwxo4TF2eV8F9ttuBTezOFpJiG2VpvHIRh2/5AVVPxe1c2xWvZvciHHrH14G9rVxmoUzDraCoPiz2XUfCY7e58BRD+wOH3YFnf8uckCmHCj"
    "KFYclanFwFJyEfjwssCxcTZvC/AN2aATw4f5WQ1vow9uPO9wxkpc7af/VNQ2MPGB87BaVg/hTRhSgwJLSxZIOKUKMMJzJ/4rgAZRSIJawg33"
    "ONNvhW41/bXB33KXdqbvhVIZGkK1YGv8gjMlqIdTfdYDonWeGAE="
)

CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_SONNET_5_5_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.mock_provider_response,
        "id": "msg_011Cg32rEsFymVyWbaNmV7tB",
        "content": [
            {
                "type": "thinking",
                "thinking": "Since the files are independent, I can just read both in the same call, "
                "listing validation first to honor the user's stated order.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "I'll check `shop/validation.py` to see how `validate_quantity` handles "
                "zero, negative, and non-integer values, and also read "
                "`tests/test_validation.py` to confirm the expected behavior for zero.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_01RoPyymXeGn7TS6uLnKMkzo",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01Cv9tx97CJhxergPDGZLTxC",
                "name": "read_file",
                "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 40},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 873,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 447,
            "output_tokens_details": {"thinking_tokens": 251},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.expected_litellm_response,
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Since the files are independent, I can just read both in the same "
                        "call, listing validation first to honor the user's stated order.\n"
                        "\n"
                        "I'll check `shop/validation.py` to see how `validate_quantity` "
                        "handles zero, negative, and non-integer values, and also read "
                        "`tests/test_validation.py` to confirm the expected behavior for "
                        "zero.\n"
                        "\n",
                        "annotations": [],
                    }
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "Since the files are independent, I can just read both in the same call, listing "
                            "validation first to honor the user's stated order.\n"
                            "\n",
                            "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": "I'll check `shop/validation.py` to see how `validate_quantity` handles zero, negative, "
                            "and non-integer values, and also read `tests/test_validation.py` to confirm the "
                            "expected behavior for zero.\n"
                            "\n",
                            "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_1,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "toolu_01RoPyymXeGn7TS6uLnKMkzo",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_01RoPyymXeGn7TS6uLnKMkzo",
                "namespace": None,
                "status": "completed",
            },
            {
                "arguments": '{"path": "tests/test_validation.py", "start_line": 1, "end_line": 40}',
                "call_id": "toolu_01Cv9tx97CJhxergPDGZLTxC",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_01Cv9tx97CJhxergPDGZLTxC",
                "namespace": None,
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 877,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 873,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 447,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 251, "text_tokens": 196},
            "total_tokens": 1324,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Since the files are independent, I can "
                    "just read both in the same call, "
                    "listing validation first to honor the "
                    "user's stated order.\n"
                    "\n",
                    "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
                },
                {
                    "type": "thinking",
                    "thinking": "I'll check `shop/validation.py` to see "
                    "how `validate_quantity` handles zero, "
                    "negative, and non-integer values, and "
                    "also read `tests/test_validation.py` "
                    "to confirm the expected behavior for "
                    "zero.\n"
                    "\n",
                    "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_1,
                },
            ],
        },
    },
)


@pytest.mark.parametrize(
    "case", [CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE], ids=[CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE.id]
)
def test_responses_prompt_cache_hit_anthropic(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)


def unsupported_params_error(fields: str, *, bridge: bool) -> Mapping[str, JsonValue]:
    bridge_sentence: Final = "The Responses API to chat completions bridge cannot translate them. " if bridge else ""
    return {
        "error": {
            "message": f"litellm.UnsupportedParamsError: anthropic does not support parameters: {fields}, for "
            f"model=claude-haiku-5-5. {bridge_sentence}To drop these, set `litellm.drop_params=True` or for proxy:\n\n"
            "`litellm_settings:\n drop_params: true`\n. \n If you want to use these params dynamically send "
            f"allowed_openai_params={fields} in your request.\n\nLiteLLM: model group 'anthropic/claude-haiku-5-5' "
            "failed with the error above. No fallback was attempted.",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    }


PROMPT_CACHE_OPTIONS: Final[Mapping[str, JsonValue]] = {
    "mode": "explicit",
    "ttl": "30m",
    "comparison_response_id": "resp_synthetic_prior_01",
    "prewarm": True,
}

CLAUDE_HAIKU_5_5_PROMPT_CACHE_KEY_UNSUPPORTED_TEST_CASE: Final = RejectionTestCase(
    scenario="prompt_cache_key_unsupported",
    litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "prompt_cache_key": "synthetic-cache-key"},
    expected_litellm_status_code=400,
    expected_litellm_response=unsupported_params_error("['prompt_cache_key']", bridge=False),
)

CLAUDE_HAIKU_5_5_PROMPT_CACHE_KEY_DROPPED_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="prompt_cache_key_dropped",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "prompt_cache_key": "synthetic-cache-key",
        "drop_params": True,
    },
)

CLAUDE_HAIKU_5_5_PROMPT_CACHE_RETENTION_UNSUPPORTED_TEST_CASE: Final = RejectionTestCase(
    scenario="prompt_cache_retention_unsupported",
    litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "prompt_cache_retention": "24h"},
    expected_litellm_status_code=400,
    expected_litellm_response=unsupported_params_error("['prompt_cache_retention']", bridge=False),
)

CLAUDE_HAIKU_5_5_PROMPT_CACHE_RETENTION_DROPPED_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="prompt_cache_retention_dropped",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "prompt_cache_retention": "24h",
        "drop_params": True,
    },
)

CLAUDE_HAIKU_5_5_PROMPT_CACHE_OPTIONS_UNSUPPORTED_TEST_CASE: Final = RejectionTestCase(
    scenario="prompt_cache_options_unsupported",
    litellm_endpoint=CLAUDE_HAIKU_5_5_TEST_CASE.litellm_endpoint,
    litellm_request={**CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request, "prompt_cache_options": PROMPT_CACHE_OPTIONS},
    expected_litellm_status_code=400,
    expected_litellm_response=unsupported_params_error("['prompt_cache_options']", bridge=True),
)

CLAUDE_HAIKU_5_5_PROMPT_CACHE_OPTIONS_DROPPED_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="prompt_cache_options_dropped",
    litellm_request={
        **CLAUDE_HAIKU_5_5_TEST_CASE.litellm_request,
        "prompt_cache_options": PROMPT_CACHE_OPTIONS,
        "drop_params": True,
    },
)


@pytest.mark.parametrize(
    "case",
    [
        CLAUDE_HAIKU_5_5_PROMPT_CACHE_KEY_UNSUPPORTED_TEST_CASE,
        CLAUDE_HAIKU_5_5_PROMPT_CACHE_KEY_DROPPED_TEST_CASE,
        CLAUDE_HAIKU_5_5_PROMPT_CACHE_RETENTION_UNSUPPORTED_TEST_CASE,
        CLAUDE_HAIKU_5_5_PROMPT_CACHE_RETENTION_DROPPED_TEST_CASE,
        CLAUDE_HAIKU_5_5_PROMPT_CACHE_OPTIONS_UNSUPPORTED_TEST_CASE,
        CLAUDE_HAIKU_5_5_PROMPT_CACHE_OPTIONS_DROPPED_TEST_CASE,
    ],
    ids=lambda case: case.id,
)
def test_responses_prompt_cache_params_anthropic(
    case: TranslationTestCase | RejectionTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    match case:
        case RejectionTestCase():
            assert_rejection(case, gateway, provider)
        case TranslationTestCase():
            assert_translation(case, gateway, provider)
