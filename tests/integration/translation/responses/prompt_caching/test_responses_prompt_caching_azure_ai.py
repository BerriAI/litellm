import json
from dataclasses import replace
from typing import Final
from unittest.mock import ANY

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.responses.bases.azure_ai import CLAUDE_FABLE_5_1_TEST_CASE, CLAUDE_HAIKU_5_5_TEST_CASE
from integration.translation.runner import assert_translation

CLAUDE_HAIKU_5_5_CACHE_HIT_THINKING_SIGNATURE_0: Final = (
    "CAQSigYKEAgSGAI4AUIIdGhpbmtpbmcSDN/XoZMwEppAeWDGmxoM/11DgkoGMFpDINNAIjCfFDmj3tewcz8c9UjL+01psTk6E1rlkSIYmcJH"
    "Hx6XtPWsgJDUgKSz2IfPM/CADUMqpwUIur1FUoHdWfUasx1dqp3FuKO96MPj2UY+TH5Youusp0VbxMwTm1f03+bmK6mZkZkmgwk5qRgj56vT"
    "p6vd99GSvO3AOVdjBQrAU8m1VT1Npw1cDxDa2a0TZPxJDaAt6j2M0ujvcNHE+d3KEFsC4HTJA6qIlYJrgbuOYH90PhqVCh7TjlEpNQ/OMZ/l"
    "vSVBVvxVugSXAn01U1fhPE40RYuQxFzoAa5v0wvHzEjVOFV+piv/Wbm4QkNVxQ3OmEGuFwLAwK18XHQjqkqchQDctMJIiODTCu2s8mtTXGfX"
    "fC2Zowpyt+k8csmXbZa0QVaVY8EpmmaDv2aeunUgiPehivaXJHBl1fJA9UB9XxmQhfappJHNHvHS78GcFfRRflv4TrcpVBUFYzoerw3PKN2U"
    "vkaAwV0MlnLeOoAPsG0UcJm+ETTfh/ZbR74vPBDiFy3BSmsqQHjbBk4OGUsV15/9+/2ftIkv/2+j+4kP2wZDlMTHAwY7nNwBhdgjpdJ1Ty8K"
    "/PnzHGjNocCjLIaupO85mXFPXBLY5vsS2tq4OirPRHFrrsYOKIMY49oBVGAYcRxgFh++Zc2P7ENgrGB1AuPJmTmIN2srt2kjXfdldffHRina"
    "GeImXB1+G31+7CHmZvzeicDCUa8V3rl1h2eXufnHT9ZoGBERbzfPzW/f99EWT4UxTrPO6beZR5U7KxS6gb+B48j+6NvOuVsIOQfCt4wUxaHQ"
    "W3w8mBXZrmjxcZzRq7sgDu47pvb33GddMF22wY6ZKy9i1bfCqxewdM9CdAJATJ6U+YJDwg1bqon5D+LjernnjPgk8pXpIWewLdt0kHVAFQQz"
    "o4xTC0ofJp3L8/xNaxwmLFJ6P43o7VHsxWRKmGq/6XSJQJv9psAnccAhIF4dWxMaMHioSfj1GAE="
)

CLAUDE_HAIKU_5_5_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.mock_provider_response,
        "id": "msg_011Cg37cq1Hecba5uoPsLUid",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll read shop/validation.py to check its contents, starting with the first 60 lines "
                "or so since I don't know the file's full length yet.\n\n",
                "signature": CLAUDE_HAIKU_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
            },
            {
                "type": "tool_use",
                "id": "toolu_01MnZE6CSPdC7bHqEiAaKehz",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 873,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 175,
            "output_tokens_details": {"thinking_tokens": 75},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'll read shop/validation.py to check its contents, starting with the first 60 lines "
                        "or so since I don't know the file's full length yet.\n\n",
                        "annotations": [],
                    },
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "I'll read shop/validation.py to check its contents, starting with the first 60 lines "
                            "or so since I don't know the file's full length yet.\n\n",
                            "signature": CLAUDE_HAIKU_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "toolu_01MnZE6CSPdC7bHqEiAaKehz",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_01MnZE6CSPdC7bHqEiAaKehz",
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
            "output_tokens": 175,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 75, "text_tokens": 100},
            "total_tokens": 1052,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "I'll read shop/validation.py to check its contents, starting with the first 60 lines "
                    "or so since I don't know the file's full length yet.\n\n",
                    "signature": CLAUDE_HAIKU_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
                },
            ],
        },
    },
)

CLAUDE_FABLE_5_1_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_1_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_FABLE_5_1_TEST_CASE.mock_provider_response,
        "id": "msg_011Cg37VYYdzvnhETMSHzcVV",
        "content": [
            {"type": "text", "text": "I'll read the validation function in `shop/validation.py`."},
            {
                "type": "tool_use",
                "id": "toolu_01SamuxVRxpP9kLotKTnYMfn",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 873,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 119,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "not_available",
        },
    },
    expected_litellm_response={
        **CLAUDE_FABLE_5_1_TEST_CASE.expected_litellm_response,
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'll read the validation function in `shop/validation.py`.",
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 40}',
                "call_id": "toolu_01SamuxVRxpP9kLotKTnYMfn",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_01SamuxVRxpP9kLotKTnYMfn",
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
            "output_tokens": 119,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 119},
            "total_tokens": 996,
            "cost": None,
        },
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_HAIKU_5_5_CACHE_HIT_TEST_CASE, CLAUDE_FABLE_5_1_CACHE_HIT_TEST_CASE],
    ids=[CLAUDE_HAIKU_5_5_CACHE_HIT_TEST_CASE.id, CLAUDE_FABLE_5_1_CACHE_HIT_TEST_CASE.id],
)
def test_responses_prompt_cache_hit_azure_ai(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
