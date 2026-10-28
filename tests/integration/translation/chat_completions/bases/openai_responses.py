from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.templates import CHAT_COMPLETIONS_REQUEST, CHAT_COMPLETIONS_RESPONSE
from integration.translation.templates.openai import (
    OPENAI_RESPONSES_REQUEST_FROM_CHAT_COMPLETIONS,
    OPENAI_RESPONSES_RESPONSE,
)

GPT_5_6_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "openai/responses/gpt-5.6-sol",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/responses",
    expected_provider_headers={"authorization": "Bearer synthetic-openai-key", "content-type": "application/json"},
    expected_provider_request={
        "model": "gpt-5.6-sol",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hello."}]}],
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "stream": False,
    },
    mock_provider_response={
        "id": "resp_0c11adb5b3fcca34006ac3f043a44087d0ba76b3b869764314",
        "object": "response",
        "created_at": 1791225923,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "background": False,
        "billing": {"payer": "developer"},
        "completed_at": 1791225924,
        "error": None,
        "frequency_penalty": 0.0,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "max_tool_calls": None,
        "model": "gpt-5.6-sol",
        "moderation": None,
        "output": [
            {
                "id": "msg_0c11adb5b3fcca34006ac3f0441f4c87d0bb9cf4f5c7dfbfd1",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "Hello!"}],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "parallel_tool_calls": True,
        "presence_penalty": 0.0,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "prompt_cache_retention": "24h",
        "reasoning": {"context": "all_turns", "effort": "medium", "mode": "standard", "summary": None},
        "safety_identifier": None,
        "service_tier": "default",
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tool_usage": {
            "image_gen": {
                "input_tokens": 0,
                "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "total_tokens": 0,
            },
            "web_search": {"num_requests": 0},
        },
        "tools": [],
        "top_logprobs": 0,
        "top_p": 0.98,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 19,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 25,
        },
        "user": None,
        "metadata": {},
    },
    expected_litellm_response={
        "id": "resp_0c11adb5b3fcca34006ac3f043a44087d0ba76b3b869764314",
        "created": ANY,
        "model": "openai/responses/gpt-5.6-sol",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello!", "role": "assistant"}}],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 19,
            "total_tokens": 25,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "access_programs": {"cyber": "daybreak_blue"},
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
        "tool_usage": {
            "image_gen": {
                "input_tokens": 0,
                "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "total_tokens": 0,
            },
            "web_search": {"num_requests": 0},
        },
    },
)

GPT_6_1_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "openai/responses/gpt-6.1-sol"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={**OPENAI_RESPONSES_REQUEST_FROM_CHAT_COMPLETIONS, "model": "gpt-6.1-sol"},
    mock_provider_response={
        **OPENAI_RESPONSES_RESPONSE,
        "model": "gpt-6.1-sol",
        "id": "resp_0e8134ce088f63e7006ad023a2e5b887d0a0f422419e0bfbe7",
        "created_at": 1792025506,
        "completed_at": 1792025510,
        "instructions": (
            "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
            "Explain which file you need next, then call a read-only tool. Never change files. Use the "
            "chat_completions client for this project review. Consider zero, negative and non-integer "
            "quantities before choosing the next file to inspect."
        ),
        "output": [
            {
                "id": "rs_0e8134ce088f63e7006ad023a3ff5887d0801528ceb0190f4b",
                "type": "reasoning",
                "content": [],
                "encrypted_content": "gAAAAABq0COmpSSf5bN4H0xC2pbhfp_niGPAm58n4-JyZsHy0ytyhg6KzzTZiR0cjuNAHvZR_w8oavaY1zzH-MkWItVhAx-PQJQ560Da0Dwqi_wwJ0831R8W6lxyeFxgBZXpZ4_TsbT1qwt7IwVC6fJE4gDeFGiuo4H7plTfbE7C2DaO_0fkI5Zt_iRZLA3VOJPFnzegipK4A5AxckZNVZLI705xzMfi7KU6J886jCmsqCxjrQKLwP4fB0jQoRidBCensrctV2C7imXgTAlTwOysSd8gf7Yhu7A9qD1rUj_jvLGJCiqBCe9Cyv9I7cX82VewV4029mGOngvQOUAgVbqT_i9LkQ3HfBXu4SYTK_LXU2P0lw_P69qKYGTZfdwxqui8VVtgJI7kDrZY-l7DvqaIncKJ-LgDJBvxaYDMM7H1Co68Yd64RM3U1v4kTezKN_MBg-1Ku0m1e3Zi7YZDHOHkFDDGFXzXCpbGkgDJCHsG7i-UTE1uAjkgdP9h0MUFosMYB8pl2NxtN8NnEHsBfyYBV56Sms8ObilQaK2eEWvnDH_wkGLt2RAGOJwfovSEwKnWgOemEZtA_IYbxhRNupRJkPy0oJ4xPdAxIICncSkwaF_NMYeIetF-0tvqK2mqWZVnI13w_CPUW0WI0nlfSnlqO-7gD4Xh80-cvFAqr44X_yaRODPOOc6aWoXb_23A2AoFW6vEKufFf3zni2JEHiCREgvUAp8p931WYo7H_lDpOqIHGiQVjvRfUN9pRPnpPbF6NwCUNalgQPt4UZal_ubh3kdF1NRqKuTQMcalPfuw7Zdc7-ZFTV3JdUApBwsF2jGFXzAfmwHytnOohHOAm79n_7ORaJV3ecKjyg0t-N777eu-7LJX79tvy1tNXR-0DsodobU-mHuyxqC4Qhmqk8jC1IeNG7du1IUwpFJnOyM4O1D6xUYGslBWXsIa_exQpTkxH0PYSYeCGA9GCeWsrn8gSkOgwBmtsC56Bow8VldAr_t_AS5UIrSUsNvn1SF-nZJKbe1mvL-egWOuQq8dZb3wEF7vVxqgn4_f54pGXyDeEdxPdxAKhqT672MKb3l_K11Yn10kbhNr1XqX-m323zLcNizjPlH8rJGUWht7RGQd6VOxEjmhKC_ZWFsu7wQgqsGX0BQ2e-6hZE2Kq_9u_MfSieYJyt9jb6XXrqefSP2tyWsrrS6HrBBrSkh2d5PfiP4MQ2gTesP9tj-yVlq4JU3FZMgqS5W7m2BmhD8gGonaLSK5zJboTYjpZHoS0swfxX2_BJDiARNrPFZ58t6sSXDZauTZq8CbdLdXUBA6RaO1kp-yscPIuB2DHjPjPUVrRYbNWSPIcjqHDYY11UHPs7lWL8U5j-j3EfYwiwQ85YnVfoB7jzJFeiywAUvbum20MJXBOE2wrvrR6zXJAGYlEwxKbl_ibZULRRuJ-VSGeijiDNtavD5v5-AAmow5yPuuVUH9_kx7Un8ZbbESZH1rptCToDmxeRrIa51Xw8fIbONz1fVNzljCUvqvsoyjPQAlaZy_YfLIOYD5MVhVX8XiR8mIeSna4ELdWXu8EEmPWyl676GbSZvMzQmBVqDFeELEusHm_UNX3oI5oOMJQ8ZA9p-JyJlx8LWDe3wqTaKKzoajxn2Wz6qvjKFStkGrk0voFzEPEnBbIAIPkh7Uv2ORupzlQeF6Hor11DxlDAIaUQf1344U37C-a1ODbqL8tg7HOm1WSKdwWeKG2-SYk9K8yB1hEZuqmxHquiXybUU-EYrQFGv88d-K0ou-wAjolMdXlIUaT5-OmqA8HCprI4Nr1BdxQYLLwlNyxm5Y610a1Bd9clo8YbW-I_zSNVm74nF1_HzKP3Ym9H3_f1ZWj55BOAw9i1jmk6kb7P2SAomQ9WVCHvhbyPUhrMMlDxc5VpQOMlFRIZjCEb4wirnD4oonK5mt_TY1l_XMc7DAFb91wpdF3dB6Nh8W2_YYC8_vHfvvSq5WtVKBVIBzZM7l188EX3zHUoe4a9pbQj2Yj2-fAF4qnae4DZ1IwYvoUrln3YETLC37Qb7c3cwM_-i9YLiB_F-qA-pVnmwH8Ah2a3cOF-sl50BE7ey_rHw0fhnflc2bicoLMA82gXdE3iuAsPX-sJvouXgLB1KYmFZt_8Hm3QlrT2c8mj_Jt10YdPC_haBCvzOQDvi5kfPM2upPrvgm7bPEgyjLZECEaQj7bjgAsZc7PqnNp0u41Quq2WxUbjMrE6BPjcrhF5Llx6PQjrz2d5_023M1xBMGxKNXaLNmyi91YpvC0Sqd9qUKndUQ6fPA4z03_FP7-UnzYPuJQaBX662f1Vd-sSUvvM4zdPw6YX-muGqHF5qld2He6eMaUUXhjDuwvy2LW5WRQfSbWJ89XsUYqDwDM0j8TEexma6UZ07uNV_xjse6P8JfL-L8BDiHkR7ZNRMDSYqsHQjqwoYZNUs25XwWPr7kjn-43H0mW_dXPcManZ6qBlKKETmdtr8CZrtNxZ1_TnHUeNkATzhmtzyLQRmouxUq6tbPMumlcXcwhvDBSKSR33a4LZG1kknf_YlTDLA4x7sHImDRMYgM9d6SYxBuTSq1OfTtSv7i4Wz5_Pef3iNFI8JH6Q6b24vED1lnWvVEImDKReDYdDLoNJr2BbSt6iZ0hsGSQ5EMsE3jlizNIBwdqI5z2MKTCYWAXSbxXe62zgaklRD6IoCER9h4SqjsnOh9esKu1bg-AkSa04Le9PP4i2whspq6Xa7f4DJpRYb-hEdY",
                "summary": [],
            },
            {
                "id": "msg_0e8134ce088f63e7006ad023a5aa0c87d0aee7de2e31f58a5c",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "I’ll read `shop/validation.py` first to see how it handles zero, negative "
                        "values, and non-integer quantities.",
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0e8134ce088f63e7006ad023a615bc87d0b7aaafc731d1f086",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_sPhtuzlfkfNe16kBteKPS72V",
                "name": "read_file",
            },
        ],
        "tools": [
            {
                "type": "function",
                "description": "Search source and test files for a literal term.",
                "name": "search_files",
                "output_schema": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query", "paths"],
                    "additionalProperties": False,
                },
                "strict": False,
            },
            {
                "type": "function",
                "description": "Read a range of lines from a project file.",
                "name": "read_file",
                "output_schema": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False,
                },
                "strict": False,
            },
        ],
        "usage": {
            "input_tokens": 312,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 166,
            "output_tokens_details": {"reasoning_tokens": 100},
            "total_tokens": 478,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "openai/responses/gpt-6.1-sol",
        "id": "resp_0e8134ce088f63e7006ad023a2e5b887d0a0f422419e0bfbe7",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I’ll read `shop/validation.py` first to see how it handles zero, "
                    "negative values, and non-integer quantities.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                                "name": "read_file",
                            },
                            "id": "call_sPhtuzlfkfNe16kBteKPS72V",
                            "type": "function",
                        }
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_0e8134ce088f63e7006ad023a3ff5887d0801528ceb0190f4b",
                            "encrypted_content": "gAAAAABq0COmpSSf5bN4H0xC2pbhfp_niGPAm58n4-JyZsHy0ytyhg6KzzTZiR0cjuNAHvZR_w8oavaY1zzH-MkWItVhAx-PQJQ560Da0Dwqi_wwJ0831R8W6lxyeFxgBZXpZ4_TsbT1qwt7IwVC6fJE4gDeFGiuo4H7plTfbE7C2DaO_0fkI5Zt_iRZLA3VOJPFnzegipK4A5AxckZNVZLI705xzMfi7KU6J886jCmsqCxjrQKLwP4fB0jQoRidBCensrctV2C7imXgTAlTwOysSd8gf7Yhu7A9qD1rUj_jvLGJCiqBCe9Cyv9I7cX82VewV4029mGOngvQOUAgVbqT_i9LkQ3HfBXu4SYTK_LXU2P0lw_P69qKYGTZfdwxqui8VVtgJI7kDrZY-l7DvqaIncKJ-LgDJBvxaYDMM7H1Co68Yd64RM3U1v4kTezKN_MBg-1Ku0m1e3Zi7YZDHOHkFDDGFXzXCpbGkgDJCHsG7i-UTE1uAjkgdP9h0MUFosMYB8pl2NxtN8NnEHsBfyYBV56Sms8ObilQaK2eEWvnDH_wkGLt2RAGOJwfovSEwKnWgOemEZtA_IYbxhRNupRJkPy0oJ4xPdAxIICncSkwaF_NMYeIetF-0tvqK2mqWZVnI13w_CPUW0WI0nlfSnlqO-7gD4Xh80-cvFAqr44X_yaRODPOOc6aWoXb_23A2AoFW6vEKufFf3zni2JEHiCREgvUAp8p931WYo7H_lDpOqIHGiQVjvRfUN9pRPnpPbF6NwCUNalgQPt4UZal_ubh3kdF1NRqKuTQMcalPfuw7Zdc7-ZFTV3JdUApBwsF2jGFXzAfmwHytnOohHOAm79n_7ORaJV3ecKjyg0t-N777eu-7LJX79tvy1tNXR-0DsodobU-mHuyxqC4Qhmqk8jC1IeNG7du1IUwpFJnOyM4O1D6xUYGslBWXsIa_exQpTkxH0PYSYeCGA9GCeWsrn8gSkOgwBmtsC56Bow8VldAr_t_AS5UIrSUsNvn1SF-nZJKbe1mvL-egWOuQq8dZb3wEF7vVxqgn4_f54pGXyDeEdxPdxAKhqT672MKb3l_K11Yn10kbhNr1XqX-m323zLcNizjPlH8rJGUWht7RGQd6VOxEjmhKC_ZWFsu7wQgqsGX0BQ2e-6hZE2Kq_9u_MfSieYJyt9jb6XXrqefSP2tyWsrrS6HrBBrSkh2d5PfiP4MQ2gTesP9tj-yVlq4JU3FZMgqS5W7m2BmhD8gGonaLSK5zJboTYjpZHoS0swfxX2_BJDiARNrPFZ58t6sSXDZauTZq8CbdLdXUBA6RaO1kp-yscPIuB2DHjPjPUVrRYbNWSPIcjqHDYY11UHPs7lWL8U5j-j3EfYwiwQ85YnVfoB7jzJFeiywAUvbum20MJXBOE2wrvrR6zXJAGYlEwxKbl_ibZULRRuJ-VSGeijiDNtavD5v5-AAmow5yPuuVUH9_kx7Un8ZbbESZH1rptCToDmxeRrIa51Xw8fIbONz1fVNzljCUvqvsoyjPQAlaZy_YfLIOYD5MVhVX8XiR8mIeSna4ELdWXu8EEmPWyl676GbSZvMzQmBVqDFeELEusHm_UNX3oI5oOMJQ8ZA9p-JyJlx8LWDe3wqTaKKzoajxn2Wz6qvjKFStkGrk0voFzEPEnBbIAIPkh7Uv2ORupzlQeF6Hor11DxlDAIaUQf1344U37C-a1ODbqL8tg7HOm1WSKdwWeKG2-SYk9K8yB1hEZuqmxHquiXybUU-EYrQFGv88d-K0ou-wAjolMdXlIUaT5-OmqA8HCprI4Nr1BdxQYLLwlNyxm5Y610a1Bd9clo8YbW-I_zSNVm74nF1_HzKP3Ym9H3_f1ZWj55BOAw9i1jmk6kb7P2SAomQ9WVCHvhbyPUhrMMlDxc5VpQOMlFRIZjCEb4wirnD4oonK5mt_TY1l_XMc7DAFb91wpdF3dB6Nh8W2_YYC8_vHfvvSq5WtVKBVIBzZM7l188EX3zHUoe4a9pbQj2Yj2-fAF4qnae4DZ1IwYvoUrln3YETLC37Qb7c3cwM_-i9YLiB_F-qA-pVnmwH8Ah2a3cOF-sl50BE7ey_rHw0fhnflc2bicoLMA82gXdE3iuAsPX-sJvouXgLB1KYmFZt_8Hm3QlrT2c8mj_Jt10YdPC_haBCvzOQDvi5kfPM2upPrvgm7bPEgyjLZECEaQj7bjgAsZc7PqnNp0u41Quq2WxUbjMrE6BPjcrhF5Llx6PQjrz2d5_023M1xBMGxKNXaLNmyi91YpvC0Sqd9qUKndUQ6fPA4z03_FP7-UnzYPuJQaBX662f1Vd-sSUvvM4zdPw6YX-muGqHF5qld2He6eMaUUXhjDuwvy2LW5WRQfSbWJ89XsUYqDwDM0j8TEexma6UZ07uNV_xjse6P8JfL-L8BDiHkR7ZNRMDSYqsHQjqwoYZNUs25XwWPr7kjn-43H0mW_dXPcManZ6qBlKKETmdtr8CZrtNxZ1_TnHUeNkATzhmtzyLQRmouxUq6tbPMumlcXcwhvDBSKSR33a4LZG1kknf_YlTDLA4x7sHImDRMYgM9d6SYxBuTSq1OfTtSv7i4Wz5_Pef3iNFI8JH6Q6b24vED1lnWvVEImDKReDYdDLoNJr2BbSt6iZ0hsGSQ5EMsE3jlizNIBwdqI5z2MKTCYWAXSbxXe62zgaklRD6IoCER9h4SqjsnOh9esKu1bg-AkSa04Le9PP4i2whspq6Xa7f4DJpRYb-hEdY",
                            "summary": [],
                        }
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 166,
            "prompt_tokens": 312,
            "total_tokens": 478,
            "completion_tokens_details": {"reasoning_tokens": 100},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
        "tool_usage": {
            "image_gen": {
                "input_tokens": 0,
                "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "total_tokens": 0,
            },
            "web_search": {"num_requests": 0},
        },
    },
)
