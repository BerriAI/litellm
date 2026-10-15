from typing import Final

from integration.translation.case import TranslationTestCase
from integration.translation.messages.templates import MESSAGES_BRIDGE_RESPONSE, MESSAGES_REQUEST
from integration.translation.templates.openai import OPENAI_RESPONSES_REQUEST_FROM_MESSAGES, OPENAI_RESPONSES_RESPONSE

GPT_5_4_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "openai/gpt-5.4",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/responses",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-openai-key"},
    expected_provider_request={
        "model": "gpt-5.4",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hello."}]}],
        "include": ["reasoning.encrypted_content"],
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
    },
    mock_provider_response={
        "id": "resp_0d51005efcef0683006ac3f00d0a3087d098c079bde6c6ee3c",
        "object": "response",
        "created_at": 1791225869,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "background": False,
        "billing": {"payer": "developer"},
        "completed_at": 1791225869,
        "error": None,
        "frequency_penalty": 0.0,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "max_tool_calls": None,
        "model": "gpt-5.4-2026-03-05",
        "moderation": None,
        "output": [
            {
                "id": "msg_0d51005efcef0683006ac3f00d8dc487d0aac89eb25a71a11f",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "Hello."}],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "parallel_tool_calls": True,
        "presence_penalty": 0.0,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "prompt_cache_retention": "24h",
        "reasoning": {"context": "current_turn", "effort": "none", "mode": "standard", "summary": None},
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
        "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDo3NGI3ZDQzZjBmNjQ1MjlkYWY4NWQ4MjRmNjE3ZGYzNDRjZTdhNGI3NjI3YTYzMDMzZGUzMDVmNTE2NTRlZjc2O3Jlc3BvbnNlX2lkOnJlc3BfMGQ1MTAwNWVmY2VmMDY4MzAwNmFjM2YwMGQwYTMwODdkMDk4YzA3OWJkZTZjNmVlM2M=",
        "type": "message",
        "role": "assistant",
        "model": "openai/gpt-5.4",
        "stop_sequence": None,
        "usage": {"input_tokens": 19, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)


GPT_5_6_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "openai/gpt-5.6-sol",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/responses",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-openai-key"},
    expected_provider_request={
        "model": "gpt-5.6-sol",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hello."}]}],
        "include": ["reasoning.encrypted_content"],
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
    },
    mock_provider_response={
        "id": "resp_006b1d3db27ce558006ac3f0119c5887d0b0fda575ab760d9d",
        "object": "response",
        "created_at": 1791225873,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "background": False,
        "billing": {"payer": "developer"},
        "completed_at": 1791225874,
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
                "id": "msg_006b1d3db27ce558006ac3f012169487d0b3574a945281b342",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "Hello."}],
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
        "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDplNDZkYjA4Yjk5OGY1YmZmMTFhYTViNWIxMjJkMGQxZGVhZDM5MzI2MWZlZDU2OTc3M2FmOTNlMzBlYTMzNTc4O3Jlc3BvbnNlX2lkOnJlc3BfMDA2YjFkM2RiMjdjZTU1ODAwNmFjM2YwMTE5YzU4ODdkMGIwZmRhNTc1YWI3NjBkOWQ=",
        "type": "message",
        "role": "assistant",
        "model": "openai/gpt-5.6-sol",
        "stop_sequence": None,
        "usage": {"input_tokens": 19, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)


GPT_5_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "openai/gpt-5.6-luna",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/responses",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer synthetic-openai-key"},
    expected_provider_request={
        "model": "gpt-5.6-luna",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hello."}]}],
        "include": ["reasoning.encrypted_content"],
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
    },
    mock_provider_response={
        "id": "resp_0fc0133ee7ba1828006ac3f015182c87d0bbc9b785a3a5a335",
        "object": "response",
        "created_at": 1791225877,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "background": False,
        "billing": {"payer": "developer"},
        "completed_at": 1791225877,
        "error": None,
        "frequency_penalty": 0.0,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "max_tool_calls": None,
        "model": "gpt-5.6-luna",
        "moderation": None,
        "output": [
            {
                "id": "msg_0fc0133ee7ba1828006ac3f015b22887d090ad19f7bfc86fd3",
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
        "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDo1MGY4ZWQzY2Q2ZWQ0NWFlMGQ3YWYxODI5YTJiNzg5M2I3ZWIxMjA5MTQ2Mzk5MTMyMGMwODFlNzE1NTc2OTM2O3Jlc3BvbnNlX2lkOnJlc3BfMGZjMDEzM2VlN2JhMTgyODAwNmFjM2YwMTUxODJjODdkMGJiYzliNzg1YTNhNWEzMzU=",
        "type": "message",
        "role": "assistant",
        "model": "openai/gpt-5.6-luna",
        "stop_sequence": None,
        "usage": {"input_tokens": 19, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

GPT_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "openai/gpt-6-luna"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={**OPENAI_RESPONSES_REQUEST_FROM_MESSAGES, "model": "gpt-6-luna"},
    mock_provider_response={
        **OPENAI_RESPONSES_RESPONSE,
        "model": "gpt-6-luna",
        "id": "resp_03ec491c5caae029006ad0239c083487d099a21d60be6b5fde",
        "created_at": 1792025500,
        "access_programs": {"cyber": "daybreak_blue"},
        "completed_at": 1792025503,
        "instructions": (
            "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
            "Explain which file you need next, then call a read-only tool. Never change files. Use the "
            "messages client for this project review. Consider zero, negative and non-integer quantities "
            "before choosing the next file to inspect."
        ),
        "output": [
            {
                "id": "rs_03ec491c5caae029006ad0239c80c487d0b425db4d74878ab3",
                "type": "reasoning",
                "content": [],
                "encrypted_content": "gAAAAABq0COfZvmrZboi5Ffbs6U_yMF1jaQmGVq5BnqydEOn98Tt_afVrM4H653kYvwFlbyesZJUOa4MLSEHMVc5ZNHlI88WpsGB2C0cdOmXO-S8Xz2wv2TYaf-d7ux1ejKPGkbop3sk2yS2XxQahiybVv5r_opD4YkEVxPoKhnVTfNF0bcglJj1JQbXYSE5qNyAVFeO6kkYN_sRP628gc_-txGgIgJbLn-uTsZ_3aQZ8TmejJJaiXLcauCUyCnYHtw0BncdwxaRiDsAUhrL_HuwLgGc3KHP0mc_IjapNLhVAcqeihSO4rmidsYX6Y7bjb4yhqPqE0-NnPZRO8BOcpIb-qU89htZHrkaSA_MrZj2FD4fk8_btQHTF9C9hM-UjRUJxqMtyuOyNMZdWjBgErNoCzslro-Sty8pWlIYRjzOlQQHZLBrC3EuKvOTEBNNjWXeRCGwZTZR6C5SyE0mj11YCwi67QeYfp8ujvrziJq1Bt7u78vnlPY8bMd0AUz0SURgP2jnlreu11EMWZuR-u6ZMLqyu1HeunBMs82y-5ef37g3evBV7Ym6AjwsiP0SqTM3KeaY5HMkgaZi9G2gwO2g8ZkLsJpy-Bk01KwNYD7QXAxrUXziEAzby_AqAOOZ9_w5mkZSSoJuelWs6nB4stKCaj8lXIanf58DkNjze2Ocd9irMxIz4HUzXcYWyjDa1-0isrR_gQiOWOaCzuIFWPcVrin4Mli-niDnPYcTnNkP4ab1jmTztYb74x2seE9d5voFc7suPLzmqGuFya8fx42A-i7EP-bXM1D5_fWiTAGWbappILLOJzs3ohcO0jBe1NkkxrfVMIJD5m7CVzmOQ5ydhyWHgSNRueFPE2Bz6wqmfnU7bzrAK8DOH1NiA1LMogV7iGBlu4egfKgs7B1m7u1x1YCG5OcBHbm_67VkxxhO95IfwYYEMawg-uxXYUbAx-L_vKnIpyKq1pSaKm-hsvGMm8MvxSviOD2xkxBHElb1rZ6tkWRdR60tSRELtad5F5ItHHZe27tq-XMFZAB2UwxtvwwTZ5wD2C5PmxREMk7XF_YFR8vdVdEqfaLelx6NoeARXQ5Hm7kdiPT33RAVtyvaXuOgAPt4afpmSb0_iW9VpyU7HnHStXoKdw8gWjW-BnjJ7ZDsqy-KFTRRlj8ikcR0pAFafz-GXmiyn7EVc7rfTFYNN7c91SADAhBg2Gtk1c5EGVLCa9uocAYan6MTkzb-bOorPCI-zE-rztEp9L2Lm1TfyUNEFaIRHuG7GAwIsKvnXBwJlwlJA5McCehhqc1V6-1kqcGFSenXBvzRdXr4Wh-hAqT5gZyG0bRQIVjAiITYCbeQtahw3ioGkxUwoTq0XRZPvy0_B71b76QSbEpnVYL0b5oaBia-4qy7tJpaRkhS6pDY3bjxgWt6yPzk6sTWr_Gbt-nMgxCpkHFYI_KMSI4XvOOcPkNGgVK7X278xisXj3cx72rSaBhpRlNKkEtqsIxDTHOWNk9NZooClcTlTrHPPjo0D_3nLMdw94z5V-x0qsRsk3YrUzfIMUPeBV5ctPo5shpr4ycmLPhrnbT5U4FeaZ27AdgFDuhAeXb6AYPJ6VfOVcyZUibLQ50l2XtxjTP0uiM3fMt_7TiyYjQL_3cBv-vQRViXQyI7_7glU0b9EK-SEfl_6f9NRqWiLabP_bb0ZIrGItQ2DxEJpUmhRAErRlq0a5rlnt-We644LT-i8dvBybsnbbkwV8mqRQrAWCFHrCzPALskgmzRpuUaVBXtBhPKkUKnpwav-TseWr-Pog2iLNNfBD37EAQMPsnslLhuNeZ7-bnNJCerPiiCuNJaPd2Iy71quHCOznW7PDajSa4XvFv3GPeT_WluUD_hPRnDxFR0pFX266UJcNcNmP62ogUtfIByf5dyuwvQAFBk0Tplox1U08fKo8KabYWYVzEYQ9XiLXWgq1OSLdc9teuPP5_aFTL2VjvLgIJ8E8H8xhbk9qnPFvng0e-twGn_rKbEqIceWJ7wVqJSTMAqjJy8nBDelgPWCJVVXQ3NAgGlcAUHO0wv-9YGagfklvXrDfMrjKF8Y9YVqmg8a6FFFmCykZSmiCl-MwjHhxO9c1SlT4pDN3c5dt-V9lYGkGv-K1j392wPZOxnoH92AhX1k-lJirRrgMSj6TgQuUQynaxTL01zwD1ilNuAa6966U-KP7V5peoq-4oENlEhhRdW8HjoHl1jQlP00_7K2f5TFmzKyNPHxq7londzlTpautQp59102404xO7CRjTq4hncweSopzlR0CsSQZmy9ZDwbfOIQczBwY1lwfLvlEflZAzRgDPu-CcXTLivLNmHlXKKBKHQd35sxO3KDmgscSZuO61DvEE3SvWK_XYA46vSSHa4OSWZG1V7IDJHddPQJ65kWHbHfxCVhPY_yH52_jtgjOTUBRt_3mMFizWGenQY-tpWoe7Fy-Ex9vzXGDPi-D9QlmluqCZ1Oj0hv5LJbv65AAh5sLSR2lb_Gs1lEI8_ZMoypadUWUTAdiIQqx4nEDn6Yf9oSWVGxiPxgPlFcn_Upkf0MZqookTf7ZVlpgon2lA5MRP9O3-V9EdU8WsAfw2VV3UlygpAOHs0UQA0C691CY3PHjckcRonszezCd2JPmPlIdm-gh2aubk2IUE9NMaVdgRtc-8ysv7GHnGCvmp8FKDS5f4XytuXO_hEOOn5T8Ug-iENW4WOkIJ0jbtmz15NITlrVbPAnGhMSW0y6BfsSn-b9kloFDCA-Mbb4S-IMY8v80M4OkuhV83WPHsXl3yFEiokO02MphzPvhjoixb4OhnsTKG9vUyw4E4wJgGlxGSczNdkBcqiRISJviiRAxnpPQ4XEQ-RkuOjs1AB_cegluyWezH5LovF4S598IJYOAi1oL6Txe0gw4kJVhBmG2hKBscezt30Deu26Qx3Etk9cN3XMirfot7dBoyNZS1Jj2fpsoMbPpxwLhsIIWbWUuN2nKDqDrK199AzGD8-lki0LZh0QOxBcqbtgDDMWp_BpUtutFWL6TZm8DObtIbl4fGy12y1t4kaN-E9mLpzXQXOVjrUbP_tQlNZk7p5YOTeV_NnpunzpNpAbHXXQIukQ1xOlGvHFZ6CRFC-Yknx72aizrZf6aZH_gy9pWKmOhdqWFQ2PJLqGVLIo3mS4yoFC04AivGXdAjdrZB7odufujij9dspXWBO4Tv-dzL29O-j6p-eT6y0HuKachlFpLKpJxzsjIU1DjX0P_0pSVhemfpZDh_nV9eC2w7XNm6logVPIolXiQeypd70LOKnb-wGwxRNlcus6BUSF_BIlplV1S0TTne5PgYU3OCuOPJAwSEW0Q0lkm3lTU8nLyiQd25bV5UeD2wPvxACIPM7_cUyoGkSQ96Ip-NM607-e912AEVR93ZIVK-UHcu8MxSwWCykNaokgs4mEVK5QkY=",
                "summary": [],
            },
            {
                "id": "msg_03ec491c5caae029006ad0239e411c87d0bfb3c2c04bc6cd9c",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "I need `shop/validation.py` next to see how it treats zero, negative, and "
                        "non-integer quantities.",
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_03ec491c5caae029006ad0239e978c87d0a3ad2d7a8026ad6f",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":120}',
                "call_id": "call_6JuM80yE9acKhYmgxQKWll9Z",
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
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 214,
            "output_tokens_details": {"reasoning_tokens": 150},
            "total_tokens": 523,
        },
    },
    expected_litellm_response={
        **MESSAGES_BRIDGE_RESPONSE,
        "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDo0ZmY4OWU3OTgyMGFkZDVjMmM1ZmZlM2MyODk4ZDNkOGEzZDczY2IxY2ExN2JmZGMxYTRmNTRlZmFlNzhkMTIzO3Jlc3BvbnNlX2lkOnJlc3BfMDNlYzQ5MWM1Y2FhZTAyOTAwNmFkMDIzOWMwODM0ODdkMDk5YTIxZDYwYmU2YjVmZGU=",
        "model": "openai/gpt-6-luna",
        "usage": {"input_tokens": 309, "output_tokens": 214},
        "content": [
            {
                "type": "redacted_thinking",
                "data": "litellm_encrypted_reasoning:gAAAAABq0COfZvmrZboi5Ffbs6U_yMF1jaQmGVq5BnqydEOn98Tt_afVrM4H653kYvwFlbyesZJUOa4MLSEHMVc5ZNHlI88WpsGB2C0cdOmXO-S8Xz2wv2TYaf-d7ux1ejKPGkbop3sk2yS2XxQahiybVv5r_opD4YkEVxPoKhnVTfNF0bcglJj1JQbXYSE5qNyAVFeO6kkYN_sRP628gc_-txGgIgJbLn-uTsZ_3aQZ8TmejJJaiXLcauCUyCnYHtw0BncdwxaRiDsAUhrL_HuwLgGc3KHP0mc_IjapNLhVAcqeihSO4rmidsYX6Y7bjb4yhqPqE0-NnPZRO8BOcpIb-qU89htZHrkaSA_MrZj2FD4fk8_btQHTF9C9hM-UjRUJxqMtyuOyNMZdWjBgErNoCzslro-Sty8pWlIYRjzOlQQHZLBrC3EuKvOTEBNNjWXeRCGwZTZR6C5SyE0mj11YCwi67QeYfp8ujvrziJq1Bt7u78vnlPY8bMd0AUz0SURgP2jnlreu11EMWZuR-u6ZMLqyu1HeunBMs82y-5ef37g3evBV7Ym6AjwsiP0SqTM3KeaY5HMkgaZi9G2gwO2g8ZkLsJpy-Bk01KwNYD7QXAxrUXziEAzby_AqAOOZ9_w5mkZSSoJuelWs6nB4stKCaj8lXIanf58DkNjze2Ocd9irMxIz4HUzXcYWyjDa1-0isrR_gQiOWOaCzuIFWPcVrin4Mli-niDnPYcTnNkP4ab1jmTztYb74x2seE9d5voFc7suPLzmqGuFya8fx42A-i7EP-bXM1D5_fWiTAGWbappILLOJzs3ohcO0jBe1NkkxrfVMIJD5m7CVzmOQ5ydhyWHgSNRueFPE2Bz6wqmfnU7bzrAK8DOH1NiA1LMogV7iGBlu4egfKgs7B1m7u1x1YCG5OcBHbm_67VkxxhO95IfwYYEMawg-uxXYUbAx-L_vKnIpyKq1pSaKm-hsvGMm8MvxSviOD2xkxBHElb1rZ6tkWRdR60tSRELtad5F5ItHHZe27tq-XMFZAB2UwxtvwwTZ5wD2C5PmxREMk7XF_YFR8vdVdEqfaLelx6NoeARXQ5Hm7kdiPT33RAVtyvaXuOgAPt4afpmSb0_iW9VpyU7HnHStXoKdw8gWjW-BnjJ7ZDsqy-KFTRRlj8ikcR0pAFafz-GXmiyn7EVc7rfTFYNN7c91SADAhBg2Gtk1c5EGVLCa9uocAYan6MTkzb-bOorPCI-zE-rztEp9L2Lm1TfyUNEFaIRHuG7GAwIsKvnXBwJlwlJA5McCehhqc1V6-1kqcGFSenXBvzRdXr4Wh-hAqT5gZyG0bRQIVjAiITYCbeQtahw3ioGkxUwoTq0XRZPvy0_B71b76QSbEpnVYL0b5oaBia-4qy7tJpaRkhS6pDY3bjxgWt6yPzk6sTWr_Gbt-nMgxCpkHFYI_KMSI4XvOOcPkNGgVK7X278xisXj3cx72rSaBhpRlNKkEtqsIxDTHOWNk9NZooClcTlTrHPPjo0D_3nLMdw94z5V-x0qsRsk3YrUzfIMUPeBV5ctPo5shpr4ycmLPhrnbT5U4FeaZ27AdgFDuhAeXb6AYPJ6VfOVcyZUibLQ50l2XtxjTP0uiM3fMt_7TiyYjQL_3cBv-vQRViXQyI7_7glU0b9EK-SEfl_6f9NRqWiLabP_bb0ZIrGItQ2DxEJpUmhRAErRlq0a5rlnt-We644LT-i8dvBybsnbbkwV8mqRQrAWCFHrCzPALskgmzRpuUaVBXtBhPKkUKnpwav-TseWr-Pog2iLNNfBD37EAQMPsnslLhuNeZ7-bnNJCerPiiCuNJaPd2Iy71quHCOznW7PDajSa4XvFv3GPeT_WluUD_hPRnDxFR0pFX266UJcNcNmP62ogUtfIByf5dyuwvQAFBk0Tplox1U08fKo8KabYWYVzEYQ9XiLXWgq1OSLdc9teuPP5_aFTL2VjvLgIJ8E8H8xhbk9qnPFvng0e-twGn_rKbEqIceWJ7wVqJSTMAqjJy8nBDelgPWCJVVXQ3NAgGlcAUHO0wv-9YGagfklvXrDfMrjKF8Y9YVqmg8a6FFFmCykZSmiCl-MwjHhxO9c1SlT4pDN3c5dt-V9lYGkGv-K1j392wPZOxnoH92AhX1k-lJirRrgMSj6TgQuUQynaxTL01zwD1ilNuAa6966U-KP7V5peoq-4oENlEhhRdW8HjoHl1jQlP00_7K2f5TFmzKyNPHxq7londzlTpautQp59102404xO7CRjTq4hncweSopzlR0CsSQZmy9ZDwbfOIQczBwY1lwfLvlEflZAzRgDPu-CcXTLivLNmHlXKKBKHQd35sxO3KDmgscSZuO61DvEE3SvWK_XYA46vSSHa4OSWZG1V7IDJHddPQJ65kWHbHfxCVhPY_yH52_jtgjOTUBRt_3mMFizWGenQY-tpWoe7Fy-Ex9vzXGDPi-D9QlmluqCZ1Oj0hv5LJbv65AAh5sLSR2lb_Gs1lEI8_ZMoypadUWUTAdiIQqx4nEDn6Yf9oSWVGxiPxgPlFcn_Upkf0MZqookTf7ZVlpgon2lA5MRP9O3-V9EdU8WsAfw2VV3UlygpAOHs0UQA0C691CY3PHjckcRonszezCd2JPmPlIdm-gh2aubk2IUE9NMaVdgRtc-8ysv7GHnGCvmp8FKDS5f4XytuXO_hEOOn5T8Ug-iENW4WOkIJ0jbtmz15NITlrVbPAnGhMSW0y6BfsSn-b9kloFDCA-Mbb4S-IMY8v80M4OkuhV83WPHsXl3yFEiokO02MphzPvhjoixb4OhnsTKG9vUyw4E4wJgGlxGSczNdkBcqiRISJviiRAxnpPQ4XEQ-RkuOjs1AB_cegluyWezH5LovF4S598IJYOAi1oL6Txe0gw4kJVhBmG2hKBscezt30Deu26Qx3Etk9cN3XMirfot7dBoyNZS1Jj2fpsoMbPpxwLhsIIWbWUuN2nKDqDrK199AzGD8-lki0LZh0QOxBcqbtgDDMWp_BpUtutFWL6TZm8DObtIbl4fGy12y1t4kaN-E9mLpzXQXOVjrUbP_tQlNZk7p5YOTeV_NnpunzpNpAbHXXQIukQ1xOlGvHFZ6CRFC-Yknx72aizrZf6aZH_gy9pWKmOhdqWFQ2PJLqGVLIo3mS4yoFC04AivGXdAjdrZB7odufujij9dspXWBO4Tv-dzL29O-j6p-eT6y0HuKachlFpLKpJxzsjIU1DjX0P_0pSVhemfpZDh_nV9eC2w7XNm6logVPIolXiQeypd70LOKnb-wGwxRNlcus6BUSF_BIlplV1S0TTne5PgYU3OCuOPJAwSEW0Q0lkm3lTU8nLyiQd25bV5UeD2wPvxACIPM7_cUyoGkSQ96Ip-NM607-e912AEVR93ZIVK-UHcu8MxSwWCykNaokgs4mEVK5QkY=",
            },
            {
                "type": "text",
                "text": "I need `shop/validation.py` next to see how it treats zero, negative, and "
                "non-integer quantities.",
            },
            {
                "type": "tool_use",
                "id": "call_6JuM80yE9acKhYmgxQKWll9Z",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 120},
            },
        ],
    },
)

GPT_6_1_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "openai/gpt-6.1-sol"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={**OPENAI_RESPONSES_REQUEST_FROM_MESSAGES, "model": "gpt-6.1-sol"},
    mock_provider_response={
        **OPENAI_RESPONSES_RESPONSE,
        "model": "gpt-6.1-sol",
        "id": "resp_0af67cb78bead2c6006ad023a7273487d0b1768b77f6956b73",
        "created_at": 1792025511,
        "completed_at": 1792025514,
        "instructions": (
            "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
            "Explain which file you need next, then call a read-only tool. Never change files. Use the "
            "messages client for this project review. Consider zero, negative and non-integer quantities "
            "before choosing the next file to inspect."
        ),
        "output": [
            {
                "id": "rs_0af67cb78bead2c6006ad023a85d5c87d0810de0f56bf80208",
                "type": "reasoning",
                "content": [],
                "encrypted_content": "gAAAAABq0COqmCc2HPj-8NSt1bzfstH4-YXiTtNvZLyVMciprSqFmNbPXwcPXWrps8lsryqoAY5MuRQ_pO9yNIsCVBRlHXitSdObtJRB5Ld7TGQnbUekIsl2TNIFPsnnc8PYSdVrJlcG3FZmykyeSVHblM5F9SaPzHFzOfvCdYQtN52yt2lPhjYailjRidMwkl4KDGQ1BlcR3U29KKxo0020AqMMvsAht5kx5aAFaKfaIum5jctQrjgL8Y99h6sV3gc3LwI0W6ZR2IIqs6e6ADrVIGdVmSpn2GZTHmz6vUFaROU2-UTbVL7QrCy0BUg5CEqjnZ563lqsEYuWOxTVK0RXCPonWwmhfSllKffjLxtGuaq4dtl6v3lGIt-BW3-gJ1WqhtfTDV5VWTAliJhx09kgGhXz7VuxQRQMBSy--M9nBhdNJ1KfkbC0YDUcq5LT_cKh-KZU_-jNWCSQa7hvNJP-aCQ85oL-av-9fT5Wb5ovLUzELiBDooQLH8SalXQgFo-GX_wvSF-IH7byXwCka0F9fyIW6FjmRkirX8h_WF7pLDJeZDO5_8BWbxGOxDI6hEURHW9HLGG-e8S_lp1cVc7i37r-3KNqOwVlnlxEn2hj9TmDvTxRk4yv71AoSuwF8j3Wsh-im1HgmsNf9sdKZIZUTJTdvGCNuf54xXedUEdfoInFspUuiLXj_5KwWpckwJD0y1Fs8oEfTq2Y0Bn3LdBvwrAGgijcwnDsj6nRa2L9oMLrbHROG3xWIuTC6JyKIlVo8o7cGsM6XdxV6P9f3Wc-UgScNzc-fleF3oWzVq2gENtufLiwn9y3z1k_Hb62pbgIV6dmhLepqVH7WAyHajYt9GyxHB_FZbni-_rujbci3-3Jcj451W8PFg0TjhlyUzETBQwv6y6-zORZcbD7TqKe6XnUUGxHEMbPKBCmiIxOIUuOHpNOEjmdQpgOuPR16k_j0YjoA1gJJovIqwB1g6w44nebjzzsky0uRWmIhqeNaIFPPtIUDr05z1ocxgFeHJxum5YAh4d46vUY18dpffRyghwrd1rvIxk2EWW2rmruR4nEDTT405t3RX4kjnhGHuyShSLM_jinQm5LcpLDXUWgRnilpB1RVB8eZUSNYePAhgDRbacr8IC4AvQ7X8xUtcglbDJmMYQZQRFrkGfVoMUcyAkEJamLjccYDFrO5rxUFfyUHmGVxFUY85NW9XCwAFOx7h3606gjN9KVKQPMAg-dUnXbfhc4PCY3iRaS0djErOEugRN6f7iniy8td2YRI2WhPD5O1-qsvTDfiIhymTFq2L0v4f3UEeIa2wnfq4EdnAgla4D14w7T1D1Wg4SiP-dnF0_vKy8md5vlQjpbPGbYJkzB9DBNwJ8qR-7s1jCtZj2fViHuIGwTReI3OXUoeao4FHC8pndoAZbEpbTmax7BEYw7JqlUhIXWF2x8NzeYr-sjBbo1kbzKbtnidxOwrngjU428TzTPEjr9WA9VpmxXNoSOPL7I_qsN8ulnGegsnxQOEIarRK9CE7C8AwtMRMi8lClAFaMwMAoD8jWMNCSEa6mtyv08AikNCt3ij6cW2JszTRO3Qu4lRTsSnta41Qyk16Zc4xEPOs58PhY90t_atHYUWRpydUerE2nvRnPF9eTbQvSe8-iZ0BqkXqdlnVAAIzwOIfVZRua0p8eU3iPlzKteg9F02Jb5HkOsvi2Y4M-ZS5i0Tg4iOeuKHkeiFdZs6EMytIwz6lPGKLhYii6HW0Oe0KU_WQrIVNdXMcs5EWZGqMIdEhf09ZpO3ONcWIYswjgng6W8WMqwplHyHMcjnhzaVbIhs6rsVgTwGJawtbmD5q6aYnJCRq4zI68wzKOe7d6av8jwMGZS-dd8cdHj_XyShws1cWLhLJCaGW_vHKA0i-D1WTDeKSev31IQKCXBSRMB5URPkHu6dNDm1UyDiLGlJBx6iuKTizGeGhb4o2SQ1y1ISKyQRhKuNih0XgY40hn4x-6Scqu6_thPey-4F32BeTnbzNLGQ0wXE--dcIVRbAZMEB7wn7V4mRczG4gTG9ZnqarfZnHMPVUTZ0AxxZ2d_ifef1wQv9aRyKNRVLiBxvSqJtu7NnDMZT_U8mXCc_OIREi9wn-6esDbmxYYeBpNPR6Oq0fbsBbr--A_nVUz99MAAvkSnLhV3zdAHmMgjqNJNCmr2kPtGyle-GqdBx0LYQh4mtc5FdqLZUz9dz6gv_ERua952na3S1UnJr__8zVwofLEPGjiC5_g1nQuVCRE3QVSufMmw4gQL5kdUSuZGEm3RN__6DwAIS_X-E-aBv3utOMs92JWGandVEze8K7p4wQr3gSL5zlepjWHeYPZjiXCnGY=",
                "summary": [],
            },
            {
                "id": "msg_0af67cb78bead2c6006ad023a9b1e487d09ad5bb0d5cc96bde",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "I’ll read `shop/validation.py` first to check how it handles zero, "
                        "negative, and non-integer quantities.",
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0af67cb78bead2c6006ad023aa22c487d0b76512608a221ae3",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_H6dablHqZFScV63Y29mQXJvf",
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
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 140,
            "output_tokens_details": {"reasoning_tokens": 75},
            "total_tokens": 449,
        },
    },
    expected_litellm_response={
        **MESSAGES_BRIDGE_RESPONSE,
        "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDphMTYxN2NmYzNlODRjNzYxNDdjNTgxMmFhMjIwNGU0MDczOTM0M2YyMTg4YzQxNzIxMzRlOTc1ZTQ0YzQ3MjgyO3Jlc3BvbnNlX2lkOnJlc3BfMGFmNjdjYjc4YmVhZDJjNjAwNmFkMDIzYTcyNzM0ODdkMGIxNzY4Yjc3ZjY5NTZiNzM=",
        "model": "openai/gpt-6.1-sol",
        "usage": {"input_tokens": 309, "output_tokens": 140},
        "content": [
            {
                "type": "redacted_thinking",
                "data": "litellm_encrypted_reasoning:gAAAAABq0COqmCc2HPj-8NSt1bzfstH4-YXiTtNvZLyVMciprSqFmNbPXwcPXWrps8lsryqoAY5MuRQ_pO9yNIsCVBRlHXitSdObtJRB5Ld7TGQnbUekIsl2TNIFPsnnc8PYSdVrJlcG3FZmykyeSVHblM5F9SaPzHFzOfvCdYQtN52yt2lPhjYailjRidMwkl4KDGQ1BlcR3U29KKxo0020AqMMvsAht5kx5aAFaKfaIum5jctQrjgL8Y99h6sV3gc3LwI0W6ZR2IIqs6e6ADrVIGdVmSpn2GZTHmz6vUFaROU2-UTbVL7QrCy0BUg5CEqjnZ563lqsEYuWOxTVK0RXCPonWwmhfSllKffjLxtGuaq4dtl6v3lGIt-BW3-gJ1WqhtfTDV5VWTAliJhx09kgGhXz7VuxQRQMBSy--M9nBhdNJ1KfkbC0YDUcq5LT_cKh-KZU_-jNWCSQa7hvNJP-aCQ85oL-av-9fT5Wb5ovLUzELiBDooQLH8SalXQgFo-GX_wvSF-IH7byXwCka0F9fyIW6FjmRkirX8h_WF7pLDJeZDO5_8BWbxGOxDI6hEURHW9HLGG-e8S_lp1cVc7i37r-3KNqOwVlnlxEn2hj9TmDvTxRk4yv71AoSuwF8j3Wsh-im1HgmsNf9sdKZIZUTJTdvGCNuf54xXedUEdfoInFspUuiLXj_5KwWpckwJD0y1Fs8oEfTq2Y0Bn3LdBvwrAGgijcwnDsj6nRa2L9oMLrbHROG3xWIuTC6JyKIlVo8o7cGsM6XdxV6P9f3Wc-UgScNzc-fleF3oWzVq2gENtufLiwn9y3z1k_Hb62pbgIV6dmhLepqVH7WAyHajYt9GyxHB_FZbni-_rujbci3-3Jcj451W8PFg0TjhlyUzETBQwv6y6-zORZcbD7TqKe6XnUUGxHEMbPKBCmiIxOIUuOHpNOEjmdQpgOuPR16k_j0YjoA1gJJovIqwB1g6w44nebjzzsky0uRWmIhqeNaIFPPtIUDr05z1ocxgFeHJxum5YAh4d46vUY18dpffRyghwrd1rvIxk2EWW2rmruR4nEDTT405t3RX4kjnhGHuyShSLM_jinQm5LcpLDXUWgRnilpB1RVB8eZUSNYePAhgDRbacr8IC4AvQ7X8xUtcglbDJmMYQZQRFrkGfVoMUcyAkEJamLjccYDFrO5rxUFfyUHmGVxFUY85NW9XCwAFOx7h3606gjN9KVKQPMAg-dUnXbfhc4PCY3iRaS0djErOEugRN6f7iniy8td2YRI2WhPD5O1-qsvTDfiIhymTFq2L0v4f3UEeIa2wnfq4EdnAgla4D14w7T1D1Wg4SiP-dnF0_vKy8md5vlQjpbPGbYJkzB9DBNwJ8qR-7s1jCtZj2fViHuIGwTReI3OXUoeao4FHC8pndoAZbEpbTmax7BEYw7JqlUhIXWF2x8NzeYr-sjBbo1kbzKbtnidxOwrngjU428TzTPEjr9WA9VpmxXNoSOPL7I_qsN8ulnGegsnxQOEIarRK9CE7C8AwtMRMi8lClAFaMwMAoD8jWMNCSEa6mtyv08AikNCt3ij6cW2JszTRO3Qu4lRTsSnta41Qyk16Zc4xEPOs58PhY90t_atHYUWRpydUerE2nvRnPF9eTbQvSe8-iZ0BqkXqdlnVAAIzwOIfVZRua0p8eU3iPlzKteg9F02Jb5HkOsvi2Y4M-ZS5i0Tg4iOeuKHkeiFdZs6EMytIwz6lPGKLhYii6HW0Oe0KU_WQrIVNdXMcs5EWZGqMIdEhf09ZpO3ONcWIYswjgng6W8WMqwplHyHMcjnhzaVbIhs6rsVgTwGJawtbmD5q6aYnJCRq4zI68wzKOe7d6av8jwMGZS-dd8cdHj_XyShws1cWLhLJCaGW_vHKA0i-D1WTDeKSev31IQKCXBSRMB5URPkHu6dNDm1UyDiLGlJBx6iuKTizGeGhb4o2SQ1y1ISKyQRhKuNih0XgY40hn4x-6Scqu6_thPey-4F32BeTnbzNLGQ0wXE--dcIVRbAZMEB7wn7V4mRczG4gTG9ZnqarfZnHMPVUTZ0AxxZ2d_ifef1wQv9aRyKNRVLiBxvSqJtu7NnDMZT_U8mXCc_OIREi9wn-6esDbmxYYeBpNPR6Oq0fbsBbr--A_nVUz99MAAvkSnLhV3zdAHmMgjqNJNCmr2kPtGyle-GqdBx0LYQh4mtc5FdqLZUz9dz6gv_ERua952na3S1UnJr__8zVwofLEPGjiC5_g1nQuVCRE3QVSufMmw4gQL5kdUSuZGEm3RN__6DwAIS_X-E-aBv3utOMs92JWGandVEze8K7p4wQr3gSL5zlepjWHeYPZjiXCnGY=",
            },
            {
                "type": "text",
                "text": "I’ll read `shop/validation.py` first to check how it handles zero, "
                "negative, and non-integer quantities.",
            },
            {
                "type": "tool_use",
                "id": "call_H6dablHqZFScV63Y29mQXJvf",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 200},
            },
        ],
    },
)

GPT_6_ASTRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "openai/gpt-6-astra"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={**OPENAI_RESPONSES_REQUEST_FROM_MESSAGES, "model": "gpt-6-astra"},
    mock_provider_response={
        **OPENAI_RESPONSES_RESPONSE,
        "model": "gpt-6-astra",
        "id": "resp_0964a41f58153ac5006ad0238f9bc087d0a9e79fda375bee3c",
        "created_at": 1792025487,
        "completed_at": 1792025491,
        "instructions": (
            "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
            "Explain which file you need next, then call a read-only tool. Never change files. Use the "
            "messages client for this project review. Consider zero, negative and non-integer quantities "
            "before choosing the next file to inspect."
        ),
        "output": [
            {
                "id": "rs_0964a41f58153ac5006ad02391213887d0a1c21052bdd0c548",
                "type": "reasoning",
                "content": [],
                "encrypted_content": "gAAAAABq0COTU7GaEIVpMk00EB_rxvW069YSty7ozHW12516PLbLq0b935jCB0GevftK_27zbuIgpBhSTV-H3eiUTLx-CrGm41k8lA9CdDmFNKWBGef5fUtpfXIKNAPT3WRIkUe6ZrrMCLRa3zMhaitRJjxMf8RWRQGf0LqeomxdgNtJPU_YjiG2gapnmiUh8KSlqt30UqjcUVgjbw6isCgofeLdsvBDjy1GWfE81FTcapzObq-LzPlR6Lw838JvbjKJmBaT6B-g3UIy4ZpHCEOsLKRHdlSG97KHy84JwcRU4W1GB259oCQmWs7TRyA8BJao1vs5JrCmT0XdABCyrhUOERc7d8N0qBri6XJaVz3v1FmUObP1oz4I7Z47lqjX0a32_keE4AqjTP2evWOvjOnGsH5PVMsWknZWuvmKMYZefWuy0d7L5IUpPZ9pL4oZDgygSTH4nQ5t1ZWeYDIOKDf4oB0aTHAfgLO_IS-XgbRNMloPbeRm4acU9BI9KqGZShTVhBX34ZEc4ov7hpSmzTSUl1ukSsUmmxHqNLtKbz5CAZaYbBFM53KAyS6Bkexp6F1y8z6zKmy-LFENx-W8bZQIVaYmFAWwajro57XZE3qugHkWwCTOrP1HANsXIXXZ0wYhUdgvDqLxeRkOcaw7DAINL8K6zK71wt7Qc_j177tZjIr9uxXzEpQ3mu2p7Ji-ZBR6YRYx9sNrS55Xt5iMZYhW1X73I04pXIR5xsGO7hWZV1cGXCpWBXJ5vmMfpi4AXERmas0PAnDXEX4CRtdXflugJw7AvYDq5nY24xDYdCV9DuFF4VBzUXrWHG29OWbVzGwIegWrBEXiNIEoL4vJ05l_SqljbzKZtDg0kvY-6DnV7NwBkCYk6UgfRvJ5h8IKb7KPj7qcUhI3Zi7WXQf4yk2tH_MnM713aQ4ywDQorMJfzTcs11TABEPUwyuWt5_IHLB76c6bK97Uf8N7PkgseQ3BmPf9COF0K2JpyyUxBQOn8V5KjMK_2VTLGHQYsMA3AnX13t-_cde99USqDGqQa67C0EcbIbmHsAuctgYQP1rA_1GUB7A6oY2e1RUSoTNR3agm1FOnvAsVIjU9RW65gsI0uyz1VyIh6ZN3Ju7nqWldSDh00hIqT9bX6ctkSCZjb0Juhws_7yD7mIrykX51AfijJqCxOwI4clnITCrt2SLjuviYKxzfjCvj95107ui2fNVyWj7ksEFOX4gyA23ANATQgCWO0ITxVGnYAmZpiBSUNiJYsRUtCT6qQ1-NULhAn9UWuQJROqfJ4zACjY34kD9TtGhaLXJpKJjGw-YZ1iDUPGRG0NqPLBvI__8abyYLvA_une1qnEE49xseuR5-9uC51CpWr-Z5r40Q-Ner12_Em0p52GP52w3R14BHbuKqIDG0I1Xd47LjoNRLx7p25iivbyrcq-veIQJfKIPs4DGoAFkSyJe7z1kIuThQlusfYSaMffWXsOyWGS5jLsPSF8bRyUdfpR28wUqvbwkQFXs_5GRJLzAtBLSIJRW_SZ4mCt65B4kzkwzXJJBPvrmx0Vh2_rz-eufHW3S5S4VPsQg6nED3PB213LNzlMwwcm1KuEj0CpUXbffrKslKhEFfIFDdMchXP99i9UteCUipYdAWcXZCRYhAgMWQMdmEVVxtQRM34D-YVonMxnCQivNWYp99EhWGBfZZ-ffKV9c2WWboXUpVV_Oj8tqDIADyvRzJOd2-dGpDY2OdMzG31-ADUK-bXg68gHNSMRubpwPi7h_GBfYzPFCOFl3xaGRGE-6CH7AtzvHkZvUnrq-N5ArEv6hdhBA3qQZIh0rlyLXgNLbNz2KnP8lD7SOnGGz2sHVXoXHpW4Zu8V0coR6bxlAJ4Q5BTpcygg0q_BBWL2gr8uhxmP95XE0ANrRX8anoFoOmpS75iAsxBmkjYw8XyLWKUTj_4api4Xb1826PVTRW5x2rScJzmjVLM9AI3QiT05WSXE68NwOTO3GtYkNhDZYzBV6Aoya9f4lJwQKK76mfF8HRUen1AK4fD1X7cyI9NX9kisBrPS9YplUqF0PyFUsrL1aH6DXXmgrnsQ_V7QkMPEVwzd7NEFhHf1VbuJVzWMwzGK5OqROKxCjAY1mb34iK2r6LytJ6pX-oBIzKE7XfxcVTzzOutKdVnp_I3O9uI-BG-5Xj3degzcAdOZZbiPXNmh0kERrBgE620s6bAGiG5OGIfpS5mu4JbA1wL1nBfAC_pgdgh-62Kgo2GunkczL8Zrddxj_Xia_VBDyk6ptWYhETE4P-WWq-3wS2kUKEtuf_VPij-xi2imKOoF-NaA6rDUSoEKxqndBJsAZlAQOfX1OvkN8q44Am4RiZkzEfQEbHlde7ziC1WCVU9zsjWXe8V4EJ78CxWENhtbmF3BhUHTYaywoItp0PhYf1l7FVI7FWxXQCOX5h5MvOIzr4nf-8dMyMRR7q35aHyosc6RqOGY6OofOBcDcqFHpHAcW602H-vS2_fj7fOqCqBDmqcmEriPIBFbczBbFFZQ==",
                "summary": [],
            },
            {
                "id": "msg_0964a41f58153ac5006ad02392bf8487d0ba759f0a48ebb3c9",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "I’ll read `shop/validation.py` first to check how it handles zero, "
                        "negative, and non-integer quantities.\n",
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0964a41f58153ac5006ad0239340f887d0a46fc41789aa1a55",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_eqYF3Vli8ZUYaCLw3MAUvxKq",
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
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 152,
            "output_tokens_details": {"reasoning_tokens": 87},
            "total_tokens": 461,
        },
    },
    expected_litellm_response={
        **MESSAGES_BRIDGE_RESPONSE,
        "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDphODRkZDk3MTcwNzk0ZDlhYzQ5ODhlZWMzOTJmN2Q5YjQwN2MyNDY1YmJhZTA4MGE1ZGYzMThlOTViOTA2NzlkO3Jlc3BvbnNlX2lkOnJlc3BfMDk2NGE0MWY1ODE1M2FjNTAwNmFkMDIzOGY5YmMwODdkMGE5ZTc5ZmRhMzc1YmVlM2M=",
        "model": "openai/gpt-6-astra",
        "usage": {"input_tokens": 309, "output_tokens": 152},
        "content": [
            {
                "type": "redacted_thinking",
                "data": "litellm_encrypted_reasoning:gAAAAABq0COTU7GaEIVpMk00EB_rxvW069YSty7ozHW12516PLbLq0b935jCB0GevftK_27zbuIgpBhSTV-H3eiUTLx-CrGm41k8lA9CdDmFNKWBGef5fUtpfXIKNAPT3WRIkUe6ZrrMCLRa3zMhaitRJjxMf8RWRQGf0LqeomxdgNtJPU_YjiG2gapnmiUh8KSlqt30UqjcUVgjbw6isCgofeLdsvBDjy1GWfE81FTcapzObq-LzPlR6Lw838JvbjKJmBaT6B-g3UIy4ZpHCEOsLKRHdlSG97KHy84JwcRU4W1GB259oCQmWs7TRyA8BJao1vs5JrCmT0XdABCyrhUOERc7d8N0qBri6XJaVz3v1FmUObP1oz4I7Z47lqjX0a32_keE4AqjTP2evWOvjOnGsH5PVMsWknZWuvmKMYZefWuy0d7L5IUpPZ9pL4oZDgygSTH4nQ5t1ZWeYDIOKDf4oB0aTHAfgLO_IS-XgbRNMloPbeRm4acU9BI9KqGZShTVhBX34ZEc4ov7hpSmzTSUl1ukSsUmmxHqNLtKbz5CAZaYbBFM53KAyS6Bkexp6F1y8z6zKmy-LFENx-W8bZQIVaYmFAWwajro57XZE3qugHkWwCTOrP1HANsXIXXZ0wYhUdgvDqLxeRkOcaw7DAINL8K6zK71wt7Qc_j177tZjIr9uxXzEpQ3mu2p7Ji-ZBR6YRYx9sNrS55Xt5iMZYhW1X73I04pXIR5xsGO7hWZV1cGXCpWBXJ5vmMfpi4AXERmas0PAnDXEX4CRtdXflugJw7AvYDq5nY24xDYdCV9DuFF4VBzUXrWHG29OWbVzGwIegWrBEXiNIEoL4vJ05l_SqljbzKZtDg0kvY-6DnV7NwBkCYk6UgfRvJ5h8IKb7KPj7qcUhI3Zi7WXQf4yk2tH_MnM713aQ4ywDQorMJfzTcs11TABEPUwyuWt5_IHLB76c6bK97Uf8N7PkgseQ3BmPf9COF0K2JpyyUxBQOn8V5KjMK_2VTLGHQYsMA3AnX13t-_cde99USqDGqQa67C0EcbIbmHsAuctgYQP1rA_1GUB7A6oY2e1RUSoTNR3agm1FOnvAsVIjU9RW65gsI0uyz1VyIh6ZN3Ju7nqWldSDh00hIqT9bX6ctkSCZjb0Juhws_7yD7mIrykX51AfijJqCxOwI4clnITCrt2SLjuviYKxzfjCvj95107ui2fNVyWj7ksEFOX4gyA23ANATQgCWO0ITxVGnYAmZpiBSUNiJYsRUtCT6qQ1-NULhAn9UWuQJROqfJ4zACjY34kD9TtGhaLXJpKJjGw-YZ1iDUPGRG0NqPLBvI__8abyYLvA_une1qnEE49xseuR5-9uC51CpWr-Z5r40Q-Ner12_Em0p52GP52w3R14BHbuKqIDG0I1Xd47LjoNRLx7p25iivbyrcq-veIQJfKIPs4DGoAFkSyJe7z1kIuThQlusfYSaMffWXsOyWGS5jLsPSF8bRyUdfpR28wUqvbwkQFXs_5GRJLzAtBLSIJRW_SZ4mCt65B4kzkwzXJJBPvrmx0Vh2_rz-eufHW3S5S4VPsQg6nED3PB213LNzlMwwcm1KuEj0CpUXbffrKslKhEFfIFDdMchXP99i9UteCUipYdAWcXZCRYhAgMWQMdmEVVxtQRM34D-YVonMxnCQivNWYp99EhWGBfZZ-ffKV9c2WWboXUpVV_Oj8tqDIADyvRzJOd2-dGpDY2OdMzG31-ADUK-bXg68gHNSMRubpwPi7h_GBfYzPFCOFl3xaGRGE-6CH7AtzvHkZvUnrq-N5ArEv6hdhBA3qQZIh0rlyLXgNLbNz2KnP8lD7SOnGGz2sHVXoXHpW4Zu8V0coR6bxlAJ4Q5BTpcygg0q_BBWL2gr8uhxmP95XE0ANrRX8anoFoOmpS75iAsxBmkjYw8XyLWKUTj_4api4Xb1826PVTRW5x2rScJzmjVLM9AI3QiT05WSXE68NwOTO3GtYkNhDZYzBV6Aoya9f4lJwQKK76mfF8HRUen1AK4fD1X7cyI9NX9kisBrPS9YplUqF0PyFUsrL1aH6DXXmgrnsQ_V7QkMPEVwzd7NEFhHf1VbuJVzWMwzGK5OqROKxCjAY1mb34iK2r6LytJ6pX-oBIzKE7XfxcVTzzOutKdVnp_I3O9uI-BG-5Xj3degzcAdOZZbiPXNmh0kERrBgE620s6bAGiG5OGIfpS5mu4JbA1wL1nBfAC_pgdgh-62Kgo2GunkczL8Zrddxj_Xia_VBDyk6ptWYhETE4P-WWq-3wS2kUKEtuf_VPij-xi2imKOoF-NaA6rDUSoEKxqndBJsAZlAQOfX1OvkN8q44Am4RiZkzEfQEbHlde7ziC1WCVU9zsjWXe8V4EJ78CxWENhtbmF3BhUHTYaywoItp0PhYf1l7FVI7FWxXQCOX5h5MvOIzr4nf-8dMyMRR7q35aHyosc6RqOGY6OofOBcDcqFHpHAcW602H-vS2_fj7fOqCqBDmqcmEriPIBFbczBbFFZQ==",
            },
            {
                "type": "text",
                "text": "I’ll read `shop/validation.py` first to check how it handles zero, "
                "negative, and non-integer quantities.\n",
            },
            {
                "type": "tool_use",
                "id": "call_eqYF3Vli8ZUYaCLw3MAUvxKq",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 200},
            },
        ],
    },
)

GPT_5_6_TERRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "openai/gpt-5.6-terra"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={**OPENAI_RESPONSES_REQUEST_FROM_MESSAGES, "model": "gpt-5.6-terra"},
    mock_provider_response={
        **OPENAI_RESPONSES_RESPONSE,
        "model": "gpt-5.6-terra",
        "id": "resp_08526ccab9aa653e006ad023884a1c87d09a23f6d5c61d65d3",
        "created_at": 1792025480,
        "access_programs": {"cyber": "daybreak_blue"},
        "completed_at": 1792025481,
        "instructions": (
            "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
            "Explain which file you need next, then call a read-only tool. Never change files. Use the "
            "messages client for this project review. Consider zero, negative and non-integer quantities "
            "before choosing the next file to inspect."
        ),
        "output": [
            {
                "id": "rs_08526ccab9aa653e006ad02388d60087d0909f13870c67fc8c",
                "type": "reasoning",
                "content": [],
                "encrypted_content": "gAAAAABq0COJ5vm7lH4CvuQJznQ9B0y2iQwRylfHyiFtKnR3At3jt1oCNRl_9-mCg3S5oBCgnMLM9u4EviSSArrlNlXBn1QqjilF0jg47ecCqzcLjNkSXQfvCFKM3N9QUb33bNkruIA9VTZFJb2Dh4CWd_8uP4W2lHVfLiTcp5WGy301lO_5a8hbuO2sv08S4I0Ug67I-EFi_BMZyauCNqY8Zmbf7CodUc6cDluIeFClKOnXPaM-I4SunKVAvTZz_BsvmO-hhV2ajhBia68VqMhlioEyJQR0g5ouQrY_mhhMs99sOn4-V8aIrlkDfIGz4SEwXGRd55MSGRa095JCZig1Jzm_CdublE9Dkc0EsbJB6xjrJZfAyJZVGB95xOOsWHafy0MW878IqJNivjj6XULFSHOuVwm3WqA4_L0GKv5o0eor3OYggauA4AgKx0FiTaUAlCm5tNy2qSGMzCcKWp85CrItPYZUhWgkn4lcwGhGpJ20kmOHQbOGxnMtSpaTJL6PuMC0oyXWJli9qjaaIaB5cGhvoOtkBzifPh8pSZVsRNAnO7q7S72Lk4MtpRqCVIAVeB6kXTmkqMLhLghRowcd8xemlswAOCLqnR2usZncZIbY3kGqprxWA7u9w_acwISRcQZE6rmnefcK4I7NpA2zEuVcEVJndBWZbApBZE_OZFg7doNuw9cqhk-4y3VePW6fjSA8e68gX0VslTHODd2W6o16LQuGA-Zp84T0W0NuYI0Y2zY6AEY63DW2fDbKlbvRc5Shm5sPIugD0bWsR2SRkEMgvFnYalZFlwSojKIsxw1ET55tXqjiP4TLg1-Q4vnSkIR4dxP6lubbeN-SLCY2Nq6MVW5rqnh6o_AkFZHemKqdjFNzy5lR3JYs_4bhYetICCYSqLQVp8Cy1wmUkNeleyPjayQJFAF7lhY4mN9VUJxut-H7aizH9W0QGfoIukNLJOc3gZ7svAa3jElXtTef8Qv-rmQFA_y-AMRfHZeDLzhtxQnaDmKDeiqimbREwqG1BGvQeEKWCGPba7RGNuks9yYhNvi5NyVsEYXJe5Omy-pTZ47vrRUinlR4ecl75NPz1bL-lLtYhIMtvLtuJvkJ_ieD6g7oP3qEcsS34ARmBq6ZpEkelgL9ifh8DGX9gerq-JbsmnyI-c8aG9Go--hJWXs2dKJSfBq1ABLhhW3rK3Ms5163BkdtZp67F0LPISjU8i4Kh_wSxZbqXkdJy-4oYgAqG_YGQMz5tyM3GFMj-ozDuznqivj3GcD7xRTTA2CfDTbFxT1mrTpLv_NjKsfjMOSH5KF5RLwP8fatr76F0YUKW75korD9RG0tJpbSHBq67r4u0jwyNRJFZMPsmvOohr4GaTOW3-kBfUZqVU7x2X6cjmfw50jY4EOQb1r4sJBbVkgqTcw36zPSzR46dkJHUtEZMqNgb0u8AUze_Z1Or3x0DInw6xMSAs2jzYNEiIYzJ0EJ62_ciEcWT5aYUGSZGhNy2i9dIEJiOTYfzL-b7dNzs-6MoWMHzMaPPPcp4e8BLCgDJd6ryDuuHRd7NZmEnt2jdUlKr4iqEstENijjJjdl1xolCtUMPYQahG9iYPMa2tKqXAZYTyf6_cEdv6_7kUdFd6iUuU7kOplokkOi9A-GbA0Lhhct5HP3QYxMSymUoMv_NO6xjgAImYwesVohnAaN1hNmraWuavZKj4w95byYav83k0xFB_mz3xKYjLsCtTw3rF90UsWk4KgXnoZRZv0FcesyyNAXfzizkgsI0Ndt4i8EIHhP6upFi473ZMqbWo5_RQtAiKsHBD1L5NG51Lai2caL5bnvOrHtfW_aZcLGNdFQbziTpTFwafefusBSGR0hLHc0LzKpMg-XDCwzU8AnK3yngTTKnOb9j0_avEdz5xv9xnLlp5rnHtMrYRna-5HPQ2xKIXmGzbMx2h8OQYdWTqmVayubLoc-KpW0qKYxYJRRIgo=",
                "summary": [],
            },
            {
                "id": "msg_08526ccab9aa653e006ad023892c7087d0870b4b072127659f",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "Next I need `shop/validation.py` to inspect the validation branches for "
                        "zero, negative, and non-integer quantities.",
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_08526ccab9aa653e006ad02389692887d0ab14a9eaef763aee",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_iom4KPAiRrGmsYIsEHwM2vZo",
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
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 107,
            "output_tokens_details": {"reasoning_tokens": 42},
            "total_tokens": 416,
        },
    },
    expected_litellm_response={
        **MESSAGES_BRIDGE_RESPONSE,
        "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDo0Mjg5MWEzMzRjNTlhMjhlMjQ2MjQ0YTQ4YTJjOGVlMTM5NGYyZGNhZWMxYmVjZDhmZDk2N2NmNjYxNWNmZDM5O3Jlc3BvbnNlX2lkOnJlc3BfMDg1MjZjY2FiOWFhNjUzZTAwNmFkMDIzODg0YTFjODdkMDlhMjNmNmQ1YzYxZDY1ZDM=",
        "model": "openai/gpt-5.6-terra",
        "usage": {"input_tokens": 309, "output_tokens": 107},
        "content": [
            {
                "type": "redacted_thinking",
                "data": "litellm_encrypted_reasoning:gAAAAABq0COJ5vm7lH4CvuQJznQ9B0y2iQwRylfHyiFtKnR3At3jt1oCNRl_9-mCg3S5oBCgnMLM9u4EviSSArrlNlXBn1QqjilF0jg47ecCqzcLjNkSXQfvCFKM3N9QUb33bNkruIA9VTZFJb2Dh4CWd_8uP4W2lHVfLiTcp5WGy301lO_5a8hbuO2sv08S4I0Ug67I-EFi_BMZyauCNqY8Zmbf7CodUc6cDluIeFClKOnXPaM-I4SunKVAvTZz_BsvmO-hhV2ajhBia68VqMhlioEyJQR0g5ouQrY_mhhMs99sOn4-V8aIrlkDfIGz4SEwXGRd55MSGRa095JCZig1Jzm_CdublE9Dkc0EsbJB6xjrJZfAyJZVGB95xOOsWHafy0MW878IqJNivjj6XULFSHOuVwm3WqA4_L0GKv5o0eor3OYggauA4AgKx0FiTaUAlCm5tNy2qSGMzCcKWp85CrItPYZUhWgkn4lcwGhGpJ20kmOHQbOGxnMtSpaTJL6PuMC0oyXWJli9qjaaIaB5cGhvoOtkBzifPh8pSZVsRNAnO7q7S72Lk4MtpRqCVIAVeB6kXTmkqMLhLghRowcd8xemlswAOCLqnR2usZncZIbY3kGqprxWA7u9w_acwISRcQZE6rmnefcK4I7NpA2zEuVcEVJndBWZbApBZE_OZFg7doNuw9cqhk-4y3VePW6fjSA8e68gX0VslTHODd2W6o16LQuGA-Zp84T0W0NuYI0Y2zY6AEY63DW2fDbKlbvRc5Shm5sPIugD0bWsR2SRkEMgvFnYalZFlwSojKIsxw1ET55tXqjiP4TLg1-Q4vnSkIR4dxP6lubbeN-SLCY2Nq6MVW5rqnh6o_AkFZHemKqdjFNzy5lR3JYs_4bhYetICCYSqLQVp8Cy1wmUkNeleyPjayQJFAF7lhY4mN9VUJxut-H7aizH9W0QGfoIukNLJOc3gZ7svAa3jElXtTef8Qv-rmQFA_y-AMRfHZeDLzhtxQnaDmKDeiqimbREwqG1BGvQeEKWCGPba7RGNuks9yYhNvi5NyVsEYXJe5Omy-pTZ47vrRUinlR4ecl75NPz1bL-lLtYhIMtvLtuJvkJ_ieD6g7oP3qEcsS34ARmBq6ZpEkelgL9ifh8DGX9gerq-JbsmnyI-c8aG9Go--hJWXs2dKJSfBq1ABLhhW3rK3Ms5163BkdtZp67F0LPISjU8i4Kh_wSxZbqXkdJy-4oYgAqG_YGQMz5tyM3GFMj-ozDuznqivj3GcD7xRTTA2CfDTbFxT1mrTpLv_NjKsfjMOSH5KF5RLwP8fatr76F0YUKW75korD9RG0tJpbSHBq67r4u0jwyNRJFZMPsmvOohr4GaTOW3-kBfUZqVU7x2X6cjmfw50jY4EOQb1r4sJBbVkgqTcw36zPSzR46dkJHUtEZMqNgb0u8AUze_Z1Or3x0DInw6xMSAs2jzYNEiIYzJ0EJ62_ciEcWT5aYUGSZGhNy2i9dIEJiOTYfzL-b7dNzs-6MoWMHzMaPPPcp4e8BLCgDJd6ryDuuHRd7NZmEnt2jdUlKr4iqEstENijjJjdl1xolCtUMPYQahG9iYPMa2tKqXAZYTyf6_cEdv6_7kUdFd6iUuU7kOplokkOi9A-GbA0Lhhct5HP3QYxMSymUoMv_NO6xjgAImYwesVohnAaN1hNmraWuavZKj4w95byYav83k0xFB_mz3xKYjLsCtTw3rF90UsWk4KgXnoZRZv0FcesyyNAXfzizkgsI0Ndt4i8EIHhP6upFi473ZMqbWo5_RQtAiKsHBD1L5NG51Lai2caL5bnvOrHtfW_aZcLGNdFQbziTpTFwafefusBSGR0hLHc0LzKpMg-XDCwzU8AnK3yngTTKnOb9j0_avEdz5xv9xnLlp5rnHtMrYRna-5HPQ2xKIXmGzbMx2h8OQYdWTqmVayubLoc-KpW0qKYxYJRRIgo=",
            },
            {
                "type": "text",
                "text": "Next I need `shop/validation.py` to inspect the validation branches for "
                "zero, negative, and non-integer quantities.",
            },
            {
                "type": "tool_use",
                "id": "call_iom4KPAiRrGmsYIsEHwM2vZo",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 200},
            },
        ],
    },
)
