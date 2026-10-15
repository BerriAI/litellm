from typing import Final

from integration.translation.case import TranslationTestCase
from integration.translation.messages.templates import MESSAGES_REQUEST
from integration.translation.templates.openai import OPENAI_RESPONSES_REQUEST_FROM_MESSAGES, OPENAI_RESPONSES_RESPONSE

GPT_5_6_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "openai/responses/gpt-5.6-sol",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/responses",
    expected_provider_headers={"authorization": "Bearer synthetic-openai-key", "content-type": "application/json"},
    expected_provider_request={
        "model": "gpt-5.6-sol",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hello."}]}],
        "include": ["reasoning.encrypted_content"],
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
    },
    mock_provider_response={
        "id": "resp_0879800c1455ba3d006ac3f0427c4c87d0a632b130c8c464ea",
        "object": "response",
        "created_at": 1791225922,
        "status": "completed",
        "access_programs": {"cyber": "daybreak_blue"},
        "background": False,
        "billing": {"payer": "developer"},
        "completed_at": 1791225923,
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
                "id": "msg_0879800c1455ba3d006ac3f0430f2087d0bcdc00a8eba6a1ce",
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
        "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDo1ZDQ1NmY1ODlhMDIwNzI4N2M0YzU2NmRhN2NkN2ViNjJlMzZmNGUwMTYxOTE4M2VhMzhiMzdlZGZhNDdlM2EwO3Jlc3BvbnNlX2lkOnJlc3BfMDg3OTgwMGMxNDU1YmEzZDAwNmFjM2YwNDI3YzRjODdkMGE2MzJiMTMwYzhjNDY0ZWE=",
        "type": "message",
        "role": "assistant",
        "model": "openai/responses/gpt-5.6-sol",
        "stop_sequence": None,
        "usage": {"input_tokens": 19, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

GPT_6_1_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "openai/responses/gpt-6.1-sol"},
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
        "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDo3ZGE3NDY3Njc3N2IyZjVkYmQ3MDI1MTI1ZjEzMDdlYWQ0MjM3YTgzN2QxZTJmZDhkY2FkNjhlODM4ODU1Zjg2O3Jlc3BvbnNlX2lkOnJlc3BfMGFmNjdjYjc4YmVhZDJjNjAwNmFkMDIzYTcyNzM0ODdkMGIxNzY4Yjc3ZjY5NTZiNzM=",
        "type": "message",
        "role": "assistant",
        "model": "openai/responses/gpt-6.1-sol",
        "stop_sequence": None,
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
        "stop_reason": "tool_use",
        "stop_details": None,
    },
)
