from typing import Final

from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.templates import CHAT_COMPLETIONS_REQUEST, CHAT_COMPLETIONS_RESPONSE
from integration.translation.templates.openai import (
    OPENAI_RESPONSES_REQUEST_FROM_CHAT_COMPLETIONS,
    OPENAI_RESPONSES_RESPONSE,
)

GPT_5_4_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "openai/gpt-5.4",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/chat/completions",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "model": "gpt-5.4",
        "max_completion_tokens": 64,
    },
    mock_provider_response={
        "id": "chatcmpl-EVhxnsh4umJNE0UBHCO0G3rCQjLO9",
        "object": "chat.completion",
        "created": 1791225871,
        "model": "gpt-5.4-2026-03-05",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello.", "refusal": None, "annotations": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 19,
            "completion_tokens": 5,
            "total_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
    expected_litellm_response={
        "id": "chatcmpl-EVhxnsh4umJNE0UBHCO0G3rCQjLO9",
        "created": 1791225871,
        "model": "openai/gpt-5.4",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                },
                "provider_specific_fields": {},
            }
        ],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 19,
            "total_tokens": 24,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
)


GPT_5_6_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "openai/gpt-5.6-sol",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/chat/completions",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "model": "gpt-5.6-sol",
        "max_completion_tokens": 64,
    },
    mock_provider_response={
        "id": "chatcmpl-EVhxr75iujjmwE0F2bgGMQHDX3Bq6",
        "object": "chat.completion",
        "created": 1791225875,
        "model": "gpt-5.6-sol",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!", "refusal": None, "annotations": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 19,
            "completion_tokens": 5,
            "total_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
    expected_litellm_response={
        "id": "chatcmpl-EVhxr75iujjmwE0F2bgGMQHDX3Bq6",
        "created": 1791225875,
        "model": "openai/gpt-5.6-sol",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello!",
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                },
                "provider_specific_fields": {},
            }
        ],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 19,
            "total_tokens": 24,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
)


GPT_5_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "openai/gpt-5.6-luna",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/chat/completions",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "model": "gpt-5.6-luna",
        "max_completion_tokens": 64,
    },
    mock_provider_response={
        "id": "chatcmpl-EVhxvZldBlNE3r8WZ8qJkxC92PmAr",
        "object": "chat.completion",
        "created": 1791225879,
        "model": "gpt-5.6-luna",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!", "refusal": None, "annotations": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 19,
            "completion_tokens": 5,
            "total_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
    expected_litellm_response={
        "id": "chatcmpl-EVhxvZldBlNE3r8WZ8qJkxC92PmAr",
        "created": 1791225879,
        "model": "openai/gpt-5.6-luna",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello!",
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                },
                "provider_specific_fields": {},
            }
        ],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 19,
            "total_tokens": 24,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
)

GPT_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "openai/gpt-6-luna"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={**OPENAI_RESPONSES_REQUEST_FROM_CHAT_COMPLETIONS, "model": "gpt-6-luna"},
    mock_provider_response={
        **OPENAI_RESPONSES_RESPONSE,
        "model": "gpt-6-luna",
        "id": "resp_0a2459714d575226006ad02398406087d0a2501571a0097cce",
        "created_at": 1792025496,
        "access_programs": {"cyber": "daybreak_blue"},
        "completed_at": 1792025499,
        "instructions": (
            "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
            "Explain which file you need next, then call a read-only tool. Never change files. Use the "
            "chat_completions client for this project review. Consider zero, negative and non-integer "
            "quantities before choosing the next file to inspect."
        ),
        "output": [
            {
                "id": "rs_0a2459714d575226006ad0239a42d087d08ab86bc19873f8f8",
                "type": "reasoning",
                "content": [],
                "encrypted_content": "gAAAAABq0CObTtOHdIl0AomNdpnN-jGzO6wbopH5o7XAO-1__huttxkYQhzVBAV_TQ_Y-LHedrilDsn2OYqDrSZke4WPG2uMFL0z1kvWcj1zShAuI3bzLB4637l7tqQeM41ncLLPD2gfy4gS1G4oX-vWl-cJK5apCct8b6VXdWClY9H7OmLyrtYeyeTcvacBq3cJf6OJg6zI2f4O-rPsifmcGdXwojO8JWhliGBJX_98gjGwKEdBAVJHFXP57xEZbCd9b2ISUofeSbdVfvzCnwsRs5Bx40pQf_abIM99JdNNictIr0Zim2jiIEh5hfOJhlu6vRmaLT3eJ9fRE_ao_wkSaCllTwzH7z-QUBSSXmj8SqQCp0m-WExL4SPKrTLhQteqTRDx4F833tJNJbdAe65PyO_2cChbvqmIU9BMB0dftkbfWdt3JwP8be7_aSG-c250rTCB373oI8Xf5vBQcqVhJ3hfu-1ddFjWgFdk7b7V1nlBKsKK3_EzCNCMmcoGwKFShIZRL9RxyBYiStLjUT3PA9DtBay6wKyW62mG2JJPQxknrOE0BBmcD3Kg4EbH9jgThKxv4zQkFTSl_-gsoSFFnUbhRDCcSQpirefKYf_xBWpBU55_jjHWfkhZXfOuU4eArnD6g4Lt7T3wd9SQyRh3I5xdtzNUxd4fswu99REgL_S3tfUmPohLrFGBywb4SSFbqaz7HwR8y3NN9Aet7XVdTskdfv0g3_sBKQyCf1o3-hc4lNBFxXFaEEFXMQTPZKOjfPJNP7ZKxfueiwadGg0aVV-dPCAMnMuSncc5RC17f9ohZyGncTQHT1lLJJPOiAcQPThZABVu4Ptnuhn2oXs0HGk_q6y3cWDXOZOojrFncvDGHC8dUx2KNhNKsDGVGteHay8UDQBmEnSMez2mfHdCRRcuO6R9ZBWV-7TMAMxYgJ_nnGk-n2WAEmIrnWtSQkvInKtYf3A8bB9nyJ6Beu8IL20baT1SD3mtnm4y-ozQ8uuVsuVRNAatotjX23kjzx0kXpiS9xNeD5nYmxIDpO9TAb3nx2F5KPbEYbOB-7B_yvjteQFk6EfApTmwH9xX2nOGfN42ttX063EtBK4IvSbbGWLbjmfbgHB_hdn6Iev7zJxZ53M_WyPAMWcbvx7OZijvfAznAwGyRDDFla-CZmQhiFDPq5GwVsLwSHK-XIi-nPMJfBaBBXdP03KqG6whBNA2q6ZvyAvdr5hyFab6atRocfK6eSGbhpmrY5VTDEW7P1jqmG2myIFEMtgNCw7HA9DC11RD0cRk6e2Fwl_xrmHm9G7NL06RQT5NaGJIBRRHRt9fqAj4JlWFbXzj866Ob21ApfBHvP7GLrTWQeNJf5IcJYrZbgTK0JOn0twLTf0ekZoDzrw7qu1a9OnC3rsDQsMYRNtaOkAyHOfQhcQwZmg264AIhqmy_69fUZxBuLq-aLIVSDiJUujKRqiTtjAlvdqXObg-1lnDf9IhSB4qXn0Vcraows4g9b2-gc2t6GoGwqf3Cv7fXFK4jWajNsNenMgEjHyXVTGRY5fIBLBhoRzOcuy8FuIXN6WeYj6JAbnGfnB4Z4yssWJrSiqcH4X5YUHSZDYkB3w9dsZpTDCr8kNoW6oJrnrX6tegNQJUIi6iaQHR2vxptmnvueTKbC3sj2J0ICPWhLKbjOHMOHgGoLaT_s-SiH4CjrvoZALYOWMKyP0EMghJl9mNFLt-mnexO6Egm6J9xAuglMUxFm1txFuDBAp7jq1TR6z1mLm4-ZorpVIvbJv85FS4qQmpGpY_nU1IFTwcKET4skSsJk9utzfW5nfG6A9kejsG-eJNg9yatnenrglGNDs5RmfBKvbvr_yGK1agM3n47n9So8r7G46CviCHDoyx0nvQwog83YYAQOxEd8mlziu0ZmBrEwWxSMeGyYIPVGOWr3FrutMX7ySxxN9uk7oNzZFl9O4PoZwaCCipA0KDAVXiJ0A36Zp2sqnIeecXkWJEZHxynZ6Y8mOFn4SM-JrZ_PJBvWPpPEYTTKSCUrBh1ytBzXN2zU435TJ8I1hGBiG6atVCJ3DK4UtBhzYMwkG6Yj6jhXGBsNBUBaN8-vmCcda620JebM1U4dCtLhwsW_4J2B-S9gaYIXWtV10kIO-yKW_L8QCZQAtB44pOfKwPjy_n4PHoMV4IibbNDCDclVl8cAnO7-RJH5sewDmU1M_u5llf1x6wd8nWBKh1_kBlc6jPIZ8hkVTxy6QvthAYPV5_BPGtOEhbxj8po60tBU5d-kZMEkLBFQdKjiwTOMTBBe5SsdC32vnqL7ZZEj5ZK6xiCY4jYA8ykTyE9jTtx_rdnSpPMLviBw2gbOmlOVGoUw_AX71pojMTYDuVT_mnXrid5Dd7vVaBe8-3a9p0dnxX29k5JEo48yT9DY5JqXo2Dvg5YAadq4uueww8VGOrM25V",
                "summary": [],
            },
            {
                "id": "msg_0a2459714d575226006ad0239af77c87d09bf21be68f806551",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "I’ll read `shop/validation.py` first to see how it handles zero, negative, "
                        "and non-integer quantities.",
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0a2459714d575226006ad0239b46bc87d0ad9e2b753765029b",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_zySVmQauaNPsuWHCle7BA4CX",
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
            "output_tokens": 141,
            "output_tokens_details": {"reasoning_tokens": 76},
            "total_tokens": 453,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "openai/gpt-6-luna",
        "id": "resp_0a2459714d575226006ad02398406087d0a2501571a0097cce",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I’ll read `shop/validation.py` first to see how it handles zero, "
                    "negative, and non-integer quantities.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                                "name": "read_file",
                            },
                            "id": "call_zySVmQauaNPsuWHCle7BA4CX",
                            "type": "function",
                        }
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_0a2459714d575226006ad0239a42d087d08ab86bc19873f8f8",
                            "encrypted_content": "gAAAAABq0CObTtOHdIl0AomNdpnN-jGzO6wbopH5o7XAO-1__huttxkYQhzVBAV_TQ_Y-LHedrilDsn2OYqDrSZke4WPG2uMFL0z1kvWcj1zShAuI3bzLB4637l7tqQeM41ncLLPD2gfy4gS1G4oX-vWl-cJK5apCct8b6VXdWClY9H7OmLyrtYeyeTcvacBq3cJf6OJg6zI2f4O-rPsifmcGdXwojO8JWhliGBJX_98gjGwKEdBAVJHFXP57xEZbCd9b2ISUofeSbdVfvzCnwsRs5Bx40pQf_abIM99JdNNictIr0Zim2jiIEh5hfOJhlu6vRmaLT3eJ9fRE_ao_wkSaCllTwzH7z-QUBSSXmj8SqQCp0m-WExL4SPKrTLhQteqTRDx4F833tJNJbdAe65PyO_2cChbvqmIU9BMB0dftkbfWdt3JwP8be7_aSG-c250rTCB373oI8Xf5vBQcqVhJ3hfu-1ddFjWgFdk7b7V1nlBKsKK3_EzCNCMmcoGwKFShIZRL9RxyBYiStLjUT3PA9DtBay6wKyW62mG2JJPQxknrOE0BBmcD3Kg4EbH9jgThKxv4zQkFTSl_-gsoSFFnUbhRDCcSQpirefKYf_xBWpBU55_jjHWfkhZXfOuU4eArnD6g4Lt7T3wd9SQyRh3I5xdtzNUxd4fswu99REgL_S3tfUmPohLrFGBywb4SSFbqaz7HwR8y3NN9Aet7XVdTskdfv0g3_sBKQyCf1o3-hc4lNBFxXFaEEFXMQTPZKOjfPJNP7ZKxfueiwadGg0aVV-dPCAMnMuSncc5RC17f9ohZyGncTQHT1lLJJPOiAcQPThZABVu4Ptnuhn2oXs0HGk_q6y3cWDXOZOojrFncvDGHC8dUx2KNhNKsDGVGteHay8UDQBmEnSMez2mfHdCRRcuO6R9ZBWV-7TMAMxYgJ_nnGk-n2WAEmIrnWtSQkvInKtYf3A8bB9nyJ6Beu8IL20baT1SD3mtnm4y-ozQ8uuVsuVRNAatotjX23kjzx0kXpiS9xNeD5nYmxIDpO9TAb3nx2F5KPbEYbOB-7B_yvjteQFk6EfApTmwH9xX2nOGfN42ttX063EtBK4IvSbbGWLbjmfbgHB_hdn6Iev7zJxZ53M_WyPAMWcbvx7OZijvfAznAwGyRDDFla-CZmQhiFDPq5GwVsLwSHK-XIi-nPMJfBaBBXdP03KqG6whBNA2q6ZvyAvdr5hyFab6atRocfK6eSGbhpmrY5VTDEW7P1jqmG2myIFEMtgNCw7HA9DC11RD0cRk6e2Fwl_xrmHm9G7NL06RQT5NaGJIBRRHRt9fqAj4JlWFbXzj866Ob21ApfBHvP7GLrTWQeNJf5IcJYrZbgTK0JOn0twLTf0ekZoDzrw7qu1a9OnC3rsDQsMYRNtaOkAyHOfQhcQwZmg264AIhqmy_69fUZxBuLq-aLIVSDiJUujKRqiTtjAlvdqXObg-1lnDf9IhSB4qXn0Vcraows4g9b2-gc2t6GoGwqf3Cv7fXFK4jWajNsNenMgEjHyXVTGRY5fIBLBhoRzOcuy8FuIXN6WeYj6JAbnGfnB4Z4yssWJrSiqcH4X5YUHSZDYkB3w9dsZpTDCr8kNoW6oJrnrX6tegNQJUIi6iaQHR2vxptmnvueTKbC3sj2J0ICPWhLKbjOHMOHgGoLaT_s-SiH4CjrvoZALYOWMKyP0EMghJl9mNFLt-mnexO6Egm6J9xAuglMUxFm1txFuDBAp7jq1TR6z1mLm4-ZorpVIvbJv85FS4qQmpGpY_nU1IFTwcKET4skSsJk9utzfW5nfG6A9kejsG-eJNg9yatnenrglGNDs5RmfBKvbvr_yGK1agM3n47n9So8r7G46CviCHDoyx0nvQwog83YYAQOxEd8mlziu0ZmBrEwWxSMeGyYIPVGOWr3FrutMX7ySxxN9uk7oNzZFl9O4PoZwaCCipA0KDAVXiJ0A36Zp2sqnIeecXkWJEZHxynZ6Y8mOFn4SM-JrZ_PJBvWPpPEYTTKSCUrBh1ytBzXN2zU435TJ8I1hGBiG6atVCJ3DK4UtBhzYMwkG6Yj6jhXGBsNBUBaN8-vmCcda620JebM1U4dCtLhwsW_4J2B-S9gaYIXWtV10kIO-yKW_L8QCZQAtB44pOfKwPjy_n4PHoMV4IibbNDCDclVl8cAnO7-RJH5sewDmU1M_u5llf1x6wd8nWBKh1_kBlc6jPIZ8hkVTxy6QvthAYPV5_BPGtOEhbxj8po60tBU5d-kZMEkLBFQdKjiwTOMTBBe5SsdC32vnqL7ZZEj5ZK6xiCY4jYA8ykTyE9jTtx_rdnSpPMLviBw2gbOmlOVGoUw_AX71pojMTYDuVT_mnXrid5Dd7vVaBe8-3a9p0dnxX29k5JEo48yT9DY5JqXo2Dvg5YAadq4uueww8VGOrM25V",
                            "summary": [],
                        }
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 141,
            "prompt_tokens": 312,
            "total_tokens": 453,
            "completion_tokens_details": {"reasoning_tokens": 76},
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
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "openai/gpt-6.1-sol"},
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
        "model": "openai/gpt-6.1-sol",
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

GPT_6_ASTRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "openai/gpt-6-astra"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={**OPENAI_RESPONSES_REQUEST_FROM_CHAT_COMPLETIONS, "model": "gpt-6-astra"},
    mock_provider_response={
        **OPENAI_RESPONSES_RESPONSE,
        "model": "gpt-6-astra",
        "id": "resp_0dd9c3f2f13006a2006ad0238bcfa087d0ae1a00e93e9dcfa5",
        "created_at": 1792025483,
        "completed_at": 1792025487,
        "instructions": (
            "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
            "Explain which file you need next, then call a read-only tool. Never change files. Use the "
            "chat_completions client for this project review. Consider zero, negative and non-integer "
            "quantities before choosing the next file to inspect."
        ),
        "output": [
            {
                "id": "rs_0dd9c3f2f13006a2006ad0238ce43887d08314c8e2f0619171",
                "type": "reasoning",
                "content": [],
                "encrypted_content": "gAAAAABq0COPW81YCmA7VJWEaMeFZwZsixHXQg1hUZ8RoBpCteWdhDHmuIxYPWc-lwYTsH28yN7Hjlrgvtmvn5jtVILJozkIcXbLvyXzNi3hswFvpK0NOfkzIeepbxIkFPbEWmAHnGy12UpwFSaQHWNnq3n3_yx1QHg-6L84uhXb_gVLWbQpPfJSUHvqfymkwAH1GoqMKv5GRU1vQQzxs5O8WTfgNLyX0riq6mbYoO8jjRd-QzqkHf5w0fbNsQ0a1npvML0f6vzPdDsBKUrmDiwrd5HjM5tn62CI5ApllJFemVCTa_pXf0d7ws71b-g8d9jIVFtSoDe_oIZ6ZDTiGSRrdgRW3HzZG-hOS85nqwzYcUbXnZ_2dtZUelx1bFpGgH3udKhXIJ1YYXBoTsbYubHskPIWf0DE_rnbLUogslDawx9SBcYJ5dXTD0ipUYtIsW830XTVV1cOl2sFtp4SUrtJiV55G_r7sAxxUyQvF0-OlByS-wEbEvaQfylbFkHZUOcOs69DmgcGP1a_hdD5ZiiY3k1KG63JeIYFzDo1UomGzDFpbbihgq73-FD1JgHARe1vrwJ_0Y3BhhZjJ8UmeLGqyy4C6NQAmaLEg2iK4X-bzItSD-esAhENq-ZmUUKAY74SsGCdPK9srakfVz05rGN0_XKLkkoyW74UzWY89hkYAzrQOw5zGMghVKAioFcpyjX7JVz6fEdL4OWoaTknzhjnQKv1DEq2v8RsZVg3wWuU0t1WB562OQEIaScCVH4knjEtgCB8erZyStk65uwlXK5hQ6rof6Z4-jpd4xp89rAqNcX5OxnXG1SfST50s4C_YddfNe7Y0NQ9_yORz6TChuP3weUXnqlMOWHe4bMSTRLDITfLToWr4KwgajtQ_EmvCrbQWX5BHwSMHOCRKj92eSThzJ2EStpCmV-fihb6onbQk-x3i8pB0J3hkhfRzzsd2vEpy-E4RG9AgKhntN_gwP0Uez_hgEtfv9vipI8qVe-ivu168qU5N8siLollCok_DmIpnjsupxjNlifsv6LhKr97Y7S3TDnpdRL17Mt8lXJpuChunCDC7URHf990u4_j6tuQ9WLGKAq7F8lPoh8HzVdM5ozE-proWaEzFVd035l7q0dQxhQvw0JkhhcRMkjnmIFnTCQ3vAhHc2WCsy-gR5YI472RZoxU6x9q1KOVpTonw6ioUxZoB_QxZGxUT3FPdfMU6pOS0DZ0Ws1qL-49UOSuIleiEAhmX3E_lG_QD5dcxRbCOFHo3-JML3pTrXU49GZi7ZPqzmNcRSK7nxq1751e-YmBt_KYkqLI0sRM0Wwz1X1XTSv7S97iefaAj6xXhnlyc3hi5ofzj8iTGGF-mWYrC-F6GhCizpM91ePu6uugP-FWdtaQ5YXO-9Kq7mFwGB8NE-vftRZBr6p4TbGgXgrZk9rwMpgyC60WACV24UpJhaaBfnPgojE_nuskalMjAy41VA0CyMDCRBJ29MPE9R-dVMNIld9hfBSz2kMCaDXGw83fWPNAcUPQ2fvMBrBhkSLgTRhX7CDElb4foRJS6vTnTgW-es6mk9Ja-3-O013xo09WkqPyYieF1OIWyiALlUVqDtFYdWbUrIHUVV69D03XM3zIgy05HMI5uj46gT8l_X_nDnUI8zolKjB0STnTuzwwK4xvcwl7ank70dIvKatv62MrI-ZEp7QD00ZHeos5fh2X0YxNCsavuKF6UiT_hjWo-v9Sb2qJYKwiv_Bz_pqMqrQyQP-S35FkmWLzfu_ipbNxqE9z_u4WwvPUSdyAdMsJtBc41MJFtYAsBZc5jU0o7Sd2uiVQaUG51Q19jwq6MU68DxcTWukmdB4rQMd4EHYjtNJYr_zGGqzwJAO7gmKeAPxqbwbI5xg_JYGqdVfKBWKdZCCGljYW8DoimGJugt8OJvFFVu3wAOG9KRhaNtdmDnjYd24jxpqWtn5BZSNgwtQP1nh07a7aa-jsJmNpDxu9WH9fax7FXBjz8L9UlG47kBNTpQzKuW_8GRbe2Xvx5jJ0Rh9EFoW1XOiDap3s--MBUDwcRpQvxcU7Ip9gA7JNlsQJoglTvgZiQArlVlbkx7I6UOMHGzJ68UtuFyg3Z6eyAAXTBauozf2VCtbucac_f6kOC80xwZBxfTKG8iXlKOqa3_h_Q66kzPP6h6Ni4-FW-Z5b2aVCUGP_TnG4PkqS-ciRQz2CAdXcHkgy1bXbe-qWXpPBSYs0jnkBygX_v6sa3k8k2wXFenRz0nZ0D684585Axe5Kz5gvW5sR4n2LiN_x95cqDvXCLqzJ-T6kawaRBEoI2Ym3uw7tfOx3o1qyLAvZ9eX4c-dI1kVrC6WzJeeFy2OPOjY=",
                "summary": [],
            },
            {
                "id": "msg_0dd9c3f2f13006a2006ad0238e66f487d0b8321be613a40623",
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
                "id": "fc_0dd9c3f2f13006a2006ad0238ee75087d0aeb5c9d34c955fb5",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_EEr3oYtEWt2tw0HIjEYCvhL4",
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
            "output_tokens": 135,
            "output_tokens_details": {"reasoning_tokens": 70},
            "total_tokens": 447,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "openai/gpt-6-astra",
        "id": "resp_0dd9c3f2f13006a2006ad0238bcfa087d0ae1a00e93e9dcfa5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I’ll read `shop/validation.py` first to check how it handles zero, "
                    "negative, and non-integer quantities.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                                "name": "read_file",
                            },
                            "id": "call_EEr3oYtEWt2tw0HIjEYCvhL4",
                            "type": "function",
                        }
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_0dd9c3f2f13006a2006ad0238ce43887d08314c8e2f0619171",
                            "encrypted_content": "gAAAAABq0COPW81YCmA7VJWEaMeFZwZsixHXQg1hUZ8RoBpCteWdhDHmuIxYPWc-lwYTsH28yN7Hjlrgvtmvn5jtVILJozkIcXbLvyXzNi3hswFvpK0NOfkzIeepbxIkFPbEWmAHnGy12UpwFSaQHWNnq3n3_yx1QHg-6L84uhXb_gVLWbQpPfJSUHvqfymkwAH1GoqMKv5GRU1vQQzxs5O8WTfgNLyX0riq6mbYoO8jjRd-QzqkHf5w0fbNsQ0a1npvML0f6vzPdDsBKUrmDiwrd5HjM5tn62CI5ApllJFemVCTa_pXf0d7ws71b-g8d9jIVFtSoDe_oIZ6ZDTiGSRrdgRW3HzZG-hOS85nqwzYcUbXnZ_2dtZUelx1bFpGgH3udKhXIJ1YYXBoTsbYubHskPIWf0DE_rnbLUogslDawx9SBcYJ5dXTD0ipUYtIsW830XTVV1cOl2sFtp4SUrtJiV55G_r7sAxxUyQvF0-OlByS-wEbEvaQfylbFkHZUOcOs69DmgcGP1a_hdD5ZiiY3k1KG63JeIYFzDo1UomGzDFpbbihgq73-FD1JgHARe1vrwJ_0Y3BhhZjJ8UmeLGqyy4C6NQAmaLEg2iK4X-bzItSD-esAhENq-ZmUUKAY74SsGCdPK9srakfVz05rGN0_XKLkkoyW74UzWY89hkYAzrQOw5zGMghVKAioFcpyjX7JVz6fEdL4OWoaTknzhjnQKv1DEq2v8RsZVg3wWuU0t1WB562OQEIaScCVH4knjEtgCB8erZyStk65uwlXK5hQ6rof6Z4-jpd4xp89rAqNcX5OxnXG1SfST50s4C_YddfNe7Y0NQ9_yORz6TChuP3weUXnqlMOWHe4bMSTRLDITfLToWr4KwgajtQ_EmvCrbQWX5BHwSMHOCRKj92eSThzJ2EStpCmV-fihb6onbQk-x3i8pB0J3hkhfRzzsd2vEpy-E4RG9AgKhntN_gwP0Uez_hgEtfv9vipI8qVe-ivu168qU5N8siLollCok_DmIpnjsupxjNlifsv6LhKr97Y7S3TDnpdRL17Mt8lXJpuChunCDC7URHf990u4_j6tuQ9WLGKAq7F8lPoh8HzVdM5ozE-proWaEzFVd035l7q0dQxhQvw0JkhhcRMkjnmIFnTCQ3vAhHc2WCsy-gR5YI472RZoxU6x9q1KOVpTonw6ioUxZoB_QxZGxUT3FPdfMU6pOS0DZ0Ws1qL-49UOSuIleiEAhmX3E_lG_QD5dcxRbCOFHo3-JML3pTrXU49GZi7ZPqzmNcRSK7nxq1751e-YmBt_KYkqLI0sRM0Wwz1X1XTSv7S97iefaAj6xXhnlyc3hi5ofzj8iTGGF-mWYrC-F6GhCizpM91ePu6uugP-FWdtaQ5YXO-9Kq7mFwGB8NE-vftRZBr6p4TbGgXgrZk9rwMpgyC60WACV24UpJhaaBfnPgojE_nuskalMjAy41VA0CyMDCRBJ29MPE9R-dVMNIld9hfBSz2kMCaDXGw83fWPNAcUPQ2fvMBrBhkSLgTRhX7CDElb4foRJS6vTnTgW-es6mk9Ja-3-O013xo09WkqPyYieF1OIWyiALlUVqDtFYdWbUrIHUVV69D03XM3zIgy05HMI5uj46gT8l_X_nDnUI8zolKjB0STnTuzwwK4xvcwl7ank70dIvKatv62MrI-ZEp7QD00ZHeos5fh2X0YxNCsavuKF6UiT_hjWo-v9Sb2qJYKwiv_Bz_pqMqrQyQP-S35FkmWLzfu_ipbNxqE9z_u4WwvPUSdyAdMsJtBc41MJFtYAsBZc5jU0o7Sd2uiVQaUG51Q19jwq6MU68DxcTWukmdB4rQMd4EHYjtNJYr_zGGqzwJAO7gmKeAPxqbwbI5xg_JYGqdVfKBWKdZCCGljYW8DoimGJugt8OJvFFVu3wAOG9KRhaNtdmDnjYd24jxpqWtn5BZSNgwtQP1nh07a7aa-jsJmNpDxu9WH9fax7FXBjz8L9UlG47kBNTpQzKuW_8GRbe2Xvx5jJ0Rh9EFoW1XOiDap3s--MBUDwcRpQvxcU7Ip9gA7JNlsQJoglTvgZiQArlVlbkx7I6UOMHGzJ68UtuFyg3Z6eyAAXTBauozf2VCtbucac_f6kOC80xwZBxfTKG8iXlKOqa3_h_Q66kzPP6h6Ni4-FW-Z5b2aVCUGP_TnG4PkqS-ciRQz2CAdXcHkgy1bXbe-qWXpPBSYs0jnkBygX_v6sa3k8k2wXFenRz0nZ0D684585Axe5Kz5gvW5sR4n2LiN_x95cqDvXCLqzJ-T6kawaRBEoI2Ym3uw7tfOx3o1qyLAvZ9eX4c-dI1kVrC6WzJeeFy2OPOjY=",
                            "summary": [],
                        }
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 135,
            "prompt_tokens": 312,
            "total_tokens": 447,
            "completion_tokens_details": {"reasoning_tokens": 70},
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

GPT_5_6_TERRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "openai/gpt-5.6-terra"},
    expected_provider_endpoint="/responses",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={**OPENAI_RESPONSES_REQUEST_FROM_CHAT_COMPLETIONS, "model": "gpt-5.6-terra"},
    mock_provider_response={
        **OPENAI_RESPONSES_RESPONSE,
        "model": "gpt-5.6-terra",
        "id": "resp_0b9dd9372c7413ec006ad02386760487d08468a2a84f9e6871",
        "created_at": 1792025478,
        "access_programs": {"cyber": "daybreak_blue"},
        "completed_at": 1792025479,
        "instructions": (
            "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
            "Explain which file you need next, then call a read-only tool. Never change files. Use the "
            "chat_completions client for this project review. Consider zero, negative and non-integer "
            "quantities before choosing the next file to inspect."
        ),
        "output": [
            {
                "id": "rs_0b9dd9372c7413ec006ad023870a4087d0a4a8f5a9ef33dd2e",
                "type": "reasoning",
                "content": [],
                "encrypted_content": "gAAAAABq0COIulgFZHAIST02Ob7jMU14QKmgY_QYY0QhjIokwGaR_TwJrp4Ipb5-_NFswljdp2HFfgxzTV24tE6ueee6YavBUQAU0IW576eke0inwgG9fpAS8jCmhEjrldcu8mjx4tMPnzltz46Mj4z9r-nsH6C4wufPbjV3dpEGOnZ4OfaJwaO_tEFAz_0a7c3PhbkHsIiBkdgN_w88NdR4_hz5A8Sk-n0z7_MN75nKnzkEM0x5xL739fjnbAxEXNMn5Ezw3ctwuwab61YzmLua32kk0BtnHqhMjEm62UOX53zCFeoD1esiY2NhB4Ceq3i88gvLBtWqvo68zprjJhplhaJ6AaOTIAPSRWjrtw0ONDTLcqRb_mwsW7vbym3LUwakUqA5qMGYke9V9XzgQ44Acv9EHgr_6f2gwdVwjeupv9v4Kn6t1omYSi3HrjDl7tVJ-Hxn8TpHxQzK2i22vjHpjldJux4OnKKN-2Dh3KgT5qFa1NfcVSex5pfbxFDjhh12acNLRsPHJ3eRAx308BI2fkG7RdQ5Qb22Ezc8MY0AxERYHxgCxFBXInVHBBxOdciNPogBLKTTQSUFz0utIGEPwTpaNx2Akanb4H0o2sHCLA12Zq0C8h7EUGT9Kd4NrTIq1hAwzCn6cQyAw1SWatwIxaD49wyBq9xgeouNaOPKEkKGQSPHbTKsFuy3qZ8or7WvsOhudVF3RCk8chzm-CF8Dxqc5a3QZzKmMrBsBNILSLx1MQrq4kmUaYwt5V_moG1jT3_c9dTq4YmmhISUHOohjXz6HylzWGXMZXKW2_skr1ypaMtnlRoO-ngjraJisUheMuJQtoubouShhm6UiLDA_352QXxQX8Or0sxqm80Pn52u2k7oG0RSu4ltmERsBayQLnsZTIns6nXnfyTVDJ64vrSiX3UBBzzkJfCWifNb5dBervyf4gzdNhtBivQnR0bhKp8gJ3YSOl0t-SzEiQ4VIL6cu6OvnMtGCQ84s6BDS8-196It4RsP7hNkxL3tVAIwgb9Euncn3tVfRxz4suPS_b-2UhlFb7rxgz0H26DxNtCprPlaawp4pk4puKno1k7G2zx_BWP6vw8mC4E0ws0bUOWl5icgdWVjZvAQn2t76Vxb9DdZaWDDN6WrNGU85DcAgSOl_LwAnwEZ0IfSuZT3Ys7os_HFqJ-Isnrsv1YtcSg1fLL9KooIWZOEI9sH_PKLIBLyZTYSardtyb2izficMFTz_cJS3tHARyY_L1sfphULymlRSBXpI0Evr-d4PW-4dxcYIBMELEklJwgze9wjeFKwHlv1iZjuGlnr6ieoZ9aJXkQuxSQ-uom3DZdAs0hoW5Xgd1h0X9Tnu9bcJKA3KBVzDMPYLHkZYRD5J1nEifcDpXryBFRBFk4yYAOrPYEuwXg__1qtipv7-IoTc3J0pVmid_q6rHFrl0-tRfqM8-sG-C6QJMahzsDccevWyntC2o-8tAsuUuA3gkmrWR3Yepsj8LydSrqGlXYi_o-rE-XwNjIkyS-rCsweT1iBl3JvWX_ZDXEoLFSC50Hevf7Ti2HmbnMFm6HoRz0kIuAC91EvcY3xWXXKOsb-owbUiYhuoOPSo3xtiHxMjLARPwymWN4LkdsiSndglL7lsJgK1fsZR-fF4QI=",
                "summary": [],
            },
            {
                "id": "msg_0b9dd9372c7413ec006ad02387358887d091024ccc4d1075e9",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "I need `shop/validation.py` next to verify the current handling of zero, "
                        "negative, and non-integer quantities before reviewing the corresponding "
                        "tests.",
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0b9dd9372c7413ec006ad0238795a087d0aa5e2e27ec92e9b8",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_6kurIo7rXnkqfZEy2J7ldPNG",
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
            "output_tokens": 87,
            "output_tokens_details": {"reasoning_tokens": 17},
            "total_tokens": 399,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "openai/gpt-5.6-terra",
        "id": "resp_0b9dd9372c7413ec006ad02386760487d08468a2a84f9e6871",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I need `shop/validation.py` next to verify the current handling of zero, "
                    "negative, and non-integer quantities before reviewing the corresponding "
                    "tests.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                                "name": "read_file",
                            },
                            "id": "call_6kurIo7rXnkqfZEy2J7ldPNG",
                            "type": "function",
                        }
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_0b9dd9372c7413ec006ad023870a4087d0a4a8f5a9ef33dd2e",
                            "encrypted_content": "gAAAAABq0COIulgFZHAIST02Ob7jMU14QKmgY_QYY0QhjIokwGaR_TwJrp4Ipb5-_NFswljdp2HFfgxzTV24tE6ueee6YavBUQAU0IW576eke0inwgG9fpAS8jCmhEjrldcu8mjx4tMPnzltz46Mj4z9r-nsH6C4wufPbjV3dpEGOnZ4OfaJwaO_tEFAz_0a7c3PhbkHsIiBkdgN_w88NdR4_hz5A8Sk-n0z7_MN75nKnzkEM0x5xL739fjnbAxEXNMn5Ezw3ctwuwab61YzmLua32kk0BtnHqhMjEm62UOX53zCFeoD1esiY2NhB4Ceq3i88gvLBtWqvo68zprjJhplhaJ6AaOTIAPSRWjrtw0ONDTLcqRb_mwsW7vbym3LUwakUqA5qMGYke9V9XzgQ44Acv9EHgr_6f2gwdVwjeupv9v4Kn6t1omYSi3HrjDl7tVJ-Hxn8TpHxQzK2i22vjHpjldJux4OnKKN-2Dh3KgT5qFa1NfcVSex5pfbxFDjhh12acNLRsPHJ3eRAx308BI2fkG7RdQ5Qb22Ezc8MY0AxERYHxgCxFBXInVHBBxOdciNPogBLKTTQSUFz0utIGEPwTpaNx2Akanb4H0o2sHCLA12Zq0C8h7EUGT9Kd4NrTIq1hAwzCn6cQyAw1SWatwIxaD49wyBq9xgeouNaOPKEkKGQSPHbTKsFuy3qZ8or7WvsOhudVF3RCk8chzm-CF8Dxqc5a3QZzKmMrBsBNILSLx1MQrq4kmUaYwt5V_moG1jT3_c9dTq4YmmhISUHOohjXz6HylzWGXMZXKW2_skr1ypaMtnlRoO-ngjraJisUheMuJQtoubouShhm6UiLDA_352QXxQX8Or0sxqm80Pn52u2k7oG0RSu4ltmERsBayQLnsZTIns6nXnfyTVDJ64vrSiX3UBBzzkJfCWifNb5dBervyf4gzdNhtBivQnR0bhKp8gJ3YSOl0t-SzEiQ4VIL6cu6OvnMtGCQ84s6BDS8-196It4RsP7hNkxL3tVAIwgb9Euncn3tVfRxz4suPS_b-2UhlFb7rxgz0H26DxNtCprPlaawp4pk4puKno1k7G2zx_BWP6vw8mC4E0ws0bUOWl5icgdWVjZvAQn2t76Vxb9DdZaWDDN6WrNGU85DcAgSOl_LwAnwEZ0IfSuZT3Ys7os_HFqJ-Isnrsv1YtcSg1fLL9KooIWZOEI9sH_PKLIBLyZTYSardtyb2izficMFTz_cJS3tHARyY_L1sfphULymlRSBXpI0Evr-d4PW-4dxcYIBMELEklJwgze9wjeFKwHlv1iZjuGlnr6ieoZ9aJXkQuxSQ-uom3DZdAs0hoW5Xgd1h0X9Tnu9bcJKA3KBVzDMPYLHkZYRD5J1nEifcDpXryBFRBFk4yYAOrPYEuwXg__1qtipv7-IoTc3J0pVmid_q6rHFrl0-tRfqM8-sG-C6QJMahzsDccevWyntC2o-8tAsuUuA3gkmrWR3Yepsj8LydSrqGlXYi_o-rE-XwNjIkyS-rCsweT1iBl3JvWX_ZDXEoLFSC50Hevf7Ti2HmbnMFm6HoRz0kIuAC91EvcY3xWXXKOsb-owbUiYhuoOPSo3xtiHxMjLARPwymWN4LkdsiSndglL7lsJgK1fsZR-fF4QI=",
                            "summary": [],
                        }
                    ],
                },
            }
        ],
        "usage": {
            "completion_tokens": 87,
            "prompt_tokens": 312,
            "total_tokens": 399,
            "completion_tokens_details": {"reasoning_tokens": 17},
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
