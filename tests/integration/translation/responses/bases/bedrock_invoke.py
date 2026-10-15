import json
from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase
from integration.translation.responses.templates import RESPONSES_REQUEST, RESPONSES_RESPONSE
from integration.translation.templates.bedrock_invoke import (
    BEDROCK_INVOKE_REQUEST,
    BEDROCK_INVOKE_RESPONSE,
)
from pydantic import JsonValue

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
    },
    mock_provider_response={
        "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "id": "msg_bdrk_uaxykqyuejauacvh6ordtb5bbst4xlruet5vju5zjinbcjhycmnq",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 17,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 5,
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 17,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 17,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 5,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 5},
            "total_tokens": 22,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-4-6",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-4-6/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
    },
    mock_provider_response={
        "model": "claude-sonnet-4-6",
        "id": "msg_bdrk_011CfjukowJGNdRPkmrmvmEU",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 18,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 5,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-4-6",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 18,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 18,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 5,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 5},
            "total_tokens": 23,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-5",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_bdrk_vrjdwnaebmlfxkxvd2a74ovr6xl6a3ovufg3gwh3hktoiockg56q",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 21,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-5",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 21,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 21,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 6},
            "total_tokens": 27,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "bedrock/invoke/us.anthropic.claude-opus-4-8",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-opus-4-8/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
    },
    mock_provider_response={
        "model": "claude-opus-4-8",
        "id": "msg_bdrk_j2vze3iapymvotnltudlq6kyexytgav6tdiwmintqfjsdga266bq",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 21,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "bedrock/invoke/us.anthropic.claude-opus-4-8",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 21,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 21,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 6},
            "total_tokens": 27,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)

BEDROCK_INVOKE_RESPONSES_SYSTEM: Final[JsonValue] = [
    {
        "type": "text",
        "text": "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
        "Explain which file you need next, then call a read-only tool. Never change files. Use the "
        "responses client for this project review. Consider zero, negative and non-integer quantities "
        "before choosing the next file to inspect.",
    },
]

CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSuQUKEggSEAEYAjgBQgh0aGlua2luZxIMkzWpvtxVqloRdTejGgySvyYIIigSiTigYwsiMOl8XJI9rkOBrw8tVJltn2ZOQqbsCTr8UQ1o"
    "5xBXnkf4Pj+F0fd7GRGRGWVhSLbFCSrUBMeafwRFgNCuZyPdn7n38LWOXm7+Lg8tjM0m2PfA04geMhUvcih7RPypybFHnZgmoxqTnoxhyMYA"
    "BJ6CuZ1PHCKCXXTLGF2SsBJCCKhOeDwlBnVbwgbxvTUScOtpx/12go2VloE1aRsFTdpG4oZlziwPqV8GYUv0Pzw8zlEAbdlK/A/pu7vH1eM+"
    "08tFzjRwl5/ACM8NIQ3s/baA58Y1Cfu2yTUyzxV13+MaBnGEhEMPhI/3muGRHhT20ga9jOhGz4uH67he+BTschcNoIi6Pw2ZUy6PAnfjs9Sx"
    "3tJMQyUCThBsmKS1nGSTu4R8NQMKJWnMGr0X6fyk5PphRmC4YEYTjJ4nw3Ad6Kog06Bh1LbJTphVX3V3qc4sMBbrKSPH/vd/f4OCgrWU572B"
    "lxYl3HNT+gwJa+uBfQShW11KQR8A0v8T33jnHAmysQ87+iJpTm1u+W3oMoemDsUddpICg6JQdxDSA4Ft2iuDds3XgeeOmXYl8mc8pYzuep40"
    "zUjmyJ9cPZYjWNArd1IQAFk6s5wa47jHxP72gXiiXDJabXGT1/xvsafaM9WjZ/cxmmyF2K3122br9OWknau/rJa0gg1+PluGe/vUptHi0Acx"
    "a4pNbvcQFiRutnwjPr9uqQMqk+cqRdBVS2pZAc/vnmGi7m02MjBuHRqZeFMIkc4WYK3fgt0BxnMyXMJCzQcx64oGg5ZgU15crKV/thvRkvsI"
    "AjIl4ScwNIsYlQcz3fx7c1yH2hRIWb+ahmdGbJvnDlcNmliO50qWyhRyzYBoCU55cvXtnJTHGAE="
)

CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "bedrock/invoke/us.anthropic.claude-haiku-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-haiku-5-5/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_INVOKE_REQUEST,
        "system": BEDROCK_INVOKE_RESPONSES_SYSTEM,
    },
    mock_provider_response={
        **BEDROCK_INVOKE_RESPONSE,
        "id": "msg_bdrk_ytkmzvqioxcm74hxqxgoycuu6vt3orsozxii4miswb5tk57ya2ra",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll open shop/validation.py to check how it handles zero, negative, and non-integer "
                "inputs before deciding what to look at next, starting with the first chunk of lines.\n\n",
                "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01Uivu38DoXo46h5UDP3hKz9",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 175,
            "output_tokens_details": {"thinking_tokens": 75},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "bedrock/invoke/us.anthropic.claude-haiku-5-5",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'll open shop/validation.py to check how it handles zero, negative, and "
                        "non-integer inputs before deciding what to look at next, starting with the first "
                        "chunk of lines.\n\n",
                        "annotations": [],
                    },
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "I'll open shop/validation.py to check how it handles zero, negative, and "
                            "non-integer inputs before deciding what to look at next, starting with the "
                            "first chunk of lines.\n\n",
                            "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "toolu_bdrk_01Uivu38DoXo46h5UDP3hKz9",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_bdrk_01Uivu38DoXo46h5UDP3hKz9",
                "namespace": None,
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 877,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
                "cache_write_tokens": 873,
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
                    "thinking": "I'll open shop/validation.py to check how it handles zero, negative, and non-integer "
                    "inputs before deciding what to look at next, starting with the first chunk of lines.\n\n",
                    "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                },
            ],
        },
    },
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSiwoKEggSEAEYAjgBQgh0aGlua2luZxIMOIdSxTR4GjBHazgyGgzAFsCOD/EpsKJodeAiMLAgKd6UJh7VQCix7AFKFkbf6rPK16fItxJ9"
    "PI5nOnHUS0SfBCxVY4LZmRg71SmHYiqmCTfFsnMccIbkeJS2+Xap1iQw/HZ7KYHJJsCqDxm9Ua2I8LHYyL3s2OVT9NzUiCMKZQWPQ2DqBWZ0"
    "gsnrDQ9ZYkNrtXWxypuIDhHQw7dOs89033JTcFGdnxpFwvpvARtjSHPS7O9UOY48wLeD+K0AL6bzJlmbarntN3xeqYnYFU9cAIhyjSrCYWtE"
    "5Y37pblxLuwXjHI20rSyT2carPP7whuU2LDg8GVIu0JdDO4seHyys9s+BaiMJEPnn2naeK3WAyOvM5wTyvYj81cxx9xJBlVuFyeyxxyTQgnL"
    "DyM7OtCiy35affvC1oKXwYQyiRUYso/4IuEBTiziluiO+wYg/NAHUycI+HEdFp7zOIK9AdC1WtB5xbUtclcCObYpPgdSfR8RtcgaUpltqIfJ"
    "VA+9K+hDDNTKKh/Xa4+WFvVj++6NHe7glBlQeasGOtNtrFbRJUB6dgnorTWblMq8NyzCASK5FoQo77Fv3lKdAleW9ujj1eGA2hon5UoO9BGQ"
    "w0pnLzColC0V3NtAbpuC8Jl9mm5te2ChvJJV7bYa6/3RYlA3fRTAvP6grSCRTMYkxnqtJCliTb6sjly99vkD4zV7gMvjY8BW08rVruhMShUk"
    "gHRC6XBMw6kvUboxDnMs1yCJCYyHqUI1AHaP2g24VigzgWwOPpNSRm1lgodCYZUM8/gznj1JuAwe2XoBXgfBvkRQCYXorCulmvjK3NMIBzdU"
    "fsfyGExkFdaQUJLbhWq6IF6L0b9iKLuugkavPUaodPpsg6L3Z1XHsFDZgO7GxRoHcKvV1/cEtO6DU7cc+gu4D/dSsnp8K2BY00UKfDyBqG90"
    "OwLw6IDyaiw829KOc+3lVBW0vwxhbA3AV++soacROCplQ/Uy9qQy8lB63whvFu4wGypT6yn+2k9TUSNSbgimfvO3SzuWUenEpnkSVO8CFvRQ"
    "iffW03nhllZyHHlwB9B+NSrghpom2VvAd4OBg7J4353Favtsv/nu6LRpUUalVbkmX37iwZ/Q8gEYooxqa/EZodr0+1mwHvRJiascYs67I+MV"
    "+9EMffg2inNTIKfDu5GzRJOKWULP4e60/9Q5oJYZ5Pw+LkUyrIQphe+jNAczMq4N/TBPVIQhJWgJaCNfSV3JGvvLPOgPEtZjaIQHYIVn+sID"
    "XH/+DsPUWSLKDLibTDPAoCgIQQN2U2TJLEzrs4fr1wej4fJlQ7ufZRXcRisAmbsM4AUqRXrErjjcfgg2cgtPN3Ol1Yfg7XNm5fw09WJD/DKC"
    "C34ocjCqBXsZDQtb8T3w54OxIiRLn9VMj1pizbgC+x2htmyBt6BNHns7K08DM67SiG7ph049Ze/SFoZqb4waCUu1dRYZAUKUo50Uy25Xi5Qi"
    "cHhcUOWHxIiTl50THEQCA884azGo9ZNpby+krX1dfVuGSovwvESk0GGNM+LgKNw1tpj2srpQZIsMG8Oo1fFxriIX5dCJq1v4J8gHw75q38wx"
    "ySwPds/XHbkgK3KgkVrL/XcfUP3myMke3shNr8o7tm8SulaUZn0J4hWawGegO6wS/44Q76OoeLluOvOHyERd6wMPquKzo9lZqATi0x8CrpC5"
    "GAE="
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1: Final = (
    "CAQSwwYKEwgSEAEYAjgBQgluYXJyYXRpb24SDHTsA0CfvetGZFL8axoM4khrNWxDqnQ4bENWIjDrerGCpIXFI6LZF2Hk4OYvtcRO6Ytxz5kc"
    "E7vKFBT57mJF6FitCHcxRNk1M7/C2L8q3QUy4sf0KVQTWQbGHFutqk/OUkcyJvaS/tPjOrhqUWhsWRsInGBGX3sjSLMWWNQ3bHdnql0bVDFc"
    "YgLfO2K+1Ta2P3pna3jpvBO+BM+W1xQ1FrenEWqbXtNwueTC6kQqGUZnmULo3cbMb5l1Yo2nui6i87a3Bs03jtomyzjRb+xMH2XNpePzURLf"
    "ADvZIHyAPOlHOnK6ZTk4rugCEHHTAudo9Hz/6ZDTjXTpqzDD69lxgcdviZ2oNMpExhPxJctgzNBuYdFYjT09B0Atag0HLXYdUL6L5Cz1599n"
    "AiTQl60CWALeyuJS65GikY3FNn5tXt1IPHLUsNsoEzBr07lvZ1WeyyiyoExBOCSgH0KuxSIP9TL4hQsMiI9kduv3w4J9ybSyViI4sXg3nlUz"
    "5UqNonSx9487qwcmlZxZQquU85UtyA6Kcym6oAUH3cr1+VrYvpBZzwku00ccQQ2faJGZ18AIBYwvY+HZQ0bPop7AkYSqIyeLe5A8vcm3O+o1"
    "sRpj05W9C649sYuPo9qXUabkB/WlkoLydG3X+erc1ry3c6yur9XYrtGfXcVCdA2xtYOkzYlBXrYTbVL/oes3aBTt2Yg+0OLaa8XM7rvajpF0"
    "QkYwtAQwWwyBe2gPtMWVJUuvljICUwMhKPz2bD3/7X/cuNKmso3Y+sGzoCtylrggSkdVdMNB1zTebX5iPmIk2LDGuEDyqyiJGugGo04yEzyv"
    "Q1icaJmssvQprgCmJJkHB1PnR3Lw2tMwCFgyrJxXLi8lDSyFmcl9YoHF0OBcw0+HFAuZsgkbjHXeVU/K3jFAj+GP3a7HBrwoN2jBHI9RfU74"
    "Un4iBaoK8JWdy+/vR0vdGiifISnlzmSOL9y+mmJubPkLMxwH4+vnfXuDMYvDhgn1M7vSUwiHePupdVz6HxS2LjY7XrwVgLSZHoGmvdP6NoeB"
    "kwzjb7hXaitkAzyqZls63keq+Hw+tPTlMGO5NtPeGAE="
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "bedrock/invoke/us.anthropic.claude-sonnet-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5-5/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_INVOKE_REQUEST,
        "system": BEDROCK_INVOKE_RESPONSES_SYSTEM,
    },
    mock_provider_response={
        **BEDROCK_INVOKE_RESPONSE,
        "model": "claude-sonnet-5-5",
        "id": "msg_bdrk_i3avv2mb3domrfdrl6s7b5mubzt6vckxqeea6ieovrsgmyzby7ca",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user asked to read the validation file first, though reading the test file in parallel "
                "wouldn't conflict with that instruction. I'll prioritize following the explicit order and "
                "read the validation file first.\n\n",
                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "I'll open `shop/validation.py` to see how `validate_quantity` handles zero, negative, and "
                "non-integer inputs, along with `tests/test_validation.py` to check the expected "
                "zero-quantity behavior.\n\n",
                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01FfS5J9LzLxLwq5caHEM4xW",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01FTMGZf7C2vzjc2Ux2SJyoU",
                "name": "read_file",
                "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 40},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 536,
            "output_tokens_details": {"thinking_tokens": 242},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "bedrock/invoke/us.anthropic.claude-sonnet-5-5",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The user asked to read the validation file first, though reading the test file in "
                        "parallel wouldn't conflict with that instruction. I'll prioritize following the "
                        "explicit order and read the validation file first.\n\nI'll open "
                        "`shop/validation.py` to see how `validate_quantity` handles zero, negative, and "
                        "non-integer inputs, along with `tests/test_validation.py` to check the expected "
                        "zero-quantity behavior.\n\n",
                        "annotations": [],
                    },
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "The user asked to read the validation file first, though reading the test file "
                            "in parallel wouldn't conflict with that instruction. I'll prioritize following "
                            "the explicit order and read the validation file first.\n\n",
                            "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": "I'll open `shop/validation.py` to see how `validate_quantity` handles zero, "
                            "negative, and non-integer inputs, along with `tests/test_validation.py` to "
                            "check the expected zero-quantity behavior.\n\n",
                            "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "toolu_bdrk_01FfS5J9LzLxLwq5caHEM4xW",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_bdrk_01FfS5J9LzLxLwq5caHEM4xW",
                "namespace": None,
                "status": "completed",
            },
            {
                "arguments": '{"path": "tests/test_validation.py", "start_line": 1, "end_line": 40}',
                "call_id": "toolu_bdrk_01FTMGZf7C2vzjc2Ux2SJyoU",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_bdrk_01FTMGZf7C2vzjc2Ux2SJyoU",
                "namespace": None,
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 877,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
                "cache_write_tokens": 873,
            },
            "output_tokens": 536,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 242, "text_tokens": 294},
            "total_tokens": 1413,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "The user asked to read the validation file first, though reading the test file in "
                    "parallel wouldn't conflict with that instruction. I'll prioritize following the "
                    "explicit order and read the validation file first.\n\n",
                    "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                },
                {
                    "type": "thinking",
                    "thinking": "I'll open `shop/validation.py` to see how `validate_quantity` handles zero, negative, "
                    "and non-integer inputs, along with `tests/test_validation.py` to check the expected "
                    "zero-quantity behavior.\n\n",
                    "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1,
                },
            ],
        },
    },
)

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSgwUKEggSEAEYAjgBQgh0aGlua2luZxIMT1FeqW5PaXtFp114GgyoxyN+NbjKsT1yxB4iMF13gzAPt/OOukSm+Is78v7dbu1WcrJvArvI"
    "HMK3ZeviwjQPyQ0zywjsqDWnepR7/iqeBAFSYFwVgmpUGxZIoN7tQ+XnRVF+YNYPY9HbKw6lE89F3jejjSkPFYWIUkfW2KsB3/kEmkWrXzV7"
    "dIWAxKfGGI6gvIvzVGl8UoKB7cYJqtx1888mqoLH8mGjYRtMrPang71iBUoiJjThOdSiPSTr+HnvGb6e9CpXduwBsuCATo+ZKAKiyDoYEqr9"
    "HRmvz7fi6RuKSErHfG9Zod7muUKjqw2Y/kaVJenornm8JuMYuYIfnv0II3MBHtH0nKr6ZyPuaALFl+L7sh3eazeahWzC3CeZ8LZmLp5NuOg+"
    "xsHNJQDjWJuXIUrCoYdc3gnCgcnuEYd2dz2+tmrR7K509+uCOe9EEoO1SuKU+YM9qkSBVPo3VR99cf2XpUc3K+4v1KLocUjtLlgoLmxlpES6"
    "uJLhyP65XlvIFAOGAYH0hiwvbUq55i36T9LHG4mbUdwzjum1CSzZ8trx3k1lzcsxlLVPIY3JuBE63HfqWZs/T8UobfsPbTCrMgBqpmR+wBdi"
    "8wKJPiasyregTVYWuhMrg1LFodr80/co5TLeLAvoI/oXs6Syz8ME8IMpKD4iHNr7Pj7/Le/XEAU6W+2n/b8Hx6NVRysrl729eIqV4Y2pk2Xl"
    "CUCVx0wfuAlvw5TR4Ft6Eb7/wbSOlxU/E5iqaJusZbpQLEtCTUXU4IKgmaPJZFxKWIEv3hbROZ477bvTyROQgaDlcabkiPH1o6FREM5BeGI1"
    "GAE="
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "bedrock/invoke/us.anthropic.claude-opus-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-opus-5-5/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_INVOKE_REQUEST,
        "system": BEDROCK_INVOKE_RESPONSES_SYSTEM,
    },
    mock_provider_response={
        **BEDROCK_INVOKE_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_bdrk_cwcayjjtio2jq5g6a45acoimfxqughi7fphihxe5toxmhqojwboq",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll read shop/validation.py near line 18 first, as the user asked, before looking at the "
                "test file.\n\n",
                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I'll read `shop/validation.py` around `validate_quantity` (line 18) to see how it handles "
                "zero, negative and non-integer values.",
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01J5u6tgn3ppV4c2VpmD2xy6",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 50},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 196,
            "output_tokens_details": {"thinking_tokens": 51},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "bedrock/invoke/us.anthropic.claude-opus-5-5",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'll read shop/validation.py near line 18 first, as the user asked, before looking "
                        "at the test file.\n\n",
                        "annotations": [],
                    },
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "I'll read shop/validation.py near line 18 first, as the user asked, before "
                            "looking at the test file.\n\n",
                            "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'll read `shop/validation.py` around `validate_quantity` (line 18) to see how it "
                        "handles zero, negative and non-integer values.",
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 50}',
                "call_id": "toolu_bdrk_01J5u6tgn3ppV4c2VpmD2xy6",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_bdrk_01J5u6tgn3ppV4c2VpmD2xy6",
                "namespace": None,
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 877,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
                "cache_write_tokens": 873,
            },
            "output_tokens": 196,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 51, "text_tokens": 145},
            "total_tokens": 1073,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "I'll read shop/validation.py near line 18 first, as the user asked, before looking at "
                    "the test file.\n\n",
                    "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                },
            ],
        },
    },
)

CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0: Final = (
    "CAQSxgcKEggSEAEYAjgBQgh0aGlua2luZxIM93CYh6+fYNYfl0lvGgw7Sq5PzeOUQaR9JKoiMK9/V24UwNwvJbJdQjDEFCLEIn3lZgk7//Uw"
    "Hr6kTvOTtxS6Xdgt1IphkLtCfI1rWCrhBvKpjQHMVva70EZuEDWg2HPEMrmnrvUOXJniTRUiArTXQTR6itTG2XXqxHHsARUCq/1ECbBG9j5X"
    "Ti6iF/bgGEVkeWr4rmCNYnQtGS1Rn+f83bHPbnBwyMOH29aX0JS9kstQFoaPOa4G4BnzVgFg5Tnj6JN9hh+aSoBMCliraNTo1lirHHpQXqCZ"
    "jupkIKRjDyvRfpE96tGjtWTJFJ3f8HTm2PiTGUOhCz7PoypLgCfOQWyfaptuIUKa9sZXtJVlLBmoj1XDqIqQKsCcvDKPjDnAVVsTr4dAExxP"
    "g/3ujwP5Y/+LDHFE9XPetApyfiIh3qy5L9UH1Yv9RwKLqfn3KWPlXEWoCrvgOTj7e5EfTk7Cu3WxDNp8SNs0TeFnGsCKQGASuqTtgZTYzrpD"
    "Jc1EbTZOt+CRVUgVhcbadMSdhpy/PcAtgiZ8/MvZCCv7BeORzmVYb9cYaVFVQK4/tchF/lcK+4YVvDHCCSXAwhj6Y9NyQ7j94JmFNqhjsSK4"
    "t4qALhhmOcJuCispP6ZRhf1SCoajBsRVRdtvPbFnOH1GqdErrzAEcD9VZk1mkmUN6fs9Vl9Uva6XjVr7YzQHjdkGZ5fEHcIpmJrS1ZMI+YuC"
    "quoy4qMJvp8g8YgoLYjt6GFWqwOwePTuFmMnL7VadlmEtwAionrUqEXCwSh8FVOeN8mvaoA2V2AtGp5zA4SDmbrF+XhuNS77ac0dCcQbs8iy"
    "H1a1oSySzqGf+muJBkoSCVtIAPQ48+9ha3fGM49fj2EQINCu4NhcBylZqFhHV7FJA620qqEJV8kWBJqc6j9pYHQYSirhnBLxPFPFQS3UNItx"
    "k4c5oPs3VGPea++KQhhbRnlAEPu1pKzKMM7yg9+rT28l2XiIGnWSqBo/J8ipsL/uFcthzptNtoK75pxhGduvynAI7wRbArxT6UMegIsPdRlH"
    "uXBbyXKuXhbnCoCkbb7VNbsO2tWWqE5rUdLgKyHd57SL3BlJDVd+V+/bW84zqP7QiSIMmrv3Q8lVFjFyOEXfeRZF7pXapgdAK5CWgoKXwdVF"
    "oar+HuszSP6LPqJggfD+9eqZhpZI6wmUzSVc9TlZAV+fB7pWcDVgBYG2Vk6bckiXOZrJigDuAbHObeZ6BcKQpID/9TvvtNKnQGJy4Uva2p8Y"
    "AQ=="
)

CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "bedrock/invoke/us.anthropic.claude-fable-5-1"},
    expected_provider_endpoint="/model/us.anthropic.claude-fable-5-1/invoke",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_INVOKE_REQUEST,
        "system": BEDROCK_INVOKE_RESPONSES_SYSTEM,
    },
    mock_provider_response={
        **BEDROCK_INVOKE_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_bdrk_ehpngdq5umo3xytr46gfdb3ifglm2bsofgtx5zargdynfbgdvepa",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user asked to read the validation file first, so I'll follow that order and read "
                "shop/validation.py before moving on to the test file.\n\n",
                "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I'll read `shop/validation.py` around `validate_quantity` to see how zero, negative, and "
                "non-integer values are handled.",
            },
            {
                "type": "tool_use",
                "id": "toolu_bdrk_01Ee5Sz4v3R2bgHS6gQV5m8N",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 873,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 283,
            "output_tokens_details": {"thinking_tokens": 142},
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "bedrock/invoke/us.anthropic.claude-fable-5-1",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The user asked to read the validation file first, so I'll follow that order and "
                        "read shop/validation.py before moving on to the test file.\n\n",
                        "annotations": [],
                    },
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "The user asked to read the validation file first, so I'll follow that order "
                            "and read shop/validation.py before moving on to the test file.\n\n",
                            "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'll read `shop/validation.py` around `validate_quantity` to see how zero, "
                        "negative, and non-integer values are handled.",
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 40}',
                "call_id": "toolu_bdrk_01Ee5Sz4v3R2bgHS6gQV5m8N",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_bdrk_01Ee5Sz4v3R2bgHS6gQV5m8N",
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
            "output_tokens": 283,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 142, "text_tokens": 141},
            "total_tokens": 1160,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "The user asked to read the validation file first, so I'll follow that order and read "
                    "shop/validation.py before moving on to the test file.\n\n",
                    "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0,
                },
            ],
        },
    },
)
