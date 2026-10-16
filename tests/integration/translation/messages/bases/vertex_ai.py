from typing import Final

from integration.translation.case import TranslationTestCase
from integration.translation.messages.templates import MESSAGES_BRIDGE_RESPONSE, MESSAGES_REQUEST
from integration.translation.templates.vertex_ai import (
    VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
    VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
    VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST,
    VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
)

GEMINI_3_5_FLASH_THOUGHT_SIGNATURE: Final = "AY89a1+kriNs9YsPegJQxgjTWsMlxsqPU/UkHDyz4EK+3FRrE52ygqPIHTvkblgKKiO5Fn8R9D3iu//i+YnuuZ+e7W0yxUoFScas89b5rn3vZHsihnUA7VTe3b7beth1IftteoTGZ3F8OSLrSIP8HKn4xTIx35eeNEt5oiD3K5rTQrdklqO277HvTDhu/LuZhZ0pzaWciGQEc61epIBBpC2ygu8V4L8TekUVZnz6qzSzxipdiQStQ8dZou43334hMnxOR6JjJCGxqt/1K+C0vXX6YP6f+Bph5VhPSnvy2Z1uApi3TUxAcsDDyVewIr+ha2G6DPhN1Oi5uZZNhmbvH01Uq5OOquF1uEICq2m/35LhJIbDNM9N7oI1Z6L94P9xdugeFXM2S7wdJfc8sdg09vEzNeA++GK25zRuB4hU7XyaG4XnWjmu2kswi4sf+EyrO/K0P2tb/ZCJu7kJTtWTiD1wgSIl1eqS8ORKKsXDN2hIbFrtsNR50WIvVzd6"

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "vertex_ai/claude-haiku-4-5@20251001",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-haiku-4-5@20251001:rawPredict",
    expected_provider_headers={"authorization": "Bearer scripted-token", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 64,
        "stream": False,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_vrtx_011CfjuyLkoJCtCANzSshqj7",
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
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "model": "vertex_ai/claude-haiku-4-5@20251001",
        "id": "msg_vrtx_011CfjuyLkoJCtCANzSshqj7",
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
        },
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "vertex_ai/claude-sonnet-4-6@default",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-sonnet-4-6@default:rawPredict",
    expected_provider_headers={"authorization": "Bearer scripted-token", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 64,
        "stream": False,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        "model": "claude-sonnet-4-6",
        "id": "msg_vrtx_011CfjuxNMG2o9mmi6TvgC48",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "container": None,
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
    expected_litellm_status_code=200,
    expected_litellm_response={
        "model": "vertex_ai/claude-sonnet-4-6@default",
        "id": "msg_vrtx_011CfjuxNMG2o9mmi6TvgC48",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "container": None,
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
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "vertex_ai/claude-sonnet-5@default",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-sonnet-5@default:rawPredict",
    expected_provider_headers={"authorization": "Bearer scripted-token", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 64,
        "stream": False,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_vrtx_011CfjuyZQJsY8CfdZxY6C9E",
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
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "model": "vertex_ai/claude-sonnet-5@default",
        "id": "msg_vrtx_011CfjuyZQJsY8CfdZxY6C9E",
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
        },
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "vertex_ai/claude-opus-4-8@default",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-4-8@default:rawPredict",
    expected_provider_headers={"authorization": "Bearer scripted-token", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 64,
        "stream": False,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        "model": "claude-opus-4-8",
        "id": "msg_vrtx_011Cfjuyk5joSiAW8h6nktJh",
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
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "model": "vertex_ai/claude-opus-4-8@default",
        "id": "msg_vrtx_011Cfjuyk5joSiAW8h6nktJh",
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
        },
    },
)

GEMINI_3_5_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "vertex_ai/gemini-3.5-flash",
        "max_tokens": 1024,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-3.5-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={
        "contents": [{"role": "user", "parts": [{"text": "Say hello."}]}],
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
        "generationConfig": {"max_output_tokens": 1024},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello.", "thoughtSignature": GEMINI_3_5_FLASH_THOUGHT_SIGNATURE}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 2,
            "totalTokenCount": 95,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 9}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 2}],
            "thoughtsTokenCount": 84,
        },
        "modelVersion": "gemini-3.5-flash",
        "createTime": "2026-10-05T22:54:50.809002Z",
        "responseId": "uirEaqqwMc-U9LsPjJTYuQU",
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": "uirEaqqwMc-U9LsPjJTYuQU",
        "type": "message",
        "role": "assistant",
        "model": "vertex_ai/gemini-3.5-flash",
        "stop_sequence": None,
        "usage": {"input_tokens": 9, "output_tokens": 86},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

CLAUDE_SONNET_5_5_MESSAGES_THINKING_SIGNATURE_0: Final = (
    "CAQSiQYKEggSEAIYAjgBQgh0aGlua2luZxIMgrCyOX5aamkI52eqGgy+6kkncswGkygvTJMiMP0m0811aMrw7nEXG8RULCLVjlRy"
    "YnR1NVQMPi4D8+0wd7GpOQ3kYb6rRHr4eZQrXiqkBeZyXjb/2wJIq+K8HxoIHFJ1u6xQrkMnZ/tA6M7Kt9Hvftp9EvdEDGYlEN9t"
    "aLY0iXPBS0XwePUByV3ZDxTgNx2wXj1K88BwXtpQYYdDUFFb0pC5/HlxxZE1xHeQpHX/7sfTroWGNNco1JKKMC9FfT/YkHM+CaE+"
    "LnIaPmL2Ft5t/4UhwBcgIbGIrj0INS0QObLNVSy32zLbxiyaDzC/ZVb9FC+mzRsQdkUpKDsiZMc0quyLrmuCl0cxuypCFiwPXrTM"
    "mRSMZiHN6LG/D0w9ySgHalASqB11GuAKPq34XXfK/+xbgwpGnBSKqFYn5CZkVzjV9sRdNMxRd+NjL/RN6DbyFm3xDF9VapDqjkuc"
    "MQUUJbHRDP7nIVRPGvSqE1cE2qhrimR8uKKH9MtjMG6EPAaL3gMWyIwpxM7UOX602XYgpQpxNa2yGdtv2R8740FrKRucYsKkCRbo"
    "fO7SI8p+mR8KChm0zY6Q+qFWsZvaeQZPrxvERghOKvXDBwbCtR2OIru2ru6JeBfNqJZsIsPGBBENEjI0LkB/D9/iURQaqO1wgXLb"
    "HWtSt1ErxTtGRjWEKCEETfy77eoWptmgGzitX8Qoim25TVjTK2Pig7rWfapiOrxpLFo5zGfDuKJuVqDdcS77BXQ+LpfzaD0AWoOx"
    "z3VCLyX0wgttyDDuAF423Iu6HESf0z6FbPpZNiuJaAARNZAFYhp0AFHCCwyRtGqUuas75Jzp2bpCWzlpzJVXZ9vbWS+AYkBEwaig"
    "MuGS6q5bz5efcypwiv2+OHPGOh7Wj73BgGyiwNg8KLGIR/xWEkGuNRPrApty7ABzOlZNIK9Ti1Ay2yCBIZWQSAfTPmUBcZi5xlgB"
    "LmA+bljzJJq/cJ4P7wec/5WeHyYVEwQ5pCOcS3mRj5QYAQ=="
)

CLAUDE_SONNET_5_5_MESSAGES_THINKING_SIGNATURE_1: Final = (
    "CAQS3QUKEwgSEAIYAjgBQgluYXJyYXRpb24SDBELWnnn6q16EbJTkBoM4dA0bbJp1Z9s2epTIjALivb2lWtreb6CmIP6jl8Af48M"
    "XpiMfs5leblsmVBauYg0j+JPBYYDD92ttpEDBiwq9wSfdfC1d1MhZgmJ+dB/DAUva2swtVmdUIpxDfPT2jXA1v/zydQ+GUGGwfGX"
    "FoaACIyUPYGR1nmtfrxp19wWKJkFFkw/NM+pjmK+yTLlExT3ptqwPmmCTC5q8Zh6GjWnKi0OXVdo2fkb/4OhkX8Xpib1TLL8iwD8"
    "7+jwqNbbNMIE+V5Pf96dI0yCkFFWNmtMf538o3rfsk1lvD7J1XR2ISOur0P0osPbG0huA/2oZrgLw5ZEWI+myvStm7JTnP31Jc+N"
    "0JJQDhhTZOaWqxPmzr2AB83eaLSWqpgBnuPnpzgMP0Q3qkQjGmGFO35FkvN0QiTDgsQXxqVT+seLM5KJ17h+H2+Gj16atqfD2cD9"
    "WzDrEAPxD48srrtbtcIrWkz9PuaezfqRanwj1HZWL8CKDPxkvASw3v2vwHYfCZ772KckH/xElpoa4HSLt8/MJqpNklxEDV6MIQgM"
    "y0gkfq4NlqRff2+yUsLEQdEQowX6EjFhjcvDZEJSHGoA5cNWBl0rL7GIoFOlDtqLo6yU9JzKQrhVmaST9xKSqTbQ2UlMTZaAWc51"
    "dEyf5APmS+o/xcwnMa3T4l1NWhtZgVCBv9L2Eq7FxP+VNeaIaaY+GJw0EtKo5zEsik39uBbFhzCIIMZfcjN+EAFXy0y6/AUiQjlj"
    "BSiI14q3AIhGxn9zMuPiPTMdbL+UNHNSTDOccJZJAR6P07T+5TSFtnXQhau7nooQr1cirtLYixb8KeYZRTqq4CAUXdI5gfq6P6jt"
    "3oJm5yqmTJ/gKn46uv3mtWMVKnKmflfLT8bmYcm5ZVtyN+T6g9zwraZpdmA11u/GszOiJkglV1Vr3MyTA9CNGAE="
)

CLAUDE_OPUS_5_5_MESSAGES_THINKING_SIGNATURE_0: Final = (
    "CAQSuAUKEggSEAIYAjgBQgh0aGlua2luZxIMuvOumb0X7WEvCG/pGgyEiubGiCWEMWg7YgQiMOhK4Mj8OKMMZOczi5roOTadgSLN"
    "L6MOk46J66LvPfihwIwowsxxkYa+EszhqUBYXSrTBIe+i63dRgUJd70m96/EnLB3GHAq+Q0ZhpE/USuwpJTc+MvI7KpuGtExRfnk"
    "OXzocJOw+R6FxqDO2dHHp00t9+I1gKCFUTqrSfKO0YZspMYfJGzSRjx5vx1equok7iV0k0CoQ8kMpRLhD1zTvf862EX8GpMcNkQs"
    "unrK/y7sQAhjPEDtSW4V8ipJF6E6RfJv5YOEsQKVx1kugmQEmItq4TLSMZd06Cip5CIkYrSsGhLYBfv5S2PFjXQGSOSkG9MDqjbI"
    "kPzA/1H38oIpdRjvQYMnrfAo+SaXuXybcUM36Svtj/+KCfG0A7Qb+Xhs2fT322lCpneyN1hXpqu6C0xGipHn2UDmIKxXcuOcet6J"
    "cCsjUB5BGf1H28hTtqZxunXUpZ9amTkDnyiXEEd7cwcdvFaqVd3/vRPJMSHLwbL8zY0VkWProBV54vW8GgPCFVWFw9B6LxjoPpbn"
    "7DiDlPPpQMik03iC3z2K3HPunID3Zx5LQ80nhueu0pec6rmWKijzQy/KVP5YRd8e2w2WpEqi8OwbJzFqUC7Eb8YIpjSs629/vhW2"
    "xJt09F0z2sbrs6JS8XZf0zzxWWLjoToWw/YKoJ/GOnJKxWJNI/4FutZ2Z79Xtdtk40DVQbjD4HT6MTw4r2z2UbnwkvACfUhqc5yE"
    "kH4lsmx5tSzh1kLfJEAmqwYrHDwYCvT3AUrl9f0EMPQiZpE0yPu1A52y4Ko/6uQWGHddGqtG96kLqb8BhPQR4p+XapjW/6MW6gmn"
    "kqrPPK6cl3CzcCy83Qojr1P3tt4CHu+bOeIYAQ=="
)

GEMINI_3_8_FLASH_MESSAGES_THOUGHT_SIGNATURE_0: Final = (
    "AY89a1++F/LsezPdJfrsXn9QhCwRsfuV/aT9QBFOa7bhIIgH0w67lP4bEkQBbUUZZ0V8XvQ3oFv8ZHuqJkVSGzArEgbhB2T5eu1o"
    "2zdg49SAfe/ys8JU77vwHhY630gFJq9fMFA4Eq60R97uaHZlDS2xp0chq32FIHizcI8fRVpRQf9i77gtdZ2UTaQ27C4ak0XzAsF8"
    "eYK5XWPD9mfJUj3qxwl0jJ3jVRo9ooqVHdkcRI6HXFazfuyDex3e/jd7AB9aeEQ7Ju4a1805jcm6UL7IUv1CPeGTuKnW5g/jFMho"
    "I5V8Zx2wLmP/IJimlPPvgwtyZ7CqnEEgiViLUUQoBZ9e2RlFP6SyfNs8QsyN+VjGklMM9mTZAGwVPbX2JofYNLoOXVpPQgEmH+lh"
    "PjomUTbYyuSMTrWZLBv+QpDAOs/oQaUP0SEX2V9aJHNqB/wLDN/4WygO0Aj3WRmXmJLOiwj8vl2asqtDNW4A2oePoZI4sI2Fh4uz"
    "6R/RqBfl4tTKEkAKs+3Eo3geZKaey8bNht+8abaudshgOhbNH9ON7YwM/S18LAcUXmcDWi3L6URw2BpJ/hmUIddsFeSNDJyW9sPj"
    "xGmLwMENxL2ETb/EsEnzgFfNMXmdE8JA7eWvFtc8E/TF1qO8+Z/nA9fOi3wtExL0i8TTfIuGPNaYvzS42c+9ZBtIRUa5UqFeJwXa"
    "NfHBKXZV+bsIvCLYFGHiaKReTP13KlgU2/pXCH608xM9t5F8Jqs3RSHnK3p4aDCH+9XS0pZd6j+FSiGSlDRf5xCPBV6S8ZE53547"
    "z7GWYJBeOGl61q/eEnMZ0PcGczGocHVLaIH0tNtlmSlploCdgdOCbiKH/k4ozDq8+XLhxS/LOfwOS+a6YwoZzEEsxXjwVy05h+qZ"
    "QtPU3EirPC38+Hky2bY7ZWh1Kp2PMBECnclLvTlN82rWAxfKV7693pLIAx7cyNy3aC5fU2aR0ByyB215evU8jpoe6LAnBKWdnRab"
    "vf4tvp6mZofYzFI+GWlCR/ZvH7rduHhm8L79EP0l6jgXSvkOX+kTcMj4gXB2Ty4SEZB8GfuMBhQeRrnIw5GZXl6llMrdF2fBshv5"
    "lEE/xJUvYZVCO6KiUXDz8eSoe3IpxbMPoL7zP0HEaz9jEKIo2zicb9tc2rNptkpBO8lq3XGsgsRS0OaF2HQhyDjmhfAeO3rv1o1o"
    "MwcHPRRL74cd8NhgT8u8Ub00tYb8u1xcgXRDcMKY/09/QMxvf0Fbp5wrAf2Mklc5ZGd1dqsPR45iFuI7Oq3yLsufEZ8Y51R0pe+p"
    "WjUskLdSMnbVdWhAzSevKY/0MQUhgx5BGkuguWWwsVrllJ+kGus++eDy4ti4Aowlc78brDFJ2S3ooC2h0OBh86w7725JPe/FL+q8"
    "ZkbGLlIwqr6YLEqb+xXsBO9ffwbg5LisXLD1+ZNtcEaecrEi2vcRVCKur5Kp0Wk="
)

GEMINI_3_1_PRO_PREVIEW_MESSAGES_THOUGHT_SIGNATURE_0: Final = (
    "AY89a1+Kk+qLPv2hT8v72PWRRnf/jCTgUnIw5y2JAFmJUzzsseqTRQA/L/XoGa+l56wDzylZtjpWPupaluVI2ih517CV2zWrLqCU"
    "JM/82tw2mp7qOvywPkjUuZL9Pe5qVnCFjQZYbZQIByZY0z75bq71b2gJigTfOQhsD/0LkvZUDUeyyihh1v/uhT33aUVwLCtXrJ1b"
    "vkWS8CG204JzwgFKAXzYoNV99iro/PN7REisc0rc3osleu93uRExX028ahy9vCvzXh8XFfgjMBTj+5O0clsl5EsF2wSnRiB3llfh"
    "ETYs+5v6VMHpYPXpg9M0T/S0UltgN8jDUxKVrroot7bjDHGDA1przgemEk4iJ/kkf3ZgEQS79Hyn/cF6OLnXuQikr3Mo3qYMS+S+"
    "sm9E62/xoPCuzP7vUAg8uzYzpNmqKECV+mx7bIb2jRe8fYEsE0ctZ1sc4mxjsXqLKFz/ZqGDraRpxVAp9YXNk4CVtNW7HsvF2yUC"
    "7E0Z68mR0AvUuIAizHHAasYgOAGVDYJrwTvR2cr7HQcxz1JoHWe9N09y1AOJI8qnaESryZo1NWatTgb/4ei32Zp4eB71u0S8Xq/I"
    "bRPfreLSDZe+UhNM0Qy2vscUZd6iR1NO0JY03GWxJFzSdjDJi2s3qJEIgtlWNayUIl1NbIzTDlCV2Ar1e5b9U4nJLvRb2g9dJPGP"
    "eFiNg6/wLaaNvgtxXqJNoYR8gFB2rY5NHBm23Po00vPsnb3qW7AEkOTSTc+3W5krqAsIWLYo+A17hs+kHaj1Q5kuQ0JAdZQHSLYe"
    "SJMvfMR/Cq06i+2sfXVbJYBnDqLrNXTBKY7zlat2S/Ku32s1qyi2dDnNxKNFbE7/z5xPJNSQE+icVG8dKKXKi9k9H5utqJQPuDc/"
    "mHYWCccrrEZa1A18E5TVVLpBlCeHILz80FB50WbJ8FZP2kUAzR1XlMWI/N6Td4+n1Ewl7kNneT9cBXtEoXZHyjNOCjEeKrZjB8C4"
    "LK6Ck6NmG4+1InP7dlVxiZ1uVKnnp7U2eXGNS4zU6fZ1wQbG37cAiqbXj/76f6ez4/w="
)

GEMINI_4_ARGON_MESSAGES_THOUGHT_SIGNATURE_0: Final = (
    "AY89a1/RSn20UE3dvcqrPsNoEs739i6LfLmFV/lDmMH140T7bWArO7hY0MDQce+DRbIBGU+d1ZkqSakUkfHGg2OQ3gNRavTI64N4"
    "7kw/3NpbO57RZq7o8VfnE4AqsFw+LnquTYD1JO+kT68ovHJXc+drutL5R32m5o0wu0gCxq2zAlc4qjiXWKWF+HTzaCGqx7xRygUD"
    "DPiPAodLqLhTNFNSSPs1zNDHALznl6CNHOZa2gNv8yN66BWn9+wHbt7WuEpACcQGJpfMjqHBoDDb0alSRNgO6r3+m1yIKQ+isZIJ"
    "MMybfP6dV2erAKPuoYC+YJHgAK0vi7K9aMJwySFB9syBoF2s6iBg20fZXomBn+iF1/W+vEw+Tf3wJ//WK9qnKkNkKk9OeKk3bN2B"
    "D7jEahual5cQzx5RPiDxB5DLPfxr5J+XZvXP3ZmB9V272LBT8sIPdmuc2kMt/1drXirh+z6ia8V/ePmOKcgyCBKSogFMWGFTydBh"
    "1sNz79aWiofiXlC0JSovAveQ+46nDm7D3bXPA2w0umz/ac9v/95suuGDq4ijo840DqqLLGXUu/ZSyUb++LqvSkg5YYdvc6sQQmPd"
    "ztygGH0dC6udIKvCuOYo+0ctUN/9HGSUUvjTBYT0kv7gXtrqe8cV7MPasGKHWYpaHspVu0k0qXQo4PdINS4lnetIIT+lV9L6sxvk"
    "KVpmjeE3ktitlE3GyjhyznzcB0DCNpGlWf13GXBokyv4OMtr6I+jYtE2TZRsSTBv6wEmNwnB50yWvBPSph86Qz5Yf4S68KQBEbwH"
    "bssQoZ3DgoG2sk+ybNbAEsBLCU+35/8iMk87/dngk7mbM+EaVaboyGBm9KXzh8Kk"
)


CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "vertex_ai/claude-sonnet-5-5@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-sonnet-5-5@default:rawPredict",
    expected_provider_headers={"authorization": "Bearer scripted-token", "content-type": "application/json"},
    expected_provider_request={
        **VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
        "messages": MESSAGES_REQUEST["messages"],
        "stream": False,
        "system": [
            {
                "type": "text",
                "text": (
                    "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                    "Explain which file you need next, then call a read-only tool. Never change files. Use the messages "
                    "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                    "the next file to inspect."
                ),
            },
        ],
        "tools": MESSAGES_REQUEST["tools"],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-sonnet-5-5",
        "id": "msg_vrtx_011Cg38paXLscBt8ARDQWjmd",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "Since the user asked for the validation file first, I'll focus there rather than reading tests in "
                    "parallel. I'll pull a range around line 18, say lines 1-60, to get enough context.\n\n"
                ),
                "signature": CLAUDE_SONNET_5_5_MESSAGES_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": (
                    "Next I'll check `shop/validation.py` to see how `validate_quantity` handles zero, negative, and "
                    "non-integer values, starting from the top of the file around line 18.\n\n"
                ),
                "signature": CLAUDE_SONNET_5_5_MESSAGES_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01UHVgrFsbouYKZ8mjVaM2wj",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 257,
            "output_tokens_details": {"thinking_tokens": 159},
        },
    },
    expected_litellm_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "vertex_ai/claude-sonnet-5-5@default",
        "id": "msg_vrtx_011Cg38paXLscBt8ARDQWjmd",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "Since the user asked for the validation file first, I'll focus there rather than reading tests in "
                    "parallel. I'll pull a range around line 18, say lines 1-60, to get enough context.\n\n"
                ),
                "signature": CLAUDE_SONNET_5_5_MESSAGES_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": (
                    "Next I'll check `shop/validation.py` to see how `validate_quantity` handles zero, negative, and "
                    "non-integer values, starting from the top of the file around line 18.\n\n"
                ),
                "signature": CLAUDE_SONNET_5_5_MESSAGES_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01UHVgrFsbouYKZ8mjVaM2wj",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 257,
            "output_tokens_details": {"thinking_tokens": 159},
        },
    },
)


CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "vertex_ai/claude-opus-5-5@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-5-5@default:rawPredict",
    expected_provider_headers={"authorization": "Bearer scripted-token", "content-type": "application/json"},
    expected_provider_request={
        **VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
        "messages": MESSAGES_REQUEST["messages"],
        "stream": False,
        "system": [
            {
                "type": "text",
                "text": (
                    "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                    "Explain which file you need next, then call a read-only tool. Never change files. Use the messages "
                    "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                    "the next file to inspect."
                ),
            },
        ],
        "tools": MESSAGES_REQUEST["tools"],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_vrtx_011Cg38pLy1pX8GQEfjq6NFo",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    'Since the user said "first," I\'ll focus on reading the validation file around line 18 before moving '
                    "on to the test file.\n\n"
                ),
                "signature": CLAUDE_OPUS_5_5_MESSAGES_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": (
                    "I'll start with `shop/validation.py`. `validate_quantity` is defined at line 18, so I'll read the "
                    "lines around it. That should show how zero, negative and non-integer quantities are handled."
                ),
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_018jLLjmUz5DQCg8xZ76kMFY",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 50},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 233,
            "output_tokens_details": {"thinking_tokens": 70},
        },
    },
    expected_litellm_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "vertex_ai/claude-opus-5-5@default",
        "id": "msg_vrtx_011Cg38pLy1pX8GQEfjq6NFo",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    'Since the user said "first," I\'ll focus on reading the validation file around line 18 before moving '
                    "on to the test file.\n\n"
                ),
                "signature": CLAUDE_OPUS_5_5_MESSAGES_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": (
                    "I'll start with `shop/validation.py`. `validate_quantity` is defined at line 18, so I'll read the "
                    "lines around it. That should show how zero, negative and non-integer quantities are handled."
                ),
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_018jLLjmUz5DQCg8xZ76kMFY",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 50},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 233,
            "output_tokens_details": {"thinking_tokens": 70},
        },
    },
)


CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "vertex_ai/claude-fable-5-1@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-fable-5-1@default:rawPredict",
    expected_provider_headers={"authorization": "Bearer scripted-token", "content-type": "application/json"},
    expected_provider_request={
        **VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
        "messages": MESSAGES_REQUEST["messages"],
        "stream": False,
        "system": [
            {
                "type": "text",
                "text": (
                    "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                    "Explain which file you need next, then call a read-only tool. Never change files. Use the messages "
                    "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                    "the next file to inspect."
                ),
            },
        ],
        "tools": MESSAGES_REQUEST["tools"],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_vrtx_011Cg38pBu4rsBNuTwSrvoze",
        "content": [
            {
                "type": "text",
                "text": "I'll read the validation function to see how zero, negative, and non-integer quantities are handled.",
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01SNoqWEthDpRET4kp9iGsXB",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 129,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
    expected_litellm_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "vertex_ai/claude-fable-5-1@default",
        "id": "msg_vrtx_011Cg38pBu4rsBNuTwSrvoze",
        "content": [
            {
                "type": "text",
                "text": "I'll read the validation function to see how zero, negative, and non-integer quantities are handled.",
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01SNoqWEthDpRET4kp9iGsXB",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 129,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
)


GEMINI_3_8_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "vertex_ai/gemini-3.8-flash"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-3.8-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST,
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                        "Explain which file you need next, then call a read-only tool. Never change files. Use the messages "
                        "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                        "the next file to inspect."
                    ),
                },
            ],
        },
    },
    mock_provider_response={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": (
                                "**My Thought Process on Reviewing Quantity Validation**\n\nMy primary objective is to understand how "
                                "the system handles zero, negative, and non-integer quantities, as this is crucial for ensuring the "
                                "integrity of our synthetic Python project. The user's initial request directly guides me to the "
                                '"validation file first so we can understand how zero and negative quantities are handled."\n\nIn '
                                "alignment with my system instructions, I must first identify the next relevant file, explain why "
                                "I'm choosing it, and then use a read-only tool to inspect it. I also need to explicitly keep in "
                                "mind the nuances of zero, negative, and non-integer quantities as I proceed.\n\nTherefore, the "
                                "immediate next step is to examine the validation logic. Based on this, I've deduced that "
                                "`shop/validation.py` is the file I need to inspect. This file is likely to contain the core logic "
                                "for validating quantities, and I anticipate it will provide the answers regarding how these "
                                "specific edge cases (zero, negative, and non-integer values) are managed.\n\nTo get a comprehensive "
                                "view of this validation mechanism, I will focus on the initial section of the file, particularly "
                                "around the function `validate_quantity(quantity)`, which I recall is located at line 18. I'll "
                                "request to read lines 1 through 40 to get a good understanding of the surrounding context and the "
                                "validation function itself.\n\n"
                            ),
                            "thought": True,
                        },
                        {
                            "text": (
                                "I need to read `shop/validation.py` next to see how `validate_quantity` validates incoming quantity "
                                "values, including checks for zero, negative, and non-integer quantities."
                            ),
                        },
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"end_line": 40, "path": "shop/validation.py", "start_line": 1},
                                "id": "call_190093",
                            },
                            "thoughtSignature": GEMINI_3_8_FLASH_MESSAGES_THOUGHT_SIGNATURE_0,
                        },
                    ],
                },
                "finishReason": "STOP",
            },
        ],
        "usageMetadata": {
            "promptTokenCount": 242,
            "candidatesTokenCount": 71,
            "totalTokenCount": 556,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 242}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 71}],
            "thoughtsTokenCount": 243,
        },
        "modelVersion": "gemini-3.8-flash",
        "createTime": "2026-10-15T01:13:45.750641Z",
        "responseId": "ySjQarHoLb2apucPwee_YQ",
    },
    expected_litellm_response={
        **MESSAGES_BRIDGE_RESPONSE,
        "id": "ySjQarHoLb2apucPwee_YQ",
        "model": "vertex_ai/gemini-3.8-flash",
        "usage": {"input_tokens": 242, "output_tokens": 314},
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "**My Thought Process on Reviewing Quantity Validation**\n\nMy primary objective is to understand how "
                    "the system handles zero, negative, and non-integer quantities, as this is crucial for ensuring the "
                    "integrity of our synthetic Python project. The user's initial request directly guides me to the "
                    '"validation file first so we can understand how zero and negative quantities are handled."\n\nIn '
                    "alignment with my system instructions, I must first identify the next relevant file, explain why "
                    "I'm choosing it, and then use a read-only tool to inspect it. I also need to explicitly keep in "
                    "mind the nuances of zero, negative, and non-integer quantities as I proceed.\n\nTherefore, the "
                    "immediate next step is to examine the validation logic. Based on this, I've deduced that "
                    "`shop/validation.py` is the file I need to inspect. This file is likely to contain the core logic "
                    "for validating quantities, and I anticipate it will provide the answers regarding how these "
                    "specific edge cases (zero, negative, and non-integer values) are managed.\n\nTo get a comprehensive "
                    "view of this validation mechanism, I will focus on the initial section of the file, particularly "
                    "around the function `validate_quantity(quantity)`, which I recall is located at line 18. I'll "
                    "request to read lines 1 through 40 to get a good understanding of the surrounding context and the "
                    "validation function itself.\n\n"
                ),
                "signature": "",
            },
            {
                "type": "text",
                "text": (
                    "I need to read `shop/validation.py` next to see how `validate_quantity` validates incoming quantity "
                    "values, including checks for zero, negative, and non-integer quantities."
                ),
            },
            {
                "type": "tool_use",
                "id": "call_190093",
                "name": "read_file",
                "input": {"end_line": 40, "path": "shop/validation.py", "start_line": 1},
                "provider_specific_fields": {"signature": GEMINI_3_8_FLASH_MESSAGES_THOUGHT_SIGNATURE_0},
            },
        ],
    },
)


GEMINI_3_1_PRO_PREVIEW_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "vertex_ai/gemini-3.1-pro-preview"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST,
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                        "Explain which file you need next, then call a read-only tool. Never change files. Use the messages "
                        "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                        "the next file to inspect."
                    ),
                },
            ],
        },
    },
    mock_provider_response={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": (
                                "**My Thought Process on Quantities in the Shop Module**\n\nMy immediate goal here is to gain a "
                                "precise understanding of how the `shop` module, specifically within its `validation.py` file, "
                                "handles the input of item quantities, with a particular focus on zero and negative values. This is "
                                "a critical area for ensuring data integrity and preventing potential issues downstream in any "
                                "e-commerce or inventory system.\n\nTo achieve this, I need to examine the source code directly. I've "
                                "identified that the relevant logic for quantity validation is likely contained within a function "
                                "named `validate_quantity`. Based on my indexing or prior knowledge of this codebase, I know this "
                                "function resides in `shop/validation.py`. I'll proceed by reading a section of this file that "
                                "encompasses the `validate_quantity` function. I'm aiming for a contextually rich view, so I'll "
                                "fetch lines from around where I expect the function to start. My tool call will be to `read_file`, "
                                'specifying the path as `"shop/validation.py"`. To ensure I capture the function definition and its '
                                "immediate surrounding logic, I'll set `start_line` to `1` and `end_line` to `30`. This range should "
                                "provide me with sufficient context to analyze the function's implementation.\n\nOnce I have the "
                                "content of `shop/validation.py` for those lines, I will meticulously review the `validate_quantity` "
                                "function to observe how it explicitly checks for and handles zero and negative quantity inputs. "
                                "This will involve looking for conditional statements (`if`, `elif`, `else`) and any associated "
                                "error-raising mechanisms or default behaviors.\n\n"
                            ),
                            "thought": True,
                        },
                        {
                            "text": (
                                "I will read the `shop/validation.py` file to see how `validate_quantity` handles zero and negative "
                                "quantities."
                            ),
                        },
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"start_line": 1, "end_line": 35, "path": "shop/validation.py"},
                                "id": "call_1455553",
                            },
                            "thoughtSignature": GEMINI_3_1_PRO_PREVIEW_MESSAGES_THOUGHT_SIGNATURE_0,
                        },
                    ],
                },
                "finishReason": "STOP",
            },
        ],
        "usageMetadata": {
            "promptTokenCount": 242,
            "candidatesTokenCount": 59,
            "totalTokenCount": 499,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 242}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 59}],
            "thoughtsTokenCount": 198,
        },
        "modelVersion": "gemini-3.1-pro-preview",
        "createTime": "2026-10-15T01:13:40.675781Z",
        "responseId": "xCjQasWfKcyL9LsPx7HAqQ8",
    },
    expected_litellm_response={
        **MESSAGES_BRIDGE_RESPONSE,
        "id": "xCjQasWfKcyL9LsPx7HAqQ8",
        "model": "vertex_ai/gemini-3.1-pro-preview",
        "usage": {"input_tokens": 242, "output_tokens": 257},
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "**My Thought Process on Quantities in the Shop Module**\n\nMy immediate goal here is to gain a "
                    "precise understanding of how the `shop` module, specifically within its `validation.py` file, "
                    "handles the input of item quantities, with a particular focus on zero and negative values. This is "
                    "a critical area for ensuring data integrity and preventing potential issues downstream in any "
                    "e-commerce or inventory system.\n\nTo achieve this, I need to examine the source code directly. I've "
                    "identified that the relevant logic for quantity validation is likely contained within a function "
                    "named `validate_quantity`. Based on my indexing or prior knowledge of this codebase, I know this "
                    "function resides in `shop/validation.py`. I'll proceed by reading a section of this file that "
                    "encompasses the `validate_quantity` function. I'm aiming for a contextually rich view, so I'll "
                    "fetch lines from around where I expect the function to start. My tool call will be to `read_file`, "
                    'specifying the path as `"shop/validation.py"`. To ensure I capture the function definition and its '
                    "immediate surrounding logic, I'll set `start_line` to `1` and `end_line` to `30`. This range should "
                    "provide me with sufficient context to analyze the function's implementation.\n\nOnce I have the "
                    "content of `shop/validation.py` for those lines, I will meticulously review the `validate_quantity` "
                    "function to observe how it explicitly checks for and handles zero and negative quantity inputs. "
                    "This will involve looking for conditional statements (`if`, `elif`, `else`) and any associated "
                    "error-raising mechanisms or default behaviors.\n\n"
                ),
                "signature": "",
            },
            {
                "type": "text",
                "text": (
                    "I will read the `shop/validation.py` file to see how `validate_quantity` handles zero and negative "
                    "quantities."
                ),
            },
            {
                "type": "tool_use",
                "id": "call_1455553",
                "name": "read_file",
                "input": {"start_line": 1, "end_line": 35, "path": "shop/validation.py"},
                "provider_specific_fields": {"signature": GEMINI_3_1_PRO_PREVIEW_MESSAGES_THOUGHT_SIGNATURE_0},
            },
        ],
    },
)


GEMINI_4_ARGON_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "vertex_ai/gemini-4-argon"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-4-argon:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST,
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                        "Explain which file you need next, then call a read-only tool. Never change files. Use the messages "
                        "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                        "the next file to inspect."
                    ),
                },
            ],
        },
    },
    mock_provider_response={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": (
                                "**My Thought Process on Handling Quantity Validation**\n\nMy primary objective is to meticulously "
                                "understand how the system currently manages the validation of product quantities, specifically "
                                "focusing on how it addresses zero and negative values. To achieve this, I need to first dive into "
                                "the `shop/validation.py` file. This will provide me with the foundational knowledge of the existing "
                                "logic and the rules that govern quantity inputs.\n\nFollowing the system's instructions, my next step "
                                "is to explicitly state which file I require for this understanding and then invoke the `read_file` "
                                "tool to access `shop/validation.py`. I must also keep in mind the overarching directive to consider "
                                "zero, negative, and non-integer quantities throughout this process, as this will be crucial when I "
                                "eventually propose any solutions or fixes. It's important to remember that I am not to alter any "
                                "files, only to read and analyze them, and to communicate my progress and needs via the provided "
                                "messages client.\n\n"
                            ),
                            "thought": True,
                        },
                        {
                            "text": (
                                "I will read `shop/validation.py` next to inspect how `validate_quantity` handles zero, negative, "
                                "and non-integer quantities."
                            ),
                        },
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"end_line": 50, "start_line": 1, "path": "shop/validation.py"},
                                "id": "call_258575",
                            },
                            "thoughtSignature": GEMINI_4_ARGON_MESSAGES_THOUGHT_SIGNATURE_0,
                        },
                    ],
                },
                "finishReason": "STOP",
            },
        ],
        "usageMetadata": {
            "promptTokenCount": 412,
            "candidatesTokenCount": 69,
            "totalTokenCount": 604,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 412}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 69}],
            "thoughtsTokenCount": 123,
        },
        "modelVersion": "gemini-4-argon",
        "createTime": "2026-10-15T01:13:52.985186Z",
        "responseId": "0CjQauKQPNyFpucPldSPiAY",
    },
    expected_litellm_response={
        **MESSAGES_BRIDGE_RESPONSE,
        "id": "0CjQauKQPNyFpucPldSPiAY",
        "model": "vertex_ai/gemini-4-argon",
        "usage": {"input_tokens": 412, "output_tokens": 192},
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "**My Thought Process on Handling Quantity Validation**\n\nMy primary objective is to meticulously "
                    "understand how the system currently manages the validation of product quantities, specifically "
                    "focusing on how it addresses zero and negative values. To achieve this, I need to first dive into "
                    "the `shop/validation.py` file. This will provide me with the foundational knowledge of the existing "
                    "logic and the rules that govern quantity inputs.\n\nFollowing the system's instructions, my next step "
                    "is to explicitly state which file I require for this understanding and then invoke the `read_file` "
                    "tool to access `shop/validation.py`. I must also keep in mind the overarching directive to consider "
                    "zero, negative, and non-integer quantities throughout this process, as this will be crucial when I "
                    "eventually propose any solutions or fixes. It's important to remember that I am not to alter any "
                    "files, only to read and analyze them, and to communicate my progress and needs via the provided "
                    "messages client.\n\n"
                ),
                "signature": "",
            },
            {
                "type": "text",
                "text": (
                    "I will read `shop/validation.py` next to inspect how `validate_quantity` handles zero, negative, "
                    "and non-integer quantities."
                ),
            },
            {
                "type": "tool_use",
                "id": "call_258575",
                "name": "read_file",
                "input": {"end_line": 50, "start_line": 1, "path": "shop/validation.py"},
                "provider_specific_fields": {"signature": GEMINI_4_ARGON_MESSAGES_THOUGHT_SIGNATURE_0},
            },
        ],
    },
)
