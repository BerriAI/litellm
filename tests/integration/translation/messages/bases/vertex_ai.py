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
CLAUDE_HAIKU_5_5_MESSAGES_THINKING_SIGNATURE: Final = (
    "CAQSjAYKEggUEAIYAjgBQgh0aGlua2luZxIMrSidazk3ktKpmQpeGgwTA2lAt2W83VRPLlAiMLq3TMv2CGOoAe6yCDeUOYr5gxqn"
    "OP14sWSd/KMcxDo3NLY6L+VrGJDE+Z1zY6kh8iqnBR3OjHaDlw6lZb9x0AUegPN0SWaw2zxbK6mJm/qqBxLGaUaX06VTH/53MgTq"
    "DmKTVtivawDDuwu95wf9Mk1O22SHCBP135bJL5tkHsK2kNQFDUDmB/sT4Qc1Rf4uI3GXvIxFT6l5PyQaJLflemgTF5tWVMOAuoHh"
    "8QL/6ZilKWwOH5pQf+3L0KP3w76FXi3c9GXMSXC3by7+mkKat0A9qdMyv/NVEMRBK33/tBlqgmZUVyWvTVqnL847Zdpckg1PiNYI"
    "1IAgUh3hlAnoGjwBmDZqlqVZaXV/Z5Kk8KDRXM4AwdtXzAZUmEtOLr21kK0Wp9vnrEaz0ApjxVkYAjijQsltVZpb179/LyK82j/F"
    "HWRjEGEuMVt4WF1AEA/D8gR1CREqvZ432bSE4lhXSDPbH/XeCs17gw06sGD548jQEfoknMtZ2h8F/O+BeLe3TPdLchmkKipAqNmg"
    "eJGHgMVG7b/9Scf+U0FEiDj29hLN0XyvaR58jKAk2GypZR3b4Sux6HIgmavD9jnErPj4whfLfGHCJSoy/xL3V+NZHH+TZrCk7pnI"
    "1ZrYU1F6B2qvuhElCGdf4nhjqkBJyOn3ce2DmrMiuVv0JYdy3GQqpraQB5QIneOInewshn/hXwvnIc0xXoVu+JVWwVYCxq2CkksV"
    "7IwL1iMu4lc1ZedE2+Bvf18dRpn5rMmYBbQNZURn4NuJN2vMvtf8OHf/7HGqGd6YDFbF/kKVSBBXjGPXw8usi6D/QNiP6s59PwEZ"
    "1FtjbfNm2K4rSCB81kCsYCtIGwNONWFxUNbYBW3/YOoccMeCxZR263cPyHvO1+L9NGuhwv8kwiZRMfuIQUYpKjMTb3C4M0ilZblG"
    "qfJ9nYJcJBhXTiiapuoCGxxARXiXiJyq5hyMhDlP2A26z00YAQ=="
)

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
        "id": "msg_vrtx_011ChJAfB5i24QvdEdakXtBm",
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
            "speed": "standard",
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": "msg_vrtx_011ChJAfB5i24QvdEdakXtBm",
        "type": "message",
        "role": "assistant",
        "model": "vertex_ai/claude-haiku-4-5@20251001",
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 17,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 5,
            "speed": "standard",
        },
        "container": None,
        "stop_details": None,
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
        "id": "msg_vrtx_011ChJAfA4BznmhBWM3F3gsc",
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
            "speed": "standard",
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": "msg_vrtx_011ChJAfA4BznmhBWM3F3gsc",
        "type": "message",
        "role": "assistant",
        "model": "vertex_ai/claude-sonnet-4-6@default",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 18,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 5,
            "speed": "standard",
        },
        "container": None,
        "stop_details": None,
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
        "id": "msg_vrtx_011ChJAfB6hQq1QVZW3r6Gn3",
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
            "speed": "standard",
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": "msg_vrtx_011ChJAfB6hQq1QVZW3r6Gn3",
        "type": "message",
        "role": "assistant",
        "model": "vertex_ai/claude-sonnet-5@default",
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 21,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "speed": "standard",
        },
        "container": None,
        "stop_details": None,
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
        "id": "msg_vrtx_011ChJAfFsSeXBjLjmDPH9du",
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
            "speed": "standard",
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": "msg_vrtx_011ChJAfFsSeXBjLjmDPH9du",
        "type": "message",
        "role": "assistant",
        "model": "vertex_ai/claude-opus-4-8@default",
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 21,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "speed": "standard",
        },
        "container": None,
        "stop_details": None,
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
    "CAQS5AYKEggUEAIYAjgBQgh0aGlua2luZxIM5Ic93LYhlIV0dILnGgzgvmhuTreTM7oumCkiMEKbDCSbKoQibv/gIcGyAtJvAlmn"
    "fwS8XTYDol4h20ez3WlwSrCR6eJG3MA8VYA6/Sr/BQRO9/lngvQeuARADgGAVlXdOK0Fa1Csviaz9c/Z4lMrLzQD69m0Y/PypX6l"
    "4G36x8wqjRZZCEefLI+3KOfN7mnFFiPPYlVsSejNC7oW0AY77X0RkEWjkFo1QNHKcPtIXarkh5RlazKYYLaZZTEb7o0LJshTPVwb"
    "WY/fUWNi7M3pc6kSpkZzwpJIqkw2kDUKCRwvY1fdzCNkOZueIlLSGUH+D/gQU/+9rt64PoSH19vRuqcsCygA+KluPl6PuszdsBtM"
    "Ru7t4t0qdSTnhZrykNuVlL2f/Efp2StZaZcwiK/HF+NAEw4y+Zvw1JhVNLw68kwa5WCiiVyGz6Ezu747FDvsXZN7+lAidLb4Uufb"
    "aSECQO2fNStIfviXbDCuUULvxYgf1h381+4JDZd6zt7SwrG2Is+nZJR23XjJte62fvXof+Suy42N9vF2VEQYPABcO1gkXJ1KLbwm"
    "izT5IMNqc1fLn1PVHTSUFsE5E2045DWs+nYN58GEKpdFRVu5jPq7N2XShZZpsORJ/NNBHrn0KCanRESgnb3ZS5X+g+h7sJtRP2pV"
    "evrupAGGe1rZHvcMEBc/SRc+yAzn6lOEh+3om1+2b8BPilsBaj20Sfy5PL0XXz9a/DJIOl/3Bhbpnw+sRQEC6AGqJHik9qGmIx9+"
    "i/VHj45cMTfOIbpME1QPoQ7xLD5hQPHGOKNc71NF9eqb/09T/628+ffTRAuUDlnFTWLoUsO5NQ16/K/ly7XRnDz+ftXypUU1zrcP"
    "HxAD+IxqvHwc1CfOGmQpa/UXk+HoJ8gDHuDsyjnah0VLs6YqU0P32Txud7/CXiBYbSWQYeh3xm9N5HaHlN/msTO0CkK8KND5iu2G"
    "O9ckPz/cicpz/QUCan2RWcPHDiRohVvBRsg7jonnsxmKrY9d1HlzW1o/ILDI9Jcw+EX0aBwP0OKoNCZMfyV0K5MwsCzcLUstWOgE"
    "sUiSHmvUstUANlKjQ0lNoKJEIQwnP2NeqBK9TBYSbAR0RmzAOWaltAK171yrS6QwGAE="
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
    "CAQS2gQKEggUEAIYAjgBQgh0aGlua2luZxIM36PQDmg4s3sKb0cHGgwhJpsNmEMbfgiRK70iMBlG0eVS8MsuqvDWDoyz9tiLAiKo"
    "FuJLBL/cFzZ4b/vE6GNPqRdxKdJbxRDojG3CTir1A6DpayBeRwdqN1nJRahs1oR6pMJ+Fi/QxaNIPMlJEJ06l7CPWM7FhOPlgiTL"
    "Z7BpiOIEEzrZsV9kXtkar1H54r6wuI2R95ZZ2hqE1TyZhXvLO9Hl/xd9wBR+JbOzkEa6k37j8mxybLKR87+/NE0BS25Vf2x11Tt2"
    "cO4SnIBnrJs1LDOqAOoucz94+h9faxI9WI8/hBaVoNv+u7XV9DLF5QttePYigcvAyofp6T3Pj71vq1FHCG1IMntU2WvmnGzD6RqO"
    "JAeU6wz0dCLmaI+Q5+SlE+1cwOB+9aQgNGuPRFT244/F27fkFbqwDE0Qn3+2ZsSs9xsCt15x0scKBSmku7jwT6bQGgcEAFJuvfbv"
    "Kg3i0n7RlgQNnvEI88tefNY3GAl0MzcaXY3txfBvkOiYP84saKZsZxSWj9pWvlTOoTcQE/WvfNeGmMtp7Ekun6BvRXoIXpKlDxNP"
    "Azym6zJTFoFefV7Fi4jrOWEbdD5w4u3vJmgidkO+sH4VbxQGvdGFr1YwyqifZoBM04I6pQWgyOruoPSjcZiW3GAZH+t5fZICy6Wa"
    "MD6NIQW5TV8yIgni5uQFO+34tj9BcaaxEpCX0DGDZNjqkg3370KUv5jthf9Y5aANyfNlXaqowZ3b1ZuhHxeJTdVK/UbnAghG9i6j"
    "8+qhqEzr4hgB"
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
            }
        ],
        "tools": MESSAGES_REQUEST["tools"],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-sonnet-5-5",
        "id": "msg_vrtx_011ChJAfLXjaofuNiBirf4mD",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "I'm going to read the validation file first, since that's what the user asked for. There's also "
                    "a mention of a messages client tool that isn't available to me, so I'll leave that aside and not"
                    "e it in my explanation.\n\n"
                ),
                "signature": CLAUDE_SONNET_5_5_MESSAGES_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": (
                    "I need `shop/validation.py` next. `validate_quantity` is defined there, so it should show how ze"
                    "ro, negative and non-integer quantities are treated. I'll read the area around line 18."
                ),
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01HVKAmTJu1Cj5t4W9s1ysy7",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 278,
            "output_tokens_details": {"thinking_tokens": 119},
            "speed": "standard",
        },
    },
    expected_litellm_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "id": "msg_vrtx_011ChJAfLXjaofuNiBirf4mD",
        "model": "vertex_ai/claude-sonnet-5-5@default",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'm going to read the validation file first, since that's what the user asked for. There's also a mention of a messages client tool that isn't available to me, so I'll leave that aside and note it in my explanation.\n\n",
                "signature": CLAUDE_SONNET_5_5_MESSAGES_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I need `shop/validation.py` next. `validate_quantity` is defined there, so it should show how zero, negative and non-integer quantities are treated. I'll read the area around line 18.",
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01HVKAmTJu1Cj5t4W9s1ysy7",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 278,
            "output_tokens_details": {"thinking_tokens": 119},
            "speed": "standard",
        },
    },
)


CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "vertex_ai/claude-haiku-5-5@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-haiku-5-5@default:rawPredict",
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
            }
        ],
        "tools": MESSAGES_REQUEST["tools"],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-haiku-5-5",
        "id": "msg_vrtx_011ChJAfLmcvdEELdVb3Yerp",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "I'm going to read shop/validation.py, the file the user asked for, and I'll take lines 1 to 60 s"
                    "ince I don't know its length. The system note says to keep zero, negative, and non-integer quant"
                    "ities in mind before picking the next file, so I'll check that as I read.\n\n"
                ),
                "signature": CLAUDE_HAIKU_5_5_MESSAGES_THINKING_SIGNATURE,
            },
            {
                "type": "text",
                "text": (
                    "I'll read `shop/validation.py`, starting with the top of the file and the `validate_quantity` fu"
                    "nction at line 18, to see how zero, negative, and non-integer quantities are handled."
                ),
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01Xjr9GFueWkfoqG6jQjDuJm",
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
            "output_tokens_details": {"thinking_tokens": 98},
            "speed": "standard",
        },
    },
    expected_litellm_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "id": "msg_vrtx_011ChJAfLmcvdEELdVb3Yerp",
        "model": "vertex_ai/claude-haiku-5-5@default",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'm going to read shop/validation.py, the file the user asked for, and I'll take lines 1 to 60 since I don't know its length. The system note says to keep zero, negative, and non-integer quantities in mind before picking the next file, so I'll check that as I read.\n\n",
                "signature": CLAUDE_HAIKU_5_5_MESSAGES_THINKING_SIGNATURE,
            },
            {
                "type": "text",
                "text": "I'll read `shop/validation.py`, starting with the top of the file and the `validate_quantity` function at line 18, to see how zero, negative, and non-integer quantities are handled.",
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01Xjr9GFueWkfoqG6jQjDuJm",
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
            "output_tokens_details": {"thinking_tokens": 98},
            "speed": "standard",
        },
    },
)


CLAUDE_OPUS_5_5_MESSAGES_CAPTURE_SIGNATURE_1: Final = (
    "CAQS4QUKEwgUEAIYAjgBQgluYXJyYXRpb24SDI0mrtSUHHRYJ4daxhoMV4MVPPqMgdN7ggcPIjBH2+6CD84m+2t2Y06SxuHZVa6z"
    "Qlk+k5HiYZMItQAaA6L9IIMXLvzTLQyMy2iROuIq+wR2BAlW+FUnon0LhHGtjbQaQWF046gb1+dMe2gvaeaRi96TVyj2JPUYHJKj"
    "AnNqeUx+QX6kgNbv7F7SEDfkwzAwLZEFXBa078iQCrK9IqDRQL+QC3IUvW0x3/+sZB/L3aL9VjnV59u2LAnh99wo42LKP3an5HbU"
    "d8BkNNPaYoS/xO3wvtCrm2yrSxX0i+u7c9FQKjEV0p8VfoMfYZ40x5nzMwn9ikNngZ4aHSEs2AVQ4fh5TYlgf/NStjJW1DAHIcs1"
    "K7ExehgoOAsKL36CSxBIfJp4bvxzaPkjmu3ts1s8VoYAEHw5oZtYpJ7kPufejDiA/HOcdK1NKp2608mX4A93m7bq68rdFVxGvjZZ"
    "n4/OJReRm0bSMp6iYPisfi6q+nXuxb/bIknZTm/piTMli44W9L7Wu4FQBmaoiisN/1l8aeGQYCsNoBCwi3tqHcJujxfWJvRK2lcq"
    "xlHkKDffPRo+X7dx6/zIljQGX9DNR2ke4gJziBLD5XaEc/SsHCpkGNgDzGgQY00FwZ93ELuBwisyDjQP+PYUh68bSX727Ej9JY0A"
    "M0QnTeDRBEtw8nIa2pSwqFXw7mP/c2kayd5Eh6tREQo6XCqzYWdOtkhd0wohTaPH0VVNFhYFizwBeGe3yHGorMAkBCqMj28Pmgub"
    "shkGQn8gUneS0aVgyClhYc3CdBNlQN/HLJc52l39GG357+Hd6RVqWzurkPCi+r23ET4+hMP/nyI6reqLkjEYAWBWNn0Uj+/O9VNG"
    "KtJmqltoFhGdQsJDyN63mR9j25bGxFizrliRdk1iAQ27Hz0Ny6FEY9eqgEpeg9AkJbQK3VR76Lc5axTOEB+l+uYSqxgB"
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
            }
        ],
        "tools": MESSAGES_REQUEST["tools"],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_vrtx_011ChJAfNM8Tcg88dTJwHEJE",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "I'm reading the validation file first since the user asked for that, even though the tests don't"
                    " depend on each other. Once that's done, I'll explain which file should come next.\n\n"
                ),
                "signature": CLAUDE_OPUS_5_5_MESSAGES_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": (
                    "I'm reading `validate_quantity` at line 18 in `shop/validation.py` to see how it handles zero, n"
                    "egative, float, string and boolean inputs.\n\n"
                ),
                "signature": CLAUDE_OPUS_5_5_MESSAGES_CAPTURE_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01UnVtkxjAvqa8NEnCKXqC4H",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 216,
            "output_tokens_details": {"thinking_tokens": 43},
            "speed": "standard",
        },
    },
    expected_litellm_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "id": "msg_vrtx_011ChJAfNM8Tcg88dTJwHEJE",
        "model": "vertex_ai/claude-opus-5-5@default",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'm reading the validation file first since the user asked for that, even though the tests don't depend on each other. Once that's done, I'll explain which file should come next.\n\n",
                "signature": CLAUDE_OPUS_5_5_MESSAGES_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "I'm reading `validate_quantity` at line 18 in `shop/validation.py` to see how it handles zero, negative, float, string and boolean inputs.\n\n",
                "signature": CLAUDE_OPUS_5_5_MESSAGES_CAPTURE_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01UnVtkxjAvqa8NEnCKXqC4H",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 216,
            "output_tokens_details": {"thinking_tokens": 43},
            "speed": "standard",
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
