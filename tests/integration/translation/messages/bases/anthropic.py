from collections.abc import Mapping
from typing import Final

from integration.translation.case import TranslationTestCase
from integration.translation.messages.templates import MESSAGES_REQUEST, MESSAGES_RESPONSE
from integration.translation.templates.anthropic import ANTHROPIC_MESSAGES_REQUEST, ANTHROPIC_MESSAGES_RESPONSE
from pydantic import JsonValue

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "anthropic/claude-opus-5-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-opus-5-5",
        "max_tokens": 64,
        "stream": False,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-opus-5-5",
        "id": "msg_011CfgC5HKvyve78CTFAw97f",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 23,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "model": "anthropic/claude-opus-5-5",
        "id": "msg_011CfgC5HKvyve78CTFAw97f",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 23,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "anthropic/claude-sonnet-4-6",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "stream": False,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-4-6",
        "id": "msg_011CffzUNHaEfzVxCh5hskBG",
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
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "model": "anthropic/claude-sonnet-4-6",
        "id": "msg_011CffzUNHaEfzVxCh5hskBG",
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
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
)

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "anthropic/claude-haiku-4-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-haiku-4-5",
        "max_tokens": 64,
        "stream": False,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_011CfgDz9Vj8wXvKzhoM4MQA",
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
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "model": "anthropic/claude-haiku-4-5",
        "id": "msg_011CfgDz9Vj8wXvKzhoM4MQA",
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
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "anthropic/claude-sonnet-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-sonnet-5",
        "max_tokens": 64,
        "stream": False,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_011CfgDzLXmJmnHRTpeBoDmZ",
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
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "model": "anthropic/claude-sonnet-5",
        "id": "msg_011CfgDzLXmJmnHRTpeBoDmZ",
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
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "anthropic/claude-opus-4-8",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-opus-4-8",
        "max_tokens": 64,
        "stream": False,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
    },
    mock_provider_response={
        "model": "claude-opus-4-8",
        "id": "msg_011CfgDzSFYYPC3aYE9rH98J",
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
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "model": "anthropic/claude-opus-4-8",
        "id": "msg_011CfgDzSFYYPC3aYE9rH98J",
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
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSjggKEAgSGAI4AUIIdGhpbmtpbmcSDH0nWsNq6ztJU4FmSRoMNF3KUUnzhPIVkxnLIjBUD6huzC4a6KowV1auL/9ovi4lnt+8VcIkUJat"
    "qASbHEWK6cwcKYp/QaSkeXlRJd8qqwdMqLSP3zIGOAvSrDbirhKmZzy9H6y84DP1WIscCMCeYeRbSdSSU8vpA+d3ZgTOMzXbS3p//HayGi8l"
    "xOtDgHN8vgu7I47CXNYhItB3+/k51IwVPFP1Xdf2RuZPhNsYd8/PctETsmY8FZzVih7ZUBT7LYOL6DTbDIM4emZSQl4lMDJtEJ6V+wCsy69f"
    "qHCY7PYZQyoNFM3RCghSm+xPU5RP7o4uHq3S0ajPWhw83q2waWANCTKrWuW+VD5ivZq7a/2WzzhsSiJ7oerdbzGO9mNVyl6I9BLoxtSjW3Jm"
    "ygUQO99/OAk1rQG7FbzwHO5PsEIT+Ojr1hVmPY9qjsC+6LV5Dm+3vGd4Yg0jygVoU62FjepPeIbfuJAYINo14bFF8QxNNUrEFTtRmWltrD8I"
    "WoeZ3bYjzyHdni5l4rFp4Pz2H7IIEoesZrLfoGDVydrrPaBjJx45RIpj/BPuK0GClteaSutn+4DsKcpWLSDA7UGWRFabB+KIKoZa/o8PHFW5"
    "pW5KIFShOzXbnDNQcVJrNDZ54pzBkRMCEuGvudJkOuK/A1qqQlE/lYNww2PkW5OBt+WdNMhXqk7R03k54ruEm/W9zEElIPaKBiJ3YQ0TiBrn"
    "W8G3CMxusdl9fR45Ku3KIZwTwnlQ0yx+YBNKT1F2rpLHd3sE5k8wf7o6ciZykEmINC4jh+Ygy5Juau1TsVJg0hsCx75IVa0l2oJKanWdIY8v"
    "1eoqX4OsVxB1sdlfIA7eqQH0cGca3a5oHcrmarLFDeqM1ne5MbGYJYQm6Ry7zOzUO/SRv/aWGH2qqcXtqfdHvA2C6NFpPqze+yYTU1G913jn"
    "nigXprL6tMBOKfNMz12Y3iaFG/BGslJPh5UOfYUqrp0xWRqxm6CazOfW7M75qbzyQIkpKoqGSflJO+TKE0J+52AehMDYEbrSmIYcQZ8ZcCZs"
    "pYwk7hwwQzNiLgjQZsheaJYTH05ZV1PRYW8+n4yv87yubjGvRk3QidSzOpYr/reUDOFSZOqIfBVOpwIqbju/ZvyHxaKx2PIBGyOBBu7xSz8u"
    "CD3ViTbnYmYs9CMyQ46BdKyb9/cYZ3N4hJ0ucfDir0Q80BQUU52+cFYb1LUPzkz3qL/Cc3fDr3qxjyHCvhm1a1ukpzlakXmuY5edAz0zP6vw"
    "qm35U4PeIKQq2j9PUXwJX8SYkbI9c4VmO1z1SeKJHiLzJbOQVvSN1+mhO6aTwjmB2dzTFxW0QlHlWqrwA4vcYWyYK1pHS+4YAQ=="
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1: Final = (
    "CAQShwcKEQgSGAI4AUIJbmFycmF0aW9uEgzgvB/8tNUGq1nfsxwaDG+cT0qM1p11gJfjDSIwtRicc6aAt2GBKWg1GHCUadncKjMPzXWPvDlt"
    "+F3ibcjosDbl5nyvWowUp+Vq+AyHKqMG3SAB7dNiY7nLrdCzSD2Xv2LLv0b8hPdf/vnIY73zBhH7rHOjRCRPMqNVGuXfpZVvTCvgyuXbLaAq"
    "xHg3O1GexFc8OPnXxBt0jjYXu4N/5yKAQcl+7xwwS18UnSh076vLr47RER5xRmQ5IiPNi43UO8jurj50AEucd6V6C+l85jZNUIsQPEgqKIt/"
    "vi9VSDjGkkTbLCdC/DW95wE8SWXkJO1Bs7VjHm8LHNTbYyLzc64FJkBQ+DqFjX0lZCkKLaeJML7nw3W/vo0p/E3XNDVjGgyErJYo243rIsNB"
    "rfH0PQacca0w+ipJWRPJng64+Fnyi8KlYV5qz35p4MmvSSIbkoHUZ1IDrP8CSoc3TIvcwCaTPY9b0TkmUuCTBNmvLS1I/x+Y7f9PH+rMO1Ad"
    "8PFXePqjR1Hq1smJgZMpjVMGBqKZkBoKNK3g2ScNrjqzkOMo9jYZhpfnDnl4eF02ouuOktuNTuJlPc7L1xFADckbPAkqTyVrUmGJTGL8bOz3"
    "cPltf1MzeHMoe9IOG5gxCYHlo3wb53UMOxZJ9dAK4QwdG7LyksN3sSfQ1nqpIVgWZ4iN6wLZt4H21P/MGUddVH6nCRmJSjELG9VqzEXyYquj"
    "MgUFtFhS8qYZDE/077wGT0Jw5IYyXTTEHjqV9I+UIb0zaOTWWgBefmnnahJYpq5Wjmrki5nRvAKdPY28gTidvvdbrt5eoJ5h+7kXAYXM8Hh0"
    "3LrfnIBfjuBy4/SgHAtLbrWK0jEVXF8QeGL/OP7q8Xdr6Mbb9bxL2JndZi97zn2x/ae1vicszJXsl/QuGNGrShjt6dyPT4nwoClVJX9VIGjv"
    "t88CzZ+/s7Ar3jbgOEqVEfPDjtdnlTswfRAhxzTlvWd6axok7j5G0c4bl2TA1EPQiIL6GDpP51o0biQ3NMi5InEZxUnU/Y7e6tqn5GKQLqEn"
    "2h9IgVn+rDbcq4AN4FgwrF88sQdxBTCK/9AP76jvcX/WogcIclnu+RzmzHijGcMXxrlqDvIdauQsdUKakPnW2VCmfxBu/bChJpS3CM493jrF"
    "PHy3VOS8NxDVLe4HAPDiOM0YAQ=="
)

CLAUDE_SONNET_5_5_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "id": "msg_011Cg32qT4xpsR9qgkrqjiGw",
    "content": [
        {
            "type": "thinking",
            "thinking": "The user asked me to read the validation file first, but since reading the "
            "test file is independent, I can fetch both in parallel without violating "
            "the intent of their request.\n"
            "\n",
            "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
        },
        {
            "type": "thinking",
            "thinking": "I'll check `validate_quantity` in `shop/validation.py` to see how it "
            "handles zero, negative, and non-integer values, and also review the "
            "zero-quantity test in `tests/test_validation.py`.\n"
            "\n",
            "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1,
        },
        {
            "type": "tool_use",
            "id": "toolu_01WcurCDsB8E8W42GKceb8B4",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            "caller": {"type": "direct"},
        },
        {
            "type": "tool_use",
            "id": "toolu_01NdHxDmUwwCWpWyaLT2ThJH",
            "name": "read_file",
            "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 40},
            "caller": {"type": "direct"},
        },
    ],
    "usage": {
        "input_tokens": 4,
        "cache_creation_input_tokens": 873,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 416,
        "output_tokens_details": {"thinking_tokens": 220},
        "service_tier": "standard",
        "inference_geo": "global",
    },
}

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "anthropic/claude-sonnet-5-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        **ANTHROPIC_MESSAGES_REQUEST,
        "model": "claude-sonnet-5-5",
        "messages": MESSAGES_REQUEST["messages"],
        "system": MESSAGES_REQUEST["system"],
        "tools": MESSAGES_REQUEST["tools"],
        "stream": False,
    },
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        **CLAUDE_SONNET_5_5_RESPONSE_FIELDS,
        "model": "claude-sonnet-5-5",
    },
    expected_litellm_response={
        **MESSAGES_RESPONSE,
        **CLAUDE_SONNET_5_5_RESPONSE_FIELDS,
        "model": "anthropic/claude-sonnet-5-5",
    },
)
