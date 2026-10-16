from collections.abc import Mapping
from typing import Final

from integration.translation.case import TranslationTestCase
from integration.translation.messages.templates import MESSAGES_REQUEST, MESSAGES_RESPONSE
from integration.translation.templates.anthropic import ANTHROPIC_MESSAGES_REQUEST, ANTHROPIC_MESSAGES_RESPONSE
from pydantic import JsonValue

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


CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSuQcKEAgSGAI4AUIIdGhpbmtpbmcSDKFGmJVRB+s8d6jjphoMrzic91ngVwSk+KnFIjALyD2lz/FJl0zwfGQJCf/81TXhJ1N49s46WltO"
    "EwVctdFi/mNtBuz/H8qLT6nZnJIq1gbvKJzD4syhoHZof3EHkaAfHxNRootT8J6SwvRw82stIfRyxoH3+xjCQq6iZzkkQ9HaQvdEeCexjN8N"
    "RFpwhhS/O183u+EESQtikc5PGw7Yzdwrl7YZ1wkQP6XVjyRrhHkqZHryYo1nf6G6g6jSE9eFyIc9abw6cBNwizPDuuXpcockqrQtpKabaO5B"
    "o2GBtKltSR8M/aOCDwCKhCJfentm1d9bcRVgT5Nfksz33HjoD5ADev3d+NnM6co0fWZesi4brJPdN21JZW5LjaHkgIFc2NbnSsR9eQkzIx9g"
    "G17kFtiCuaEzv25/G2gkTEbrj/aORXWQgQIcQQ0M+fBKDSFDfUy9tK3H45tEPoW6ej1c/oeYh+fCjJRgGone08Hp/B2F+f5YdAC94Y04+7bD"
    "UyAm3RlHTPiHRq5HnpHChIrhGldYQ3QU10Etr0dWj2N9UL5BbnMGVDxrGV8K8zjxAlGTZbGQNnhNWHWt5pcMR8afGOv/1R25jK5sO74gbKAZ"
    "HfNuWCBfGytBeyjM+yvYpLEaNROmrEX1uQvFPHmWxS/3x44BvJKHWvSrJ0LsDbP+eoOjFIzKma6HHqicJG8TraHIEpIaNF1XfAKyUWtXytlU"
    "Uiv1hdVjWqWuEUalOXeF2eOWQENZGfX6A3X2v61JKtsAh8BEB0IIL6ztu9YnW1aVClJN3mBPYedtNLSMQfjc5CXhLQYsFo8q/OwCqCvmhc0t"
    "SvZKH/j5beXTSGhEafdxWFLGilrEPE3qhNyHWGjW42/7n342t3kwE4TDt9MhnGuBQGOFtoO/YD/rbGJBq1nDYOBPhg/g6wJx4eMbY1+YpzmM"
    "1TcJ07X/Zwd2NgbZ8YugR8Fkxbz+wNwLgm3Wx4TB5Parg78jdek0p5DlfZ1aP8jluIxnlvzBc2al3yrNlWX79OYQSJvcJsn2P74HCWjwRmsX"
    "4lqhklY7/GiQP/dDFw8tPxFvfcU6AqAfv9OUICp0PvP3ATGhpTon06mQIo6rR6r7HUtGQdPUA1rAw2XgvUpoXl3f0l/SC2aIShpWjQUwu+Zn"
    "1VLaGBnNfj5moS5HtzbKVW9IqU/d9HDRsTlI07JB4LuDleS1JCS4mJv3vQVX1rq1mk5ETNXW/JjrhKHlB8Vxm6cOghgB"
)

CLAUDE_HAIKU_5_5_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "id": "msg_011Cg36zuYtPFhEEpad7wiAq",
    "content": [
        {
            "type": "thinking",
            "thinking": "I need to read shop/validation.py to look at the edge case handling around zero, "
            "negative, and non-integer values. I'll read a generous range starting around line "
            "1-80 since I don't know the exact file length.\n"
            "\n",
            "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
        },
        {
            "type": "text",
            "text": "I'll read `shop/validation.py` first, starting at the `validate_quantity` definition "
            "on line 18. I'm also checking the lines above it for helpers or constants that affect "
            "how zero, negative, or non-integer values are treated.",
        },
        {
            "type": "tool_use",
            "id": "toolu_01J9fZnNpvkQDk28Zj62Cc47",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            "caller": {"type": "direct"},
        },
    ],
    "usage": {
        "input_tokens": 4,
        "cache_creation_input_tokens": 873,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 289,
        "output_tokens_details": {"thinking_tokens": 117},
        "service_tier": "standard",
        "inference_geo": "global",
    },
}

CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "anthropic/claude-haiku-5-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        **ANTHROPIC_MESSAGES_REQUEST,
        "model": "claude-haiku-5-5",
        "messages": MESSAGES_REQUEST["messages"],
        "system": MESSAGES_REQUEST["system"],
        "tools": MESSAGES_REQUEST["tools"],
        "stream": False,
    },
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        **CLAUDE_HAIKU_5_5_RESPONSE_FIELDS,
        "model": "claude-haiku-5-5",
    },
    expected_litellm_response={
        **MESSAGES_RESPONSE,
        **CLAUDE_HAIKU_5_5_RESPONSE_FIELDS,
        "model": "anthropic/claude-haiku-5-5",
    },
)

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSwQYKEAgSGAI4AUIIdGhpbmtpbmcSDE4DraHPnzJaL5XCExoMhZYoXkhSWfwTcx8lIjBt8d8pYV0VNCDIdfuGLWpioTySeXeUjAmK11R+"
    "18SifTQpg+l0fRdIaI1HPrvqpikq3gX//sW77xYgKmFhC/pig1WvEWEErbbxw6XtLwFIuq+x62J2Vka5GeZeT5CBp4TcrFY718D90jSPqYm+"
    "b1a/Zg+2njnVpSLFTFss7vNmNa1YiUND2W4cIODegkc1tRQ/SRZRslMzZ9Xh39VAu2FsjJ0A9W+r6619hN9/ixT2yfxY1LT2Vf1YtTGhyDvb"
    "Cn8+GXmYfzCUiQKN1KzX8qjMoUIiaNVZgJy4kYTlqTHXHA+hkYe/I/dkoNc5Nb/arHCSBqmCW2/Jh4T7gTlokfS1YNIZbOMe4S+vByzm3r48"
    "sjf+Wf4utdKOkZCizLWjTdKA37qPQq3EN8dMeqywLODKR/4SoxojI8t0d+0Shfv7xYP9kS679M6bUpBiLbiRofIMBR+tiBfZJIMRt+/36c4M"
    "JKYYtYiU8l2xMH206+onxNvADUUBk/1WLE0AJ1up/5ZZJO3bXAubC/y3+2icxVCQ4wOjTd1N2+pCSQAD80yOdqNIYHokX9dvbFcgBP5SlpOV"
    "1Ast2tXCpcfSsdd4dP2e6Rr0APxrFEevkWiss5EVxk/STjehW1Rq4WFoxhhlQ8wibT2zAbUnaeR1aAQW4JPBrZsC8RNs+tu1HIs++mq4lQC9"
    "ls/pEn0L8QSQMbxtCGDHOmdjSDkzFuDZt01KqYKK+pnckSfhd2VxOLWXCtFZbxwviIisafYbL7ex8kOmazXXPrFBAsKLo84Y0ym2f2fmGnvS"
    "9p+ibGgizHLF3WPQJVDBdxIn9GEszB2vCWq4zdYZ3HUs+nHFCTFU3FT+MMGtxrnSFKhVcgwis627+XqmsXP0oJbsv3Txw9GaC9v6jD1OKil0"
    "WQpc1a/hTb3SwroE7x7gmEVe6QHnJjrrqG1tCWlpa9Dt6GZ82BCuU4326iPlZHdCtWOwvlLZqgreZM8RMPbdTa/rcfrGFcO/bT0q9yThzX7I"
    "ecnuu8D28QJlT3Vp779WakHBeNgvNq05GD6dvhgB"
)

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_1: Final = (
    "CAQS6QYKEQgSGAI4AUIJbmFycmF0aW9uEgw+f8JSDrP1OzQWjmgaDC+5fzOQbkLuJcHCICIwMCi871N6wTZlw9YxH6IC09M+IrnPgjtTz7lc"
    "SPv+bzBSksfh4tcVEvWKJdPcZObhKoUGONxWS5haAeeIk4yEZWsYuSP9a7PYWTyt0IhnYArjFQPGxT6p13RSod4N5u3QKo/iVz4tWT5m+vxl"
    "u+w3IUtcC7sOChRgMR+cN/kaj7O81NCdTXlxAp6w8VshNjTVSKKCxcDP8C6xBE5Fa14FPTwQeHwqyOmNeSzta1vmof1SB3u3NUYb5YAYQ5YB"
    "zszgyXpl8BakycxtgX8Q3V0jrS4D08YFXFCbPAEa6REPbcbDqGqnFnWPxOXoIpGAOXs21hk90asOqJH6AToFK4p0UWIMG9RFkvCbQ7otPI86"
    "tb8Xx5LtL6SG0BAuYqKhB0JewYrmuOLT4OyzlVYWhUVn5aEwpJmR3qc900LkVmUA2SkyUMwgVPIgzoEMrPYWtmwABoMHjRg088WZB5FRmo7h"
    "uf9VE4P5tUPUfHYipgMsK3h9WAuvV9a1+4YtRSB74woVdoDAIawRCwFg/T+Njl84mBr6mND5eTn6UmqNdqT0w3VucwLq6KChClA7ZnWItamC"
    "dZwm8LtA9ysd/xuE20fonf1Y5DhW4KI7A/P4W72TLUwyYBc+1BAMy5IFYKGZiE8yXRMLDAUsYuW5YAbQua3sDFR1DInlNexNuExlVN9lI7BC"
    "IS1tQKultivKpg2j9SJGDP1kFwwMMQzBee69ii3ACozf+kcnDY7Xch5yeLkKnZn1Qz0IMBPOkbe8kWBB1sVGYVXplDNuwNmE6SRwNIqeVdeo"
    "ps86SpZ2bPXSk3H6614J2vVpV8SHQgoVv+7BZcsVXIKIAOhO3q/JQA835W+M7sESaAs1XFH54MfD+6UUlIhep9iq+3ck25zuDkoBT5xk9N3v"
    "NeBm5JlUM33Bk8cIknUq35W/+jllBk5gEe+RQ1x85aNgTGzxIJDGhxP2tZG4pV8o0fkYR5n7PD1n0FbA3mJf8nnFcKDj0RevTz/y2woydaOd"
    "S+OGnbuOtIJsJ2dpUFgDom9QDXNs1Bfx8PV29ND2bPOIYG+TqMP09e0zFvLVDF3Cz2+3zlHbaXj0xy44tD41+EpWYJcYAQ=="
)

CLAUDE_OPUS_5_5_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "id": "msg_011Cg371xxoDKT2pPQfVcP6e",
    "content": [
        {
            "type": "thinking",
            "thinking": "Since the user explicitly asked for the validation file first, I'll read "
            "shop/validation.py around line 18 rather than parallelizing with the test file.\n"
            "\n",
            "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
        },
        {
            "type": "thinking",
            "thinking": "I'll open `shop/validation.py` around line 18 to see how `validate_quantity` handles "
            "zero, negative, and non-integer inputs before checking the tests.\n"
            "\n",
            "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_1,
        },
        {
            "type": "tool_use",
            "id": "toolu_011bPgnPBMCrDsv9vkTKQsFL",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 45},
            "caller": {"type": "direct"},
        },
    ],
    "usage": {
        "input_tokens": 4,
        "cache_creation_input_tokens": 873,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 248,
        "output_tokens_details": {"thinking_tokens": 150},
        "service_tier": "standard",
        "inference_geo": "global",
    },
}

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "anthropic/claude-opus-5-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        **ANTHROPIC_MESSAGES_REQUEST,
        "model": "claude-opus-5-5",
        "messages": MESSAGES_REQUEST["messages"],
        "system": MESSAGES_REQUEST["system"],
        "tools": MESSAGES_REQUEST["tools"],
        "stream": False,
    },
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        **CLAUDE_OPUS_5_5_RESPONSE_FIELDS,
        "model": "claude-opus-5-5",
    },
    expected_litellm_response={
        **MESSAGES_RESPONSE,
        **CLAUDE_OPUS_5_5_RESPONSE_FIELDS,
        "model": "anthropic/claude-opus-5-5",
    },
)

CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0: Final = (
    "CAQS1gYKEAgSGAI4AUIIdGhpbmtpbmcSDGejrLS2VrnijeVfRxoM9JoxxlmEbYEIEjtoIjDZE1H+Mf2sV36GpDiZ6qmC9yPoLvEpR9SvLktB"
    "uczCkKWCCiSMfKz3aswvNY+eX0Qq8wUeup4+/o2u5QfFgk6jxQgz4ZFZAlapXhbfwG2kcLegVjWxqnGBexyWZXd3Uu1l74YVsvig5pj8kIEJ"
    "xKPkRb1J4Ts+q5bOF6sJjVLZLgtclDs0Qdwj73dUcqlrTqVlwWZP5jFxwjdzD7h19kZP7NEWFN8+tYKNjnNLiNYBq3RdnioxWPfsAC97L1rY"
    "VcD1tHiRc3Wml/y5poBpvOGU/gDi96jlGELiZdqkYOPOYz01oI7snrYZZ+t1Um8FemhFBFSq1h2SSl7xihYB34cDPd7929obEKBoAexbPY2K"
    "T+NtodCZ4DINMsj1dkAkZCx/QU27lOtRiPBS7JfgOvOXHFlfXho91r9tS9p4N4a1WQeTX96w83iGLXQK2DkcUChe22cUMdyRVzJn2hnoVN3m"
    "Rl/bMQZxRiF0lcmq3fen6kWG3kbSpXIKCzbEZhfkRN1uVfZC1ed0JE8qgTYIUh4ZbXIDS1ZxEDbC086pZupahW5b00K7Q2bPuekv8QwIJsW9"
    "G+FIdUYmMCDJuRQloY4TFYeNmCwOUT5ZMNoB0N0snFobDj744j8ME3IInjX03+FqWtRWJ2tsb0E0GvQlk9AisxHviaHJ0ejIwsgcdCtOeB8n"
    "GWRlVh+GUTWoFtr75B6WN3jaIVpc57BmOWG0xiEqfJWDGSPxfMNx2FVQDF45PWBodL8hTYru0KSHaSD54h1txGFkJGeX8GDlLxe6B5xEYYpx"
    "OU3+/4+DNtg7bsGvBRjtZ2TdzrE1N89j3hGlw/ShNS7tNarAfxwCLBbgKh/1uF3IcYfDt891lvvALQG1CupaXyyJiWxCr1oizeFlVe/FLSmg"
    "H3yMwKVO8ol20DIlqrVjjqRWHH5wURpOVfNiQN8GMNAwRa20sUsbVl5aTTPkyVwYox+7s/QuW26zqdDztKasEToFU21SLyqSEnkxqNTTO2TN"
    "55HYaq5g/GWqAaVIbfdHgwgbf25NL0CEmrWADytfBY4NCLEPNHTVCES5CKErwiGKkBgB"
)

CLAUDE_FABLE_5_1_THINKING_SIGNATURE_1: Final = (
    "CAQSxQcKEQgSGAI4AUIJbmFycmF0aW9uEgxy5XyCv93dBoadwgoaDJlmUXHZn84IYPqUnSIwEAQvf5RKgehIVEOImDwlE5JQqhqPR9hP6TRD"
    "c8VHaIIO0ajqrIvzmhC+qU5q+80vKuEGspcWI4tDdYEV8Op3nTUrfE1iU46R7zlVrkZb9bjdv+VZvamiz6RMNv0UAiQNaBOgA3E8qEa21LeK"
    "nWB00LCD9lNSAtJQmLTEn9Eps9BLutgDKMBqMIKE9tpjBSDjJ738+6O3fGhc9+PghdrNUdiP2tvCIEHRDdVkSmSXO44BpqK4tbVaG08rZAwp"
    "dLETvIpH8dgneIormOZWkW0vvlQ4tC47G2KZ861kXD4KAHxhfPzMET6yuaD2phSl/+BWvhNBgvCRqaDYf6LqExDZSy5XrIzt+SGzR8ZWpsVq"
    "mY0QTAw5ECHlsj37Cilkw49OrY5zg9sQ1m7KuopASkT3gEes+TjPdofHms2Dgpfcx5svqF2kpCJBAj9ZaRqVVUMzWlIbrx2+QIrW9FR0wi2O"
    "XmxdSrw2fjlp3m1ixjy/0jjyaKCfK0N09VTih3Q/OigXuDghsF7oXjjqYqHXxzRMkULGGjmunM6CcdJGcjzWRbSef2grS/KmsZZ7Xh/l/lRc"
    "JXCZPaSHCgz8YwLSPOjRvEjMardae94UEXb/sKeAKczImcq5a0iVFvMjmUp7khBo+ygDh5uGmcos7+amUO3FiiSh0rxWbN20pHMGTsUURVXE"
    "p2dzBvZ33egIaGVVjT5iQphQbMj/aY4IP8Mq/BaFqz0tvbffE4fv4bXz8VZYY9yxu6KxYLZYghzAvWyBujcZyK8Ju79ShlKpdTD2NE8rHZh8"
    "X4IHKkL57qbcUf1onAxkEYt8fRR7o0aEuM+tZJX/vSorD1QAvuufraq6y8hP3pOtztVyRb1TFzqxwJwfaUavS2sKzPRf9DSguxK8aeaV0jXB"
    "H66hLCfXE5bPB4+ez1LWqCPOXNEGVBPFE8Z+8VRXCkDGOyG3p428KKc+5o2qc6Wj6muqJayRenHO54TtEVzGc5h6vIhjHWT8TBEd6QktdByp"
    "ino3ysKvuAEIT8jpPB+ucIMOaMZgMkJenSxKn7A+A/rNSQoLXX9uqpJDUE9+ZjucFABKApxej0o4PAsS5QSONURp+OUSjxJbEt4328Cl7LNW"
    "MY8/X/fsyyxI20j5FBn5RzU8bHk9OWauNQO+vAf9/yGnMfLyjGZNRQKqVoURbO5/YwZ13y9pscRTOGxClYbG9XlilFluGnMeSCOPpsyufhgB"
)

CLAUDE_FABLE_5_1_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "id": "msg_011Cg372jZvQrwtebLxTc8Ph",
    "content": [
        {
            "type": "thinking",
            "thinking": "The user specifically asked to read the validation file first, so even though both "
            "files are independent, I'll honor that order and read validation.py now.\n"
            "\n",
            "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_0,
        },
        {
            "type": "thinking",
            "thinking": "Reading shop/validation.py around line 18 to see the full validate_quantity function "
            "and check how it handles zero, negatives, and non-integers.\n"
            "\n",
            "signature": CLAUDE_FABLE_5_1_THINKING_SIGNATURE_1,
        },
        {
            "type": "tool_use",
            "id": "toolu_01RgnHAg9GpXpfL13hyRodfh",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
            "caller": {"type": "direct"},
        },
    ],
    "usage": {
        "input_tokens": 4,
        "cache_creation_input_tokens": 873,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 269,
        "output_tokens_details": {"thinking_tokens": 171},
        "service_tier": "standard",
        "inference_geo": "global",
    },
}

CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "anthropic/claude-fable-5-1"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        **ANTHROPIC_MESSAGES_REQUEST,
        "model": "claude-fable-5-1",
        "messages": MESSAGES_REQUEST["messages"],
        "system": MESSAGES_REQUEST["system"],
        "tools": MESSAGES_REQUEST["tools"],
        "stream": False,
    },
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        **CLAUDE_FABLE_5_1_RESPONSE_FIELDS,
        "model": "claude-fable-5-1",
    },
    expected_litellm_response={
        **MESSAGES_RESPONSE,
        **CLAUDE_FABLE_5_1_RESPONSE_FIELDS,
        "model": "anthropic/claude-fable-5-1",
    },
)
