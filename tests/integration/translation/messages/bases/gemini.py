from collections.abc import Mapping
from typing import Final

from integration.translation.case import TranslationTestCase
from integration.translation.messages.templates import MESSAGES_REQUEST, MESSAGES_RESPONSE
from integration.translation.templates.gemini import GEMINI_GENERATE_CONTENT_REQUEST, GEMINI_GENERATE_CONTENT_RESPONSE
from pydantic import JsonValue

GEMINI_MESSAGES_RESPONSE: Final[Mapping[str, JsonValue]] = {
    key: value for key, value in MESSAGES_RESPONSE.items() if key not in ("container", "diagnostics")
}

GEMINI_3_8_FLASH_THOUGHT: Final = (
    "**My Analysis of the Task and Next Steps**\n\n"
    "Okay, let's break this down. The user's directive is clear: they want me to understand how the s"
    "ystem handles quantities, specifically zero, negative, and non-integer values. The constraints a"
    "re equally clear: *I can't modify any files*, I have to explain *which* file I'm reading *before"
    "* I read it, and I need to use the messaging client.\n\n"
    "So, the first thing I need to do is identify the relevant file that likely contains the quantity"
    " validation logic. Based on the user's explicit mention, I'm going to start with `shop/validatio"
    "n.py`. This is where I'd expect to find the `validate_quantity` function or related code.\n\n"
    "Therefore, my next action is to request the contents of this file using the `read_file` tool. I'"
    "ll need to specify the correct path, likely `shop/validation.py`, and I'll include a line range "
    "to make sure I get the relevant code (I'll err on the side of caution and specify a larger range"
    " until I can see the structure of the file). I'll also ensure that I use the message client to f"
    "ulfill the user's requirement.\n\n"
    'My response will follow this structure: "I need to read `shop/validation.py` next to inspect the'
    " implementation of `validate_quantity` and see how zero, negative, and non-integer quantities ar"
    "e validated.\" Then, I'll initiate the `read_file` call with the path and line range. My focus wi"
    "ll be on identifying the precise validation rules implemented in this file.\n\n\n"
)

GEMINI_3_8_FLASH_THOUGHT_SIGNATURE: Final = (
    "EoUMCoIMAWkUfRNAt6WiTWGx73C1cgarbMkd0STsaBTv0DSrICGKWeyBN9ogaaR8u/oA25JfVEENowEsZIoFwKX85uOsevrt"
    "So/sB0HCD+9aN1brlZmpOT413RxVKqvnsk5hBEN6JmhOulxKgpcwftZKsNTnn9NWlvM8DOAXsjiK+4pHx5jNtGQe+T8vd2Wg"
    "uqLWu4Gky8NMq+kde0q4jSlC+DYEZ8v5DXO8Tgb+gob/jhGOhvDip7WFY9CPF0X2MI4pbOAAeX2GT+XhteIuYMfEJuA7HAs3"
    "IEHTkgqjS/pp6CneKt/BE4jisIalNN1A0xrd+fWd+oQsG6PkOfXpUmQdF//5qVOHKVI8PPIPF3oGH7CD4i7FqBeDnYGCPxrt"
    "lTY4DT9nMkOXE6D6wh7hUoJ+Pihckl2JcnJKFIlcnFA5WlG3ph2vgF9iULhhQpRSP3WMu3CTmppGfrZiLRzpi4AztkrwlPRk"
    "PZvXd8rFhddD6+o0Oc8Wj3jPHg9FhuAHOQyVGrM6WNLWuTCHQQjvGRfamniStmfOFSOvaBVStHQJ3FTL+y1MVqn+6PDSxjCP"
    "1uVmCO0A0bjWxeSEN1pm8Fn8iNIE+oxGbPOPmTOX/I/viZEs7w6vBiIUNna9mMS2pw1asUn3gDCTEl47YtuUa4rMavR7CdsX"
    "uaULaPZcYU9LhGr5he1tWdKe1RCvLyUM9Cah8za3ByyQUbqpwE2qmJeQpJc1bmhbMvD3za58oCRWDS6P6EWfJBdyPNhtPhbc"
    "gP8w/8uEq1aeoAVcLwJr48bVwWnsUEZ0INRPtgpy55lRayzGo131gtFnt/5lFxz6Q3lopEZlaJgd3X416LngnkS1Yw6uU/J1"
    "GPaFlVsYkw/Z/8BkD+Q9pREcYg5Kgq54BOn8fws3TsBtW5g80X8AksR1sCxmwyZkfZx8quis+0gNVmTo3dFtcnW4Q2uk0PUn"
    "h8Kv9tTkf0njAZx1ExkgxGLPnS0/7IkpkihF6AuznOl9Hz3+jNjHvscHdx0pFudTFgEa4TcYnQhuj70RJ0NdHC4XgWTUHjeo"
    "tlwkunLpmUOavMYLBBrjPcmN8+9czz6d46GGDg2wTqV7jYQT1uyuTOaCF0/y7wWeGeU7gn8KqkgpwcQCXQU2AvqkMCdY4dq/"
    "uxh/tlhkzVeMwtTCeMjnMtCMgwarj5VAwivS6e9fc3hbls0qZpJahNkSk4K3+5qABqy0rvcdNEXgMIq02mM5EJab+LJwgY9U"
    "BVN67FV5lV9of/p104w3yVPFt7lZlCh3d5xaosZyxjmSt1WlR1b+t/2Wj/08vm2qrpw4lJYa64PwkB6TeQTNiAYaqu6iuVEs"
    "/wnZBcAI6Ik3pg4nZ776igMXLgw6zOBKU0uS2gqQcidMNK2mdZGOYWuSjHsMRHztlC+z//Rc9KMuAJ9uEAT+mAH+jEHv+q7G"
    "ErL2nMrc4LmViqZ+QKvfpCupH0LvsG/Ab2hThlwAyt/W+vLYtjgNcYnFV5IWVwfuXkYaMkAEZZJf3TO8ja0UZh2VZnuXhbp/"
    "rrjaxKQv6CWsXXvbBviBaH3OEKir8m9+SsQVioqheegPSY1bLeDprxrryDhiDUw3bcz/jtCWsGFoTXeN3g0Thdn5DIFfpIq+"
    "cQqFOwyyzRpEEYzL78K3VsThS+i7L3dj1WubhgSZ0R6MBtE2BfMWLQYKwU3TOWgiv+GpQGorqXdPN95BVI27+ykf7dVO7NEX"
    "S23M4ucImSq6Zd3uDRZaRpHoMSGOTwNcNVNpyrpCIa7kg3MqnBdqjdPDyUDa/cj522XXwzvrj112CGTNOcw3wCbEivSOtT9b"
    "Yagt3Zg1uKgKSyQZPQx5qmxVBXBqgo/MViybYA3HYohmtrbwPWXUWzoKNDmjd5rvkzaIfjPGMhYLgKXS5f2gekpAqb6rndYH"
    "MgONwdfCmWLr/m+oHuOvlO1tHHcXtrdvbr7TU94/OPYoSFLgDGv7Q2mk1GFdz2xFQYzMH1JdEYtXGwmYXUKIUbVJ6ZQXB750"
    "td/Lfz1tZ7G2nPJMMIKl5b123IDjKrY0fZD19tevY54="
)

GEMINI_3_8_FLASH_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": GEMINI_3_8_FLASH_THOUGHT,
                        "thought": True,
                    },
                    {
                        "text": (
                            "I need to inspect `shop/validation.py` next to see how `validate_quantity` handles zero, negativ"
                            "e, and non-integer values."
                        ),
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "path": "shop/validation.py",
                                "end_line": 50,
                                "start_line": 1,
                            },
                            "id": "call_43752",
                        },
                        "thoughtSignature": GEMINI_3_8_FLASH_THOUGHT_SIGNATURE,
                    },
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
            "finishMessage": "Model generated function call(s).",
        },
    ],
    "usageMetadata": {
        "promptTokenCount": 354,
        "candidatesTokenCount": 64,
        "totalTokenCount": 773,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 354,
            },
        ],
        "thoughtsTokenCount": 355,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.8-flash",
    "responseId": "5ibQasrzCsbUz7IP89eBkAM",
}

GEMINI_3_8_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "gemini/gemini-3.8-flash"},
    expected_provider_endpoint="/models/gemini-3.8-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        **GEMINI_GENERATE_CONTENT_REQUEST,
        "system_instruction": {"parts": [{"text": MESSAGES_REQUEST["system"]}]},
    },
    mock_provider_response={**GEMINI_GENERATE_CONTENT_RESPONSE, **GEMINI_3_8_FLASH_RESPONSE_FIELDS},
    expected_litellm_response={
        **GEMINI_MESSAGES_RESPONSE,
        "id": "5ibQasrzCsbUz7IP89eBkAM",
        "model": "gemini/gemini-3.8-flash",
        "content": [
            {
                "type": "thinking",
                "thinking": GEMINI_3_8_FLASH_THOUGHT,
                "signature": "",
            },
            {
                "type": "text",
                "text": (
                    "I need to inspect `shop/validation.py` next to see how `validate_quantity` handles zero, negativ"
                    "e, and non-integer values."
                ),
            },
            {
                "type": "tool_use",
                "id": "call_43752",
                "name": "read_file",
                "input": {
                    "path": "shop/validation.py",
                    "end_line": 50,
                    "start_line": 1,
                },
                "provider_specific_fields": {
                    "signature": GEMINI_3_8_FLASH_THOUGHT_SIGNATURE,
                },
            },
        ],
        "usage": {
            "input_tokens": 354,
            "output_tokens": 419,
        },
    },
)

GEMINI_3_1_PRO_PREVIEW_THOUGHT: Final = (
    "**Analyzing Quantity Validation in shop/validation.py**\n\n"
    "Okay, so the goal is to get a handle on how quantity validation works, specifically how zero and"
    " negative quantities are dealt with in our system. I need to see the implementation. I'm going t"
    "o use `default_api:read_file` - that's the obvious choice here.\n\n"
    "Now, let's get the parameters right. The file I want is, of course, \"shop/validation.py\". I don'"
    "t need the *entire* file, just the relevant section. Let's start at line 1 and go to line 30 for"
    " good measure. That should give me enough context, perhaps including the function definition and"
    " some surrounding code. I'm expecting to see the `validate_quantity` function in there, probably"
    " around line 18 or so, but let's be sure to grab a bit above to get any relevant imports or help"
    "er functions.\n\nSo, I'll be reading `shop/validation.py` from line 1 through 30. That should give"
    " me the information I need to understand how we're handling quantity validation, especially thos"
    'e pesky zeros and negatives. Now, let\'s run the tool call: `path: "shop/validation.py"`, `start_'
    "line: 1`, `end_line: 30`. I'm expecting the output to contain the validation logic.\n\n\n"
)

GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE: Final = (
    "EqsHCqgHAWkUfROgGyzkZ+pi4VHIZ0/S9yiD/m9VZs8L+Rm8Hnfv6gV25Dw2Yc5P0GA0XK9w2l9T1N12/8TW6Jdh15A58C3R"
    "hiVDYeyIkhLlk00QjCI3VtdLasRR6T9wieHiXKKIrm/WbfAu5GougrYnUnt9OKFjiH4l8SxQQfXISCA7Fj/+/N0LCzNRug8P"
    "DI5X3x3Wug86RcIa9WwNhK1FQ00V2kMzGCdt64Ieg9YqA9sTtgWqRzEc7QbD73tS15WzeSjogaFI1aXbtpTPJCYPrUN3aUqs"
    "yXlqhn5sbKJCuDg7M7zOmneh8TbJwTrcpxvHfn8xEHZouCQhY1/WyFGWXvBjzZnJpwoheNkyA7GTOSE1vhB0I5oyS3OwSHn4"
    "zLYrcNvwu+2NZp7zc5fImPLxkIOpK2sl8N3q6M01Qnvae4XXtfZ3wNNviM1N72s8I6ZWAYkeS+U+14TZFrL3jsx/Mewi/6be"
    "1Y874WiAuq8B3lF3ACnnJ9E8PDTYripxNwhjtveOJC3YCQncxd6CMIf13Xgnj17G9MEdD1HqKnBkLDsfENAfbauaxF+pouX9"
    "+JPPHZHyQ91AzcsnXchnFHkZ61goISJILsRikIfzkiVLSopWlsOESzw2ckIA7uxl3WEpON5Qf5/YXEgMIb3OuakYEgLnHdKd"
    "WXmmzkel8yc0xxYLzk9uAy2X+ZcKJwbmmU0oJkSOe48zc26xcBZEouznB+LO9q7S9qzh3nN9owa/OAVp/z/5OA+jGCxr6VbI"
    "/7bMHcbI1S5oZXN0jjKAlMqNzlo6UWWBB/TPqH2c2PtEYYcNZWhpoQruvBGyGFtT+fj/YrumG1c8EAsBC4NmBecdTMFqfZsH"
    "imnQ/qPenuO+qof7iyxZKRhSRTC5qL5PlwuE4tKL4QzwjDDUrOJQ4EapDjYGC7fR143nP52KZGTONQlYtl8HMOh5FT07PExO"
    "tHvXMfCh10xkfkWGVAbVWO2mcp5naAAK9eoSeLydygv2P3fZCProx/xPU3RvzDafLGbc2xEO2N67RQnvCIUQ9SwD9iFy4AZR"
    "HPaxjxamQLgHW0hDQuRjBzqtXkO5z7y2AKQi1twxv2UAbbZ6ZR605/1ZNbJXHlsf6TCGKZ1/4TXjeQHjLTGf/qb0Jj9ZFxgo"
    "0ffs0HxBygsZY7AXvIMVHwgHeyt9nGiz8yPGReQ/1WJEdcmy0GNnYebPuYgWOk9IHAjrpO/zna9cznwDdmzOjmbXlggwqyFI"
    "kudjw2Yz"
)

GEMINI_3_1_PRO_PREVIEW_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": GEMINI_3_1_PRO_PREVIEW_THOUGHT,
                        "thought": True,
                    },
                    {
                        "text": (
                            "I will read the `shop/validation.py` file to examine the `validate_quantity` function and unders"
                            "tand how it handles zero, negative, and non-integer quantities."
                        ),
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "start_line": 1,
                                "path": "shop/validation.py",
                                "end_line": 30,
                            },
                            "id": "call_29922",
                        },
                        "thoughtSignature": GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE,
                    },
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
            "finishMessage": "Model generated function call(s).",
        },
    ],
    "usageMetadata": {
        "promptTokenCount": 354,
        "candidatesTokenCount": 69,
        "totalTokenCount": 656,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 354,
            },
        ],
        "thoughtsTokenCount": 233,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.1-pro-preview",
    "responseId": "7CbQarXiCMOb6dkPso3s2Ao",
}

GEMINI_3_1_PRO_PREVIEW_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "gemini/gemini-3.1-pro-preview"},
    expected_provider_endpoint="/models/gemini-3.1-pro-preview:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        **GEMINI_GENERATE_CONTENT_REQUEST,
        "system_instruction": {"parts": [{"text": MESSAGES_REQUEST["system"]}]},
    },
    mock_provider_response={**GEMINI_GENERATE_CONTENT_RESPONSE, **GEMINI_3_1_PRO_PREVIEW_RESPONSE_FIELDS},
    expected_litellm_response={
        **GEMINI_MESSAGES_RESPONSE,
        "id": "7CbQarXiCMOb6dkPso3s2Ao",
        "model": "gemini/gemini-3.1-pro-preview",
        "content": [
            {
                "type": "thinking",
                "thinking": GEMINI_3_1_PRO_PREVIEW_THOUGHT,
                "signature": "",
            },
            {
                "type": "text",
                "text": (
                    "I will read the `shop/validation.py` file to examine the `validate_quantity` function and unders"
                    "tand how it handles zero, negative, and non-integer quantities."
                ),
            },
            {
                "type": "tool_use",
                "id": "call_29922",
                "name": "read_file",
                "input": {
                    "start_line": 1,
                    "path": "shop/validation.py",
                    "end_line": 30,
                },
                "provider_specific_fields": {
                    "signature": GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE,
                },
            },
        ],
        "usage": {
            "input_tokens": 354,
            "output_tokens": 302,
        },
    },
)

GEMINI_4_ARGON_THOUGHT: Final = (
    "**My Approach to Addressing Potential Quantity Issues**\n\n"
    "Okay, before I make any suggestions, I need to fully understand how this system handles potentia"
    "lly problematic quantities like zero, negatives, and non-integers. The instructions are clear: I"
    " should examine the relevant files first. Therefore, my immediate next step is to examine the va"
    "lidation logic. I'm going to start by reading the `shop/validation.py` file. This is where I ant"
    "icipate finding the rules around quantity checks. I need to understand this to determine if the "
    "reported issue is, in fact, an issue, and also to understand the context. So, I'll start by read"
    "ing that file. I will use the `read_file` tool and specify `shop/validation.py` so that I can ge"
    "t the contents.\n\n\n"
)

GEMINI_4_ARGON_THOUGHT_SIGNATURE: Final = (
    "EukECuYEAWkUfRP1AGBm+VfUg2RBxd+BkfJYxv3HJpn4BHb8bKt8sO2dnYDaU7aro4ErMNkRlXK4cJ/2fewoubknteog7q1r"
    "huhmhyZ3Ez8UX5nfeiRktTczwkCT2RL0qY0Yd/qaIWoZcI+QM2XKqHbbHEwW5+MJ9kaQBBzoIz+nyFVfvBmsus9QatIjp2Kk"
    "BDXa5HiuVyXobtRz4pgHIJQ0IUb7EBVshaFsa1hwLhKvWBTa9kNOrXWdbcLExvRTzPSyIOC3cxMxwV6UDLmEwrSgqPD+hZ+C"
    "YWT9SuQqc1EtbvzNWN2/y4y7ZwekleAEF1Wz+njeqWF01FX4+Der/Cc2lWkoCY9u7jJd4InpLnhKdBUGiSEtTubGMiTVWyAd"
    "RKBct6bfstopgnGgqGPM8cW/PCof7+c0gW2nWHU7IDpxB9E5bVb4GIE79+TpWAfVt+qR3nAnk0xN2k6USC8lnf6d9bvhJ+J+"
    "Z6loijZBaTOCL8FBMmZFwpwlLyj/9HojuS8DIL8OP1umNF4xqIz8FoGMjUyaeoFX6kHSim3r7kjkfP7h+yZa3seFTqjRLjGu"
    "PWrB8abJkP39TNqNaWtGspmAg4Z2PDBSrkW8G29BXkfdbRzV1/1MCqcVed0RE+D3d42/jhzG2j5ul1c0kuZ0RkACrX4WQUWU"
    "RhmE4s2z8jaSo6sTmi7nMjKSPV5p2tn9n/yaJoZyDXyXlVMhVTIE5+22YvPEnxWHPQ/aTthfxoi31Mq2fwCrTbkf1OoW/VK/"
    "QL903naPf4jpJexUiFDLgylJu+PHEkowuWE0QDC27kgSydvF0wHIcCHHsdU="
)

GEMINI_4_ARGON_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": GEMINI_4_ARGON_THOUGHT,
                        "thought": True,
                    },
                    {
                        "text": (
                            "I will read `shop/validation.py` to examine how `validate_quantity` handles zero, negative, and "
                            "non-integer quantities."
                        ),
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "end_line": 50,
                                "path": "shop/validation.py",
                                "start_line": 1,
                            },
                            "id": "call_1649114",
                        },
                        "thoughtSignature": GEMINI_4_ARGON_THOUGHT_SIGNATURE,
                    },
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
            "finishMessage": "Model generated function call(s).",
        },
    ],
    "usageMetadata": {
        "promptTokenCount": 412,
        "candidatesTokenCount": 68,
        "totalTokenCount": 604,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 412,
            },
        ],
        "candidatesTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 68,
            },
        ],
        "thoughtsTokenCount": 124,
    },
    "modelVersion": "gemini-4-argon",
    "responseId": "8ibQavD9KaOO6dkPs7f0GQ",
}

GEMINI_4_ARGON_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "gemini/gemini-4-argon"},
    expected_provider_endpoint="/models/gemini-4-argon:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        **GEMINI_GENERATE_CONTENT_REQUEST,
        "system_instruction": {"parts": [{"text": MESSAGES_REQUEST["system"]}]},
    },
    mock_provider_response={**GEMINI_GENERATE_CONTENT_RESPONSE, **GEMINI_4_ARGON_RESPONSE_FIELDS},
    expected_litellm_response={
        **GEMINI_MESSAGES_RESPONSE,
        "id": "8ibQavD9KaOO6dkPs7f0GQ",
        "model": "gemini/gemini-4-argon",
        "content": [
            {
                "type": "thinking",
                "thinking": GEMINI_4_ARGON_THOUGHT,
                "signature": "",
            },
            {
                "type": "text",
                "text": (
                    "I will read `shop/validation.py` to examine how `validate_quantity` handles zero, negative, and "
                    "non-integer quantities."
                ),
            },
            {
                "type": "tool_use",
                "id": "call_1649114",
                "name": "read_file",
                "input": {
                    "end_line": 50,
                    "path": "shop/validation.py",
                    "start_line": 1,
                },
                "provider_specific_fields": {
                    "signature": GEMINI_4_ARGON_THOUGHT_SIGNATURE,
                },
            },
        ],
        "usage": {
            "input_tokens": 412,
            "output_tokens": 192,
        },
    },
)

GEMINI_3_5_FLASH_THOUGHT_SIGNATURE: Final = "ErkECrYEAWkUfRORlQ5j/gkLmyx08FVeIMgg57FJXrjcLwBlt9iuELI6Uc5h+NR/Vm+gEsbdZ5mgFTmKjPF9K/eft8IlXMKm6odJjQOgOYsbt2JHzANo0bpfTmlA6fIi0G2zLbvBVASA6Bdxu1aPQuO4voioQRwm2vomRxH1YbWp8sKXk0DBVbefosldrL0zLJFsi5dFYCtPvw0n9olPVgptHzEdiyXqG+63aPxTooARRQutUH0XWAKR0V+P7qWPt55QKlLaKQqKeBndg9JrEplJihg1sp++y+NAi18fsqXteUS2zIeDtdePGM/GS5oVibwE25zJoziPRdtJhGasFSaA7a3znhW9PF0pBIPAIRPKE4NsQ0FRhpy7ksIXY+0uJ4N+WPPejrtKK6z5x+P0tFkFP0ZNNPM8FZbir1ncVhVxkZS/wWmhc/8TZoRA9ghlTpYhHJ+C4fRVqQqnyRR3SDpVTzB4/sCjBlb434dTH0U3jB4h6V9b/Zx4k4pwUwZTNr2FfgOt2bR7u05DOa+H73OzsNG6zBnMYgBndQdRgk58+l4+UcZdpGKB0lkbHdfD2bminBypEmeJKNRpuc7Smuu0YxcZiY03tzzhHdrUmItqC39OEr2CzRcT9DjFpiWydo3ej9ZkEXeyxoMCckpMmGWh6xbjAnX9gkrPFmUE2rJqblDJWa51i6u/p9Y6ciCq6j4lAy9eBfULIRGQt9pKOXagltOnX0vR0MTDYgWe4dVDSzCI1TV7X/SQYjo="

GEMINI_3_5_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "gemini/gemini-3.5-flash",
        "max_tokens": 1024,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/models/gemini-3.5-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        "contents": [{"parts": [{"text": "Say hello."}], "role": "user"}],
        "generationConfig": {"max_output_tokens": 1024},
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello.", "thoughtSignature": GEMINI_3_5_FLASH_THOUGHT_SIGNATURE}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 2,
            "totalTokenCount": 137,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
            "thoughtsTokenCount": 125,
            "serviceTier": "standard",
        },
        "modelVersion": "gemini-3.5-flash",
        "responseId": "4hjEav32Mt6P6dkPn5iDqAg",
    },
    expected_litellm_response={
        "id": "4hjEav32Mt6P6dkPn5iDqAg",
        "type": "message",
        "role": "assistant",
        "model": "gemini/gemini-3.5-flash",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 127},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)
