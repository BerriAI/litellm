from dataclasses import replace
import json
from typing import Final
from unittest.mock import ANY

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.responses.bases.anthropic import (
    CLAUDE_FABLE_5_2_TEST_CASE,
    CLAUDE_SONNET_5_5_TEST_CASE,
)
from integration.translation.runner import assert_translation

CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0: Final = (
    "CAQSowkKEAgSGAI4AUIIdGhpbmtpbmcSDOwQCAqgczGjZGBQDhoMT4bVPdH6VMFI48MwIjC029tj1Uerfhj93na2igkejc43fgJVSxspenJp"
    "9C9WIR+Pw8eVIU2u2oLaCRh9ZaAqwAgXjHbSSi1epawZB+b2WVRclEY69ZZaFJOtC+iA5VBnzI70bhzfYwUL2Bh9vuETb9UyiGNuXlEjTY73"
    "AUNWyLn1JGS1y4ZqwoN5u8TZV9zR7nnPoXTIdAlp8vWz0uyE2UgAlqyNX8XNEq2+CsK9R2RFskyXhd8FRSQ49QPj8e+Dcv0OLLpQrrPIa7Nv"
    "aG7A0E/5D/VEU94iC538mQ/Eh2NcwBS+gqCCGbeLhHMvmSZqDUp4s4kGMhBPILcAEx1f1FE1ZCsUSKTPMR91XJsarHkdOjgpgBx2yiNr7EAa"
    "qP8DEDTMIUZaM4rHuJfCIE2Gu/KOvBaCV5b8gl4HhvONjxfnRqoDkwB79hhB5kARXXcoYPigyJAebXTUKogtdfrLlqJ18KqFiq+gwLk2IP5v"
    "l78q3TNiDz4hWoMFdXeaffH2++6FscuZU3im68D2aSWzfUvJoG/5kqFvMKsMUMQ7JTRq8bwdy0Pzc38peH9hgy9OwYjShmmp/lX+8tH4Mpu9"
    "4ScwOx0oCKdlAJyFBd4I9kOJ03SM2bZI/NkWTZ11LrtLNqxUpdzESIuaPqcwpBlMQu09oKUthvL7rs5U4WhOD7V+l92UFphiR+hI2Wc+d7Ys"
    "kh5PgGpiIGxq5QpMGUOwgy/+/KyZcHc3EDAD4oY/BVpqc4zjjOdFOc6KvtKLn1lqZc47+XL2HxI0Kh7qPqCue3yArxkn16Jk+bdinaJPfVQI"
    "rKxIav0gWowtmxcjFOVYXRDzhlaqOfnGZOLDOwJ8TpESRBxKWJ3fYrUjLZjY6X9MdPWQQ88h/DsQijwqXNQFYs/hMOMqEYcW7oIMmbQXv/KT"
    "05nX5aMMo0AFD3Iych6bVwNC0b0S87wSolWLBBhnhYyEUJfAW6p28ARtLdkJlqrsOSupzMkKxTxjXafEbM0GcObF9FgaeDh/3gpG9ahFex8W"
    "MaBqIakbogrEeQrrt21DPgeb6lWsEMj6lEupfmY5a3Ch2fsKYIFmvvQ4JN9b5C0K7H3GFfH4qTclqTEegX4Gzm7tjuwQsv9INVilYVUkpMgo"
    "7JD52wVVtqSJqGgQ8xKLWXR1M5oMjGCE2bRhbgbVfMarvhJW17FJvS/7Yugcgxu/n/dhmo6pxFHlSujiPOBODr8fWPGJKwhpGqKdi+BUQyGF"
    "b41hPjRZEuyYSO7OdmqblZ4MlkHFLt7CfCvQQZePBvNyDniiCuwZBRgO3zlRO5VKVsom3KkzZfG+tsfzKwMMrNeN1sQ7udJsp2giBRN1wBu6"
    "LBYfs8ASv8Y1OEmFVew5/aByf7oGaH74DlxLi/E0cquY37LiUWl1iTx4SS13wv0NpMaU03X6ahmsEpdHbR2E2I8sBkn8F7b5ydnWpp9OT8Tg"
    "9tWMVfprteYjcVkJoKjkp4M7QG4jKSb3/2eofN44aPeX3KfkpbjkMONZkQ7A8z4o1ezDw2swOGVGUBgB"
)

CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_1: Final = (
    "CAQSmgcKEQgSGAI4AUIJbmFycmF0aW9uEgytCDiV0k0xcr3T9oYaDBoLSK23c8n6+DLIhCIwebnBjlb04xHa2geO2nvLWXbdL/mEVCXz+MjE"
    "oDSqFLe0Ivh+hn+mIUtg83a3B7NDKrYGXA1HllbZ1yM4570ouFk4bKliGKHWdi0coYw1u9YGlThcas5S7s1hL2+R1hqj0QBP/M/hDMFNbf/1"
    "T/idB+7QKbXFsYtonKoTlaQdwhXJehGZfIPNXFM7TYE8hB1KFnJpkJGLM2YU5j42Y1HL0kTh8lNvTdsl2GfF4elrtPAJjeAHEME2b4WfGEbp"
    "gJbPFJ6qPWOaVPJbYa7ZpdlOcGYviY9uiXDrXdUNkJx+fQgZ78pX5MIZPUeCkiGIDmxQJtuJC/961w9DEutBOy2KuaQHkuWDZqRCeFAzgW9s"
    "hmvT+jo5SxPDtn5E7KTt8DoDoClb6VNEGd+5BX9SSIfl39XyWKsrT7fs/dXDzicl4wUYbD527ZLWM6kJd3PpWcV6He3ltvELyXjYjCGv79+D"
    "pM87AwmmpkuYiuTyXrwdDBRfHSf00eoLyFzW/FZNYanwcbTTrReiW1ehhGivaLe1v0ppO2NLSHli2u1+xpdR8bK8Nn8J4q+QnCwON70fuqf+"
    "3sQLrspyOO7PlrpV2Ybq8Sng0dC0mBAAv7CpHeR1IaQPC936cfpL7bbB75/m15Pqy2R8A257/g3zZIA53oxwW2B3y/UNzL5m6XSzpmLaYmRM"
    "z4ahp50+rytQTCqjG0mr27g2vWerU8U+qqjyzTKrT795/CkVDQ2koodup6jLsfwx1W6bq5aJRVYUEY1ZJKywZylilmWrfTz8MZ4TajBx/yUu"
    "5X12jCdjzTOLVdGeOgD5a2lsxZzgU/Y9RgyzeT6RHKNnbjVUXaJa0g5K0MclCfVWFPhuIAwn2Gya/o6eF1k4Pebzr8aIOU8nY3YxXwN9hytO"
    "YCwxo4TF2eV8F9ttuBTezOFpJiG2VpvHIRh2/5AVVPxe1c2xWvZvciHHrH14G9rVxmoUzDraCoPiz2XUfCY7e58BRD+wOH3YFnf8uckCmHCj"
    "KFYclanFwFJyEfjwssCxcTZvC/AN2aATw4f5WQ1vow9uPO9wxkpc7af/VNQ2MPGB87BaVg/hTRhSgwJLSxZIOKUKMMJzJ/4rgAZRSIJawg33"
    "ONNvhW41/bXB33KXdqbvhVIZGkK1YGv8gjMlqIdTfdYDonWeGAE="
)

CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_0: Final = (
    "CAQSqAcKEAgUGAI4AUIIdGhpbmtpbmcSDDrnR+A9F0JLEEERYhoMemipsiFcfekR06FVIjAF9b7aoMXdIj2axLxXH6SvtnJN"
    "PMzZ+UVEmEufykgHa3qHXk6GCmOJUN5/+WkTTZYqxQZNE0qPZloVI6Ob74jdmqkk2/vEs+A4gpT8bwSgKfGFChSLKG10Iq2I"
    "2w5Y11LpE3JUuBK+dEmNrZ4V6cX9Rf1pgwyGveeXEc051I83CsGZIOdIndyJD9IViVXV0pxOFuHAsmECb2KEXN+2Az6aiRWT"
    "SPnkE7B4TaLJzJ137BODz5YSg7suuvkots180cJaG0UWEUIMODYW/Mx9eNCXgaSL9NYBQQKPops3e1wfK7ORa354pvyPikJc"
    "M5Zl4J474Z8DXYzIjCTUIuDIv3PMcpuDxxZw+9JIMf0ZVq1bSuaYOSuPYJ7GHOW0k7Z4H1pyHKGXottF7JjyMVT6USISpQHc"
    "PzonwA0o0PB6h0W/lQgQWLV4bSHLlMAfxTP0gQg30ouhY1SUCwRnV8DsY98rLGYRCuHliCiroJJwLS3xgZbGX4cyeAaWwIbX"
    "WGt5UEISDNMIWVh35Tan7/QZLEp3keN3GeUpW7yqL8MMy7NcnpzgmdAQ9GI+jNhe38S0W/T4wyplOMcGeVEh6nsPIL9AYyKs"
    "oY0JRcdwHORvkt2MguBZelaEn4PzY4GjULyX1rTtYWT20RvuJ2Fy/wSk0fV3DvWdvP4//MaQHPUDg+uyGhOwDGxfaBUGRtq1"
    "ExL/pvcZPzTXuClCHNzVjA+uslYWRzZh2+0vt6a/UnfsJgppMLwNUBc++Xj/GJfTqu/L6+uz7aKLvc++T3xO9l6qnNPB6aRZ"
    "c+QzH+zkIFgPSFWtkbKa7ZDCPmFP5mSkpwy0ropNAOIn7McFFnnHhT7t4A89vDCuf7c1YREAcxsp2Vm1Oqj9y2aIetH9WlZ2"
    "/erVSCCRs0Gk/0TWl2ZmdCmlrjCfGUlYxa1Eth4TwKT8ABAx1As4DR1jCJYhGlBlu+q127kTqoHlRAjbUkZVr3QwrFsEAn1r"
    "KmpEmM/8U9Xyli5O5A4omkE3AWgCka0OjxMAEHmZ7rByPkdIk9lxziGLZmGC9coiZpmhW44R57sa6e8YqVb4A40xr9YQbZOq"
    "z/UsNEUR0+Dcaxqm8dFYGgT531lJTYyMUgTBsThtmAdm2QcKGC0RExa402yjAeQ1NP/LKT0zPzMxqTZwZQFaNkg0uDlCWR/A"
    "VAW90/cYAQ=="
)

CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_1: Final = (
    "CAQSuQcKEQgUGAI4AUIJbmFycmF0aW9uEgxDxvnmfGRjqB1YjXgaDBAm3KwEhbcyZ9hk8iIwXdrmxCfu8CzbjDnMdKDtCiZ4"
    "oUoymuQQAtHCMAOYagbLTd4dIHJxyQnViLx5uwFYKtUGljXlCPvW7reS76mNLKeZ6//CWaZHLiJJFYd+SBsb0OUn5dWrKIgD"
    "1MUzFs8ttkAMAjOsjlE7X41GV7nwQmMgC9Z/2k2mN/7aZCMPPp/sgTQ0jjxgBff6TpEDpLPmpxsPbY1olkyHrq0NANgQ14NB"
    "WpI8XWu7QUUZtE+N/36v+27NKiIepOawt8emDCbxXkc5PT3pzlB8NovWsoaLqYZQ83g4MbZK37eyO9beZnn8FnogrwZD33T7"
    "D/xBOloXXJySpJEOGL1KV8iVIPBPXD60PhMm/sNtsp7rEGxahTbUvaAD5UIOLG4VQJOYggv575i3/0w3EzUNh3PwSwHaHP2a"
    "7fUCQhlbry335gjaVesAI2S/QQy33B+s6GuMgQut+ZoReWhPFnUvPMN398tJ/u+EXgg9rNMPgEsUCZYRInkq/YJqYmiy9U0P"
    "jZU8zPc8Oyv+Lya7TKXGfNk/j37B9PlgWRJY61i83XHVhrhizifVqlTHHqHhvR/P2cXylIkiPcvhxkIZwCKTnw52v7DuKGBh"
    "tL/dNzoPWrGB8MC1WmI/9og/xOqZzFVAYXC3offSFI4zfgDw69jbW3FSuubiRu8t4cb66xYugezzn90hFkoHgzPyy3Gmphu9"
    "nPlwlZiujhF3D08LT036JaO/lg7NPXYeXEBPSHEraTx1QBgjTFDbQAQfNNiNrEIu9VFAviW0/nJQLCkByuRRglXVAIi9oEoI"
    "xVx+7LCQGuCfEWUrtA55rDvysTns1VdYSUg+D7M4O1IKhxd9EgFlDAf3lQhyoV80Tez8IcBNEmlFxPCTLVcT4DpJYzBY0w9o"
    "L1D13xpvwmCT/kmE1vp1gkxV9P1GpoOzSCYtCdfINk9c+xuDx6FJ9171uPkfLaaTwsUfiL57YmFKmTxZDx5gPnzAHkdAJJrD"
    "3ze81H4rgH/ZmBQNIUwWllUaOdyu0QOom0VudqyTVcwcH/EPFcnBeUE6Jl5nZF4HdDWcd7GQOcV6yqUzrTyQuuHRTxa4y2Sb"
    "eeuSy7HvZ8nIvKbFtp1OQdzlM0bEl0ZN90NgjWhH4BqORLaI1Mhvl4Avdg6EbH8VWV/WHWiArwTK+4tXmJztRBlj4vYmzhUP"
    "dn0uz/zxMT1g3tYnUOWAHAswL9xIHhgB"
)

CLAUDE_FABLE_5_2_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_2_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_FABLE_5_2_TEST_CASE.mock_provider_response,
        "id": "msg_011ChJEmeMRhhdsXrxiuZgGr",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "I need to read the validation file first, as the user asked, so I'll open it now, starting from "
                    "the top and covering roughly the first 60 lines where the function at line 18 sits. The test "
                    "file can wait since the user specifically said to begin with validation.\n\n"
                ),
                "signature": CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": (
                    "Next I'm reading `shop/validation.py` from the top, since `validate_quantity` is defined at line"
                    " 18 and its imports and helper checks affect how zero, negative and non-integer quantities are "
                    "handled.\n\n"
                ),
                "signature": CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_01TtEKaadQGdWr9ByZysAzi6",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 873,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 265,
            "output_tokens_details": {"thinking_tokens": 86},
            "service_tier": "standard",
            "inference_geo": "global",
            "speed": "standard",
        },
    },
    expected_litellm_response={
        **CLAUDE_FABLE_5_2_TEST_CASE.expected_litellm_response,
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "I need to read the validation file first, as the user asked, so I'll open it now, starting from "
                            "the top and covering roughly the first 60 lines where the function at line 18 sits. The test "
                            "file can wait since the user specifically said to begin with validation.\n\nNext I'm reading "
                            "`shop/validation.py` from the top, since `validate_quantity` is defined at line 18 and its "
                            "imports and helper checks affect how zero, negative and non-integer quantities are handled.\n\n"
                        ),
                        "annotations": [],
                    }
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": (
                                "I need to read the validation file first, as the user asked, so I'll open it now, starting from "
                                "the top and covering roughly the first 60 lines where the function at line 18 sits. The test "
                                "file can wait since the user specifically said to begin with validation.\n\n"
                            ),
                            "signature": CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": (
                                "Next I'm reading `shop/validation.py` from the top, since `validate_quantity` is defined at line"
                                " 18 and its imports and helper checks affect how zero, negative and non-integer quantities are "
                                "handled.\n\n"
                            ),
                            "signature": CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_1,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "toolu_01TtEKaadQGdWr9ByZysAzi6",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_01TtEKaadQGdWr9ByZysAzi6",
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
                "cache_write_tokens": 0,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
            },
            "output_tokens": 265,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 86, "text_tokens": 179},
            "total_tokens": 1142,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": (
                        "I need to read the validation file first, as the user asked, so I'll open it now, starting from "
                        "the top and covering roughly the first 60 lines where the function at line 18 sits. The test "
                        "file can wait since the user specifically said to begin with validation.\n\n"
                    ),
                    "signature": CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_0,
                },
                {
                    "type": "thinking",
                    "thinking": (
                        "Next I'm reading `shop/validation.py` from the top, since `validate_quantity` is defined at line"
                        " 18 and its imports and helper checks affect how zero, negative and non-integer quantities are "
                        "handled.\n\n"
                    ),
                    "signature": CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_1,
                },
            ],
        },
    },
)

CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_SONNET_5_5_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.mock_provider_response,
        "id": "msg_011Cg32rEsFymVyWbaNmV7tB",
        "content": [
            {
                "type": "thinking",
                "thinking": "Since the files are independent, I can just read both in the same call, "
                "listing validation first to honor the user's stated order.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "I'll check `shop/validation.py` to see how `validate_quantity` handles "
                "zero, negative, and non-integer values, and also read "
                "`tests/test_validation.py` to confirm the expected behavior for zero.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_01RoPyymXeGn7TS6uLnKMkzo",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01Cv9tx97CJhxergPDGZLTxC",
                "name": "read_file",
                "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 40},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 873,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 447,
            "output_tokens_details": {"thinking_tokens": 251},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.expected_litellm_response,
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Since the files are independent, I can just read both in the same "
                        "call, listing validation first to honor the user's stated order.\n"
                        "\n"
                        "I'll check `shop/validation.py` to see how `validate_quantity` "
                        "handles zero, negative, and non-integer values, and also read "
                        "`tests/test_validation.py` to confirm the expected behavior for "
                        "zero.\n"
                        "\n",
                        "annotations": [],
                    }
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "Since the files are independent, I can just read both in the same call, listing "
                            "validation first to honor the user's stated order.\n"
                            "\n",
                            "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": "I'll check `shop/validation.py` to see how `validate_quantity` handles zero, negative, "
                            "and non-integer values, and also read `tests/test_validation.py` to confirm the "
                            "expected behavior for zero.\n"
                            "\n",
                            "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_1,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "toolu_01RoPyymXeGn7TS6uLnKMkzo",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_01RoPyymXeGn7TS6uLnKMkzo",
                "namespace": None,
                "status": "completed",
            },
            {
                "arguments": '{"path": "tests/test_validation.py", "start_line": 1, "end_line": 40}',
                "call_id": "toolu_01Cv9tx97CJhxergPDGZLTxC",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_01Cv9tx97CJhxergPDGZLTxC",
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
            "output_tokens": 447,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 251, "text_tokens": 196},
            "total_tokens": 1324,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Since the files are independent, I can "
                    "just read both in the same call, "
                    "listing validation first to honor the "
                    "user's stated order.\n"
                    "\n",
                    "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
                },
                {
                    "type": "thinking",
                    "thinking": "I'll check `shop/validation.py` to see "
                    "how `validate_quantity` handles zero, "
                    "negative, and non-integer values, and "
                    "also read `tests/test_validation.py` "
                    "to confirm the expected behavior for "
                    "zero.\n"
                    "\n",
                    "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_1,
                },
            ],
        },
    },
)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_FABLE_5_2_CACHE_HIT_TEST_CASE, CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE],
    ids=[CLAUDE_FABLE_5_2_CACHE_HIT_TEST_CASE.id, CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE.id],
)
def test_responses_prompt_cache_hit_anthropic(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
