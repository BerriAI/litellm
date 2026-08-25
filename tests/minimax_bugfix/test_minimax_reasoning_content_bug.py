"""
Repro for GitHub issue #38197: MiniMax-M2.7 returns message.content empty,
usage all zero, while reasoning_content has valid output, finish_reason=stop.

Root cause (confirmed against source + MiniMax docs, 2026-08-25):
MiniMax M2.7 (with reasoning_split unset/false, the default) returns its
answer wrapped as "<think>...</think><rest of answer>" in a single content
string. litellm.litellm_core_utils.prompt_templates.common_utils.
_parse_content_for_reasoning() splits this with a regex whose second capture
group is everything AFTER the closing </think> tag. When the model's real
answer sits entirely inside the <think> block with nothing after it, that
capture group is an empty string — so `content` comes back "" while the
actual answer is sitting, discarded, in `reasoning_content`.

Tests the parsing function directly — no network, no API key, no mock
server needed, since this isolates exactly where the bug lives.
"""
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _parse_content_for_reasoning,
)


class TestMinimaxReasoningContentBug:
    def test_answer_entirely_inside_think_tag_yields_empty_content(self):
        raw = "<think>The answer to 2+2 is 4.</think>"

        reasoning_content, content = _parse_content_for_reasoning(raw)

        print("reasoning_content:", repr(reasoning_content))
        print("content:", repr(content))

        assert reasoning_content == "The answer to 2+2 is 4."
        # Fixed: content now falls back to reasoning_content when nothing
        # follows the closing </think> tag, instead of being empty.
        assert content == "The answer to 2+2 is 4."

    def test_answer_after_think_tag_still_works(self):
        raw = "<think>Let me work this out.</think>The answer is 4."

        reasoning_content, content = _parse_content_for_reasoning(raw)

        assert reasoning_content == "Let me work this out."
        assert content == "The answer is 4."
