import json
import sys
from typing import Final, Literal

import pytest
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.proxy.guardrails.guardrail_hooks.custom_code.custom_code_guardrail import CustomCodeGuardrail
from litellm.litellm_core_utils.realtime_guardrails import (
    RealtimeGuardrailBlock,
    RealtimeGuardrailRewrite,
    RealtimeMessageText,
    realtime_guarded_item,
    realtime_guardrail_verdict,
    rewritten_item,
)
from litellm.types.guardrails import BlockedWord, ContentFilterAction, ContentFilterPattern, GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs


class RaisingGuardrail(CustomGuardrail):
    def __init__(self, error: Exception, realtime_violation_message: str | None = None) -> None:
        super().__init__(
            guardrail_name="raising",
            event_hook=GuardrailEventHooks.pre_call,
            default_on=True,
            realtime_violation_message=realtime_violation_message,
        )
        self.error = error

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLogging | None = None,
    ) -> GenericGuardrailAPIInputs:
        raise self.error


class MaskingGuardrail(CustomGuardrail):
    def __init__(self, word: str) -> None:
        super().__init__(guardrail_name=f"mask-{word}", event_hook=GuardrailEventHooks.pre_call, default_on=True)
        self.word = word
        self.screened: list[tuple[str, ...]] = []

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLogging | None = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = inputs.get("texts", [])
        self.screened.append(tuple(texts))
        return {**inputs, "texts": [text.replace(self.word, "[MASKED]") for text in texts]}


class LastTextRedactingGuardrail(MaskingGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLogging | None = None,
    ) -> GenericGuardrailAPIInputs:
        last: Final = inputs.get("texts", [""])[-1]
        if self.word not in last:
            return inputs
        return {**inputs, "texts": [last.replace(self.word, "[MASKED]")]}


def _item_create(item: object) -> dict:
    return {"type": "conversation.item.create", "item": item}


@pytest.mark.parametrize("role", ["user", "assistant", "system"])
def test_only_the_typed_text_of_a_message_is_screened_whatever_its_role(role: str) -> None:
    client_event: Final = _item_create(
        {
            "type": "message",
            "role": role,
            "content": [{"type": "input_audio", "audio": "AAAA"}, {"type": "input_text", "text": "hello"}],
        }
    )

    assert realtime_guarded_item(client_event) == RealtimeMessageText(role=role, texts=("hello",))


@pytest.mark.parametrize("part_type", ["output_text", "text"])
def test_assistant_text_written_by_the_client_is_screened(part_type: str) -> None:
    client_event: Final = _item_create(
        {"type": "message", "role": "assistant", "content": [{"type": part_type, "text": "hello"}]}
    )

    assert realtime_guarded_item(client_event) == RealtimeMessageText(role="assistant", texts=("hello",))


@pytest.mark.parametrize(
    "client_event",
    [
        pytest.param(
            _item_create({"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": "AAAA"}]}),
            id="audio-only user message",
        ),
        pytest.param(
            _item_create({"type": "function_call_output", "call_id": "call_1", "output": ""}), id="empty tool output"
        ),
        pytest.param(_item_create("hello"), id="item that is not an object"),
    ],
)
def test_items_without_client_text_are_not_screened(client_event: dict) -> None:
    assert realtime_guarded_item(client_event) is None


@pytest.mark.asyncio
async def test_a_plain_string_block_reason_becomes_the_violation_message() -> None:
    guardrail: Final = RaisingGuardrail(HTTPException(status_code=400, detail="hello is not allowed"))

    block: Final = await realtime_guardrail_verdict(
        ("hello",), "user", (GuardrailEventHooks.pre_call,), {}, None, (guardrail,)
    )

    assert isinstance(block, RealtimeGuardrailBlock)
    assert block.message == "hello is not allowed"


@pytest.mark.asyncio
async def test_a_configured_violation_message_replaces_the_block_reason() -> None:
    guardrail: Final = RaisingGuardrail(HTTPException(status_code=400, detail={"error": "keyword 'hello' detected"}))
    configured: Final = RaisingGuardrail(
        HTTPException(status_code=400, detail="hello is not allowed"), realtime_violation_message="Please rephrase"
    )

    block: Final = await realtime_guardrail_verdict(
        ("hello",), "user", (GuardrailEventHooks.pre_call,), {}, None, (guardrail,)
    )
    configured_block: Final = await realtime_guardrail_verdict(
        ("hello",), "user", (GuardrailEventHooks.pre_call,), {}, None, (configured,)
    )

    assert isinstance(block, RealtimeGuardrailBlock)
    assert block.message == "keyword 'hello' detected"
    assert isinstance(configured_block, RealtimeGuardrailBlock)
    assert configured_block.message == "Please rephrase"


@pytest.mark.asyncio
async def test_masked_texts_come_back_as_a_rewrite_and_reach_later_guardrails_masked() -> None:
    first: Final = MaskingGuardrail("hunter2")
    second: Final = MaskingGuardrail("swordfish")

    verdict: Final = await realtime_guardrail_verdict(
        ("my hunter2", "and swordfish", "hi"), "user", (GuardrailEventHooks.pre_call,), {}, None, (first, second)
    )

    assert verdict == RealtimeGuardrailRewrite(texts=("my [MASKED]", "and [MASKED]", "hi"))
    assert sorted(second.screened[:2]) == [("my [MASKED] and swordfish hi",), ("my [MASKED]and swordfishhi",)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parts", "screened"),
    [
        pytest.param(
            ("Compare this photo ", "with that one"),
            ("Compare this photo with that one",),
            id="a space at the boundary",
        ),
        pytest.param(
            ("Compare this photo", "with that one"),
            ("Compare this photowith that one", "Compare this photo with that one"),
            id="no space at the boundary",
        ),
    ],
)
async def test_a_clean_multi_part_item_sends_each_guardrail_only_the_joins_it_needs_one_text_per_call(
    parts: tuple[str, ...], screened: tuple[str, ...]
) -> None:
    guardrail: Final = MaskingGuardrail("hunter2")

    verdict: Final = await realtime_guardrail_verdict(
        parts, "user", (GuardrailEventHooks.pre_call,), {}, None, (guardrail,)
    )

    assert verdict is None
    assert sorted(guardrail.screened) == sorted((text,) for text in screened)


@pytest.mark.asyncio
async def test_texts_a_guardrail_leaves_unchanged_need_no_rewrite() -> None:
    verdict: Final = await realtime_guardrail_verdict(
        ("hello",), "user", (GuardrailEventHooks.pre_call,), {}, None, (MaskingGuardrail("hunter2"),)
    )

    assert verdict is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parts", "phrase"),
    [
        pytest.param(("tell me the secret", "plan"), "secret plan", id="split at a word break"),
        pytest.param(("tell me about secret-", "project"), "secret-project", id="split mid-word"),
    ],
)
async def test_a_phrase_split_across_typed_text_parts_is_still_blocked(parts: tuple[str, str], phrase: str) -> None:
    client_event: Final = _item_create(
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": part} for part in parts]}
    )
    guarded_item: Final = realtime_guarded_item(client_event)
    assert isinstance(guarded_item, RealtimeMessageText)

    phrase_filter: Final = ContentFilterGuardrail(
        guardrail_name="phrase-filter",
        blocked_words=[BlockedWord(keyword=phrase, action=ContentFilterAction.BLOCK)],
        event_hook=GuardrailEventHooks.pre_call,
        default_on=True,
    )

    verdict: Final = await realtime_guardrail_verdict(
        guarded_item.texts, guarded_item.role, (GuardrailEventHooks.pre_call,), {}, None, (phrase_filter,)
    )

    assert isinstance(verdict, RealtimeGuardrailBlock)


@pytest.mark.asyncio
async def test_a_mask_inside_one_part_keeps_every_part_in_place() -> None:
    first_photo: Final = {"type": "input_image", "image_url": "data:image/png;base64,AAA"}
    second_photo: Final = {"type": "input_image", "image_url": "data:image/png;base64,BBB"}
    client_event: Final = _item_create(
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Compare this photo"},
                first_photo,
                {"type": "input_text", "text": "with hunter2's photo"},
                second_photo,
            ],
        }
    )
    guarded_item: Final = realtime_guarded_item(client_event)
    assert isinstance(guarded_item, RealtimeMessageText)

    verdict: Final = await realtime_guardrail_verdict(
        guarded_item.texts, "user", (GuardrailEventHooks.pre_call,), {}, None, (MaskingGuardrail("hunter2"),)
    )

    assert isinstance(verdict, RealtimeGuardrailRewrite)
    assert rewritten_item(client_event, verdict.texts)["content"] == [
        {"type": "input_text", "text": "Compare this photo"},
        first_photo,
        {"type": "input_text", "text": "with [MASKED]'s photo"},
        second_photo,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parts", "word", "masked"),
    [
        pytest.param(("my secret", "plan is here"), "secret plan", "my [MASKED] is here", id="split at a word break"),
        pytest.param(("my code is hun", "ter2 ok"), "hunter2", "my code is [MASKED] ok", id="split mid-word"),
    ],
)
async def test_a_mask_across_parts_replaces_them_with_the_masked_text_the_model_would_have_read(
    parts: tuple[str, ...], word: str, masked: str
) -> None:
    verdict: Final = await realtime_guardrail_verdict(
        parts, "user", (GuardrailEventHooks.pre_call,), {}, None, (MaskingGuardrail(word),)
    )

    assert verdict == RealtimeGuardrailRewrite(texts=(masked,))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parts",
    [
        pytest.param(("abc", "123-45-6789 hunter2 rocks"), id="masks inside one part"),
        pytest.param(("abc", "123-45-6789 hunter2", "rocks"), id="a mask inside a part and one across parts"),
    ],
)
async def test_no_mask_one_text_part_gets_is_lost_when_its_joined_text_masks_differently(
    parts: tuple[str, ...],
) -> None:
    content_filter: Final = ContentFilterGuardrail(
        guardrail_name="ssn-and-keyword-mask",
        patterns=[
            ContentFilterPattern(pattern_type="prebuilt", pattern_name="us_ssn", action=ContentFilterAction.MASK)
        ],
        blocked_words=[BlockedWord(keyword="hunter2 rocks", action=ContentFilterAction.MASK)],
        event_hook=GuardrailEventHooks.pre_call,
        default_on=True,
    )

    verdict: Final = await realtime_guardrail_verdict(
        parts, "user", (GuardrailEventHooks.pre_call,), {}, None, (content_filter,)
    )

    assert isinstance(verdict, RealtimeGuardrailRewrite)
    forwarded: Final = " ".join(verdict.texts)
    assert "123-45-6789" not in forwarded
    assert "hunter2 rocks" not in forwarded
    assert forwarded.startswith("abc")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parts", "rewritten"),
    [
        pytest.param(("my code is", "hunter2"), ("my code is", "[MASKED]"), id="mask inside one part"),
        pytest.param(("my code is hun", "ter2"), ("my code is [MASKED]",), id="mask across parts"),
    ],
)
async def test_a_guardrail_that_answers_with_one_text_still_masks_parts_joined_without_a_space(
    parts: tuple[str, ...], rewritten: tuple[str, ...]
) -> None:
    verdict: Final = await realtime_guardrail_verdict(
        parts, "user", (GuardrailEventHooks.pre_call,), {}, None, (LastTextRedactingGuardrail("hunter2"),)
    )

    assert verdict == RealtimeGuardrailRewrite(texts=rewritten)


@pytest.mark.asyncio
async def test_a_mask_in_a_message_with_more_text_parts_than_the_recursion_limit_still_comes_back() -> None:
    parts: Final = (*(f"part {index}." for index in range(sys.getrecursionlimit())), "my code is hunter2.")

    verdict: Final = await realtime_guardrail_verdict(
        parts, "user", (GuardrailEventHooks.pre_call,), {}, None, (MaskingGuardrail("hunter2"),)
    )

    assert verdict == RealtimeGuardrailRewrite(texts=(*parts[:-1], "my code is [MASKED]."))


class GuardrailLogRecordingGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(guardrail_name="log-recorder", event_hook=GuardrailEventHooks.pre_call, default_on=True)
        self.guardrail_log_lengths: list[int] = []

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLogging | None = None,
    ) -> GenericGuardrailAPIInputs:
        self.guardrail_log_lengths.append(
            len(request_data["litellm_metadata"].get("standard_logging_guardrail_information", []))
        )
        return inputs


@pytest.mark.asyncio
async def test_a_guardrail_sees_only_the_current_turns_guardrail_log_however_long_the_session_runs() -> None:
    session_request_data: Final = {"litellm_metadata": {"user_api_key_team_id": "team-a"}}
    content_filter: Final = ContentFilterGuardrail(
        guardrail_name="phrase-filter",
        blocked_words=[BlockedWord(keyword="hunter2", action=ContentFilterAction.BLOCK)],
        event_hook=GuardrailEventHooks.pre_call,
        default_on=True,
    )
    recorder: Final = GuardrailLogRecordingGuardrail()

    for turn in ("hello", "how are you", "goodbye"):
        await realtime_guardrail_verdict(
            (turn,), "user", (GuardrailEventHooks.pre_call,), session_request_data, None, (content_filter, recorder)
        )

    assert recorder.guardrail_log_lengths == [1, 1, 1]
    assert session_request_data == {"litellm_metadata": {"user_api_key_team_id": "team-a"}}


class ScopedGuardrail(RaisingGuardrail):
    def __init__(
        self, *, skip_system_message: bool | None, skip_tool_message: bool | None, scan_only_tool_results: bool | None
    ) -> None:
        super().__init__(HTTPException(status_code=400, detail="blocked"))
        self.skip_system_message_in_guardrail = skip_system_message
        self.skip_tool_message_in_guardrail = skip_tool_message
        self.scan_only_tool_results = scan_only_tool_results


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "screened_roles"),
    [
        pytest.param({}, ("user", "assistant", "system", "tool"), id="no scope"),
        pytest.param({"skip_system_message": True}, ("user", "assistant", "tool"), id="skip system"),
        pytest.param({"skip_tool_message": True}, ("user", "assistant", "system"), id="skip tool"),
        pytest.param({"scan_only_tool_results": True}, ("tool",), id="only tool results"),
    ],
)
async def test_a_guardrail_screens_only_the_roles_its_scope_allows(
    scope: dict[str, bool], screened_roles: tuple[str, ...]
) -> None:
    guardrail: Final = ScopedGuardrail(
        skip_system_message=scope.get("skip_system_message"),
        skip_tool_message=scope.get("skip_tool_message"),
        scan_only_tool_results=scope.get("scan_only_tool_results"),
    )

    verdicts: Final = {
        role: await realtime_guardrail_verdict(
            ("hello",), role, (GuardrailEventHooks.pre_call,), {}, None, (guardrail,)
        )
        for role in ("user", "assistant", "system", "tool")
    }

    assert tuple(role for role, verdict in verdicts.items() if verdict is not None) == screened_roles


def test_a_rewrite_puts_the_masked_text_in_place_of_the_typed_text_parts() -> None:
    audio: Final = {"type": "input_audio", "audio": "AAAA"}
    client_event: Final = _item_create(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "a"}, audio, {"type": "input_text", "text": "b"}],
        }
    )

    assert rewritten_item(client_event, ("[MASKED] b",)) == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "[MASKED] b"}, audio],
    }


def test_a_rewrite_replaces_the_tool_output() -> None:
    client_event: Final = _item_create({"type": "function_call_output", "call_id": "call_1", "output": "hunter2"})

    assert rewritten_item(client_event, ("[MASKED]",)) == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "[MASKED]",
    }


@pytest.mark.asyncio
async def test_a_guardrail_that_crashes_is_not_treated_as_a_block() -> None:
    guardrail: Final = RaisingGuardrail(RuntimeError("guardrail backend down"))

    with pytest.raises(RuntimeError, match="guardrail backend down"):
        await realtime_guardrail_verdict(("hello",), "user", (GuardrailEventHooks.pre_call,), {}, None, (guardrail,))


_TEAM_POLICY_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    '    if request_data["team_id"] == "team-strict":\n'
    '        return block("blocked for this team")\n'
    "    return allow()\n"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("team_id", "blocked"),
    [pytest.param("team-strict", True, id="policy team"), pytest.param("team-other", False, id="other team")],
)
async def test_a_team_scoped_guardrail_sees_the_session_team(team_id: str, blocked: bool) -> None:
    team_policy = CustomCodeGuardrail(
        custom_code=_TEAM_POLICY_CODE, guardrail_name="team-policy", event_hook="pre_call", default_on=True
    )

    verdict = await realtime_guardrail_verdict(
        ("hello",),
        "user",
        (GuardrailEventHooks.pre_call,),
        {"litellm_metadata": {"user_api_key_team_id": team_id}},
        None,
        (team_policy,),
    )

    assert isinstance(verdict, RealtimeGuardrailBlock) is blocked
