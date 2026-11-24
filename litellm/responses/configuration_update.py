from collections.abc import Mapping, Sequence
from typing import Final, Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.responses.additional_tools import ADDITIONAL_TOOLS_INPUT_ITEM_TYPE

CONFIGURATION_UPDATE_INPUT_ITEM_TYPE: Final = "configuration_update"


class ReasoningUpdate(TypedDict):
    effort: ReadOnly[str]


class TextUpdate(TypedDict):
    format: ReadOnly[Mapping[str, object]]


class ConfigurationUpdateItem(TypedDict):
    type: ReadOnly[Literal["configuration_update"]]
    reasoning: ReadOnly[NotRequired[ReasoningUpdate]]
    text: ReadOnly[NotRequired[TextUpdate]]


class AdditionalToolsItem(TypedDict):
    type: ReadOnly[Literal["additional_tools"]]
    role: ReadOnly[Literal["developer"]]
    tools: ReadOnly[list[Mapping[str, object]]]  # mutable-ok: wire payload, guardrails and auth read it as a JSON list


def configuration_update_item(
    *, reasoning: ReasoningUpdate | None = None, text: TextUpdate | None = None
) -> ConfigurationUpdateItem | None:
    if reasoning is None:
        return None if text is None else ConfigurationUpdateItem(type=CONFIGURATION_UPDATE_INPUT_ITEM_TYPE, text=text)
    if text is None:
        return ConfigurationUpdateItem(type=CONFIGURATION_UPDATE_INPUT_ITEM_TYPE, reasoning=reasoning)
    return ConfigurationUpdateItem(type=CONFIGURATION_UPDATE_INPUT_ITEM_TYPE, reasoning=reasoning, text=text)


def additional_tools_item(tools: Sequence[Mapping[str, object]]) -> AdditionalToolsItem:
    return AdditionalToolsItem(type=ADDITIONAL_TOOLS_INPUT_ITEM_TYPE, role="developer", tools=list(tools))
