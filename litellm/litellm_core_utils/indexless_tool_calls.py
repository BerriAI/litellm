from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field


@dataclass
class _ChoiceToolCalls:
    ids: dict[str, int] = field(default_factory=dict)  # mutable-ok: per-stream state
    latest: int | None = None


@dataclass
class IndexlessToolCalls:
    """Gemini's OpenAI-compatible endpoint and Mistral stream each tool call whole with an `id` and no `index`."""

    _choices: dict[int, _ChoiceToolCalls] = field(default_factory=dict)  # mutable-ok: per-stream state

    def assign(self, tool_calls: object, choice_index: object = 0) -> None:
        if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, str):
            return
        state: _ChoiceToolCalls | None = None
        for tool_call in tool_calls:
            is_mapping = isinstance(tool_call, MutableMapping)
            current = tool_call.get("index") if is_mapping else getattr(tool_call, "index", 0)
            if current is not None:
                continue
            if state is None:
                key = choice_index if isinstance(choice_index, int) else 0
                state = self._choices.setdefault(key, _ChoiceToolCalls())
            call_id = tool_call.get("id") if is_mapping else getattr(tool_call, "id", None)
            if isinstance(call_id, str) and call_id:
                index = state.ids.setdefault(call_id, len(state.ids))
            else:
                index = 0 if state.latest is None else state.latest
            if is_mapping:
                tool_call["index"] = index
            else:
                setattr(tool_call, "index", index)
            state.latest = index

    def assign_choices(self, choices: object) -> None:
        if not isinstance(choices, Sequence):
            return
        for choice in choices:
            if isinstance(choice, Mapping) and isinstance(delta := choice.get("delta"), Mapping):
                self.assign(delta.get("tool_calls"), choice.get("index", 0))
