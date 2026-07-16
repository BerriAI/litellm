from dataclasses import dataclass, field

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

_MAX_VISIBLE_ROWS = 15

_STYLE = Style.from_dict(
    {
        "prompt": "bold",
        "highlighted": "reverse",
        "hint": "fg:ansibrightblack",
    }
)


def _fuzzy_match(query: str, choices: tuple[str, ...]) -> tuple[str, ...]:
    """fzf-style subsequence filter: query chars must appear in order (not necessarily contiguous)
    in a choice, case-insensitive. Preserves the choices' original relative order among matches."""
    if not query:
        return choices
    needle = query.lower()

    def _matches(choice: str) -> bool:
        haystack = choice.lower()
        pos = 0
        for ch in needle:
            pos = haystack.find(ch, pos)
            if pos == -1:
                return False
            pos += 1
        return True

    return tuple(choice for choice in choices if _matches(choice))


@dataclass
class _PickerState:
    all_choices: tuple[str, ...]
    multiselect: bool
    query: str = ""
    cursor: int = 0
    selected: set[str] = field(default_factory=set)

    @property
    def filtered(self) -> tuple[str, ...]:
        return _fuzzy_match(self.query, self.all_choices)


def _confirmed_selection(state: _PickerState) -> list[str]:
    if state.multiselect:
        return [choice for choice in state.all_choices if choice in state.selected]
    filtered = state.filtered
    return [filtered[state.cursor]] if filtered else []


def _cancel(event: KeyPressEvent, state: _PickerState) -> None:
    event.app.exit(exception=KeyboardInterrupt)


def _confirm(event: KeyPressEvent, state: _PickerState) -> None:
    event.app.exit(result=_confirmed_selection(state))


def _toggle(event: KeyPressEvent, state: _PickerState) -> None:
    if not state.multiselect:
        return
    filtered = state.filtered
    if filtered:
        state.selected.symmetric_difference_update({filtered[state.cursor]})


def _move_up(event: KeyPressEvent, state: _PickerState) -> None:
    state.cursor = max(0, state.cursor - 1)


def _move_down(event: KeyPressEvent, state: _PickerState) -> None:
    state.cursor = min(max(0, len(state.filtered) - 1), state.cursor + 1)


def _backspace(event: KeyPressEvent, state: _PickerState) -> None:
    state.query = state.query[:-1]
    state.cursor = 0


def _type_char(event: KeyPressEvent, state: _PickerState) -> None:
    if event.data and event.data.isprintable():
        state.query += event.data
        state.cursor = 0


def _build_key_bindings(state: _PickerState) -> KeyBindings:
    kb = KeyBindings()
    kb.add("c-c")(lambda event: _cancel(event, state))
    kb.add("c-d")(lambda event: _cancel(event, state))
    kb.add("enter")(lambda event: _confirm(event, state))
    kb.add("tab")(lambda event: _toggle(event, state))
    kb.add("up")(lambda event: _move_up(event, state))
    kb.add("c-p")(lambda event: _move_up(event, state))
    kb.add("down")(lambda event: _move_down(event, state))
    kb.add("c-n")(lambda event: _move_down(event, state))
    kb.add("backspace")(lambda event: _backspace(event, state))
    kb.add(Keys.Any)(lambda event: _type_char(event, state))
    return kb


def _prompt_text(state: _PickerState, message: str) -> StyleAndTextTuples:
    toggle_hint = "tab to toggle, " if state.multiselect else ""
    return [
        ("class:prompt", f"{message}: "),
        ("", state.query),
        ("class:hint", f"  ({toggle_hint}type to filter, enter to confirm)"),
    ]


def _visible_window(state: _PickerState, filtered: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
    state.cursor = min(state.cursor, len(filtered) - 1)
    start = max(0, min(state.cursor - _MAX_VISIBLE_ROWS + 1, len(filtered) - _MAX_VISIBLE_ROWS))
    return start, filtered[start : start + _MAX_VISIBLE_ROWS]


def _choices_text(state: _PickerState) -> StyleAndTextTuples:
    filtered = state.filtered
    if not filtered:
        return [("class:hint", "  (no matches)")]
    start, visible = _visible_window(state, filtered)
    lines: StyleAndTextTuples = []
    for offset, choice in enumerate(visible):
        index = start + offset
        marker = "> " if index == state.cursor else "  "
        check = ("[x] " if choice in state.selected else "[ ] ") if state.multiselect else ""
        style = "class:highlighted" if index == state.cursor else ""
        lines.append((style, f"{marker}{check}{choice}\n"))
    return lines


def fuzzy_pick(choices: tuple[str, ...], message: str, multiselect: bool) -> list[str]:
    """Interactive type-to-filter picker (fzf-style): type to narrow the list, arrow keys to move
    the highlight, Enter to confirm. In multiselect mode, Tab toggles the highlighted item into the
    result set and Enter confirms whatever has been toggled (not the bare highlight); Ctrl-C/Ctrl-D
    raise KeyboardInterrupt rather than returning an empty result, so a cancel aborts the caller
    instead of silently looping back to prompt again.
    """
    state = _PickerState(all_choices=choices, multiselect=multiselect)
    layout = Layout(
        HSplit(
            [
                Window(content=FormattedTextControl(lambda: _prompt_text(state, message)), height=1),
                Window(content=FormattedTextControl(lambda: _choices_text(state))),
            ]
        )
    )
    app: Application[list[str]] = Application(
        layout=layout, key_bindings=_build_key_bindings(state), style=_STYLE, full_screen=False
    )
    return app.run()


__all__ = ["fuzzy_pick"]
