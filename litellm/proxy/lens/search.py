import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

LensField: TypeAlias = Literal["name", "agent", "status", "schedule"]

_SETTINGS: Final = "data->'settings'"
FIELD_VALUES: Final[MappingProxyType[LensField, str]] = MappingProxyType(
    {
        "name": f"ARRAY[{_SETTINGS}->>'name']",
        "agent": f"ARRAY[{_SETTINGS}->>'q', {_SETTINGS}->>'agent_name', {_SETTINGS}->>'service']",
        "status": "ARRAY[COALESCE(data->'jobs'->0->>'status', 'never')]",
        "schedule": f"ARRAY[CASE WHEN ({_SETTINGS}->>'enabled')::boolean THEN 'watching' ELSE 'paused' END]",
    }
)

_SCOPE_LABEL: Final = f"""COALESCE(NULLIF({_SETTINGS}->>'q', ''), NULLIF(concat_ws(' · ',
    NULLIF({_SETTINGS}->>'agent_name', ''),
    NULLIF({_SETTINGS}->>'service', ''),
    (SELECT string_agg((f->>'key') || ': ' || (f->>'value'), ' · ')
     FROM jsonb_array_elements({_SETTINGS}->'filters') AS f)), ''), 'All activity')"""
FREE_TEXT: Final = f"ARRAY[{_SETTINGS}->>'name', {_SCOPE_LABEL}]"

_TOKEN: Final = re.compile(r'(?:"[^"]*"?|\S)+')


def search_terms(q: str) -> tuple[str, ...]:
    """Whitespace-separated terms of a search, keeping quoted stretches whole."""
    return tuple(_TOKEN.findall(q))


_FIELD_TOKEN: Final = re.compile(r"^(-?)([A-Za-z_]+):(.*)$", re.DOTALL)
_QUOTED: Final = re.compile(r'^"([^"]*)"?$')


@dataclass(frozen=True, slots=True)
class LensFilter:
    field: LensField
    pattern: str
    exclude: bool


@dataclass(frozen=True, slots=True)
class LensSearch:
    text: tuple[str, ...]
    filters: tuple[LensFilter, ...]


@dataclass(frozen=True, slots=True)
class _Text:
    value: str


@dataclass(frozen=True, slots=True)
class _Field:
    field: LensField
    exclude: bool
    value: str


def like_literal(value: str) -> str:
    return re.sub(r"([\\%_])", r"\\\1", value)


def _unquote(raw: str) -> str:
    match: Final = _QUOTED.match(raw)
    return match.group(1) if match else raw


def _as_field(key: str) -> LensField | None:
    match key:
        case "name" | "agent" | "status" | "schedule":
            return key
        case _:
            return None


def _clause(raw: str) -> _Text | _Field:
    match: Final = _FIELD_TOKEN.match(raw)
    field: Final = _as_field(match.group(2).lower()) if match else None
    if match is None or field is None:
        return _Text(_unquote(raw))
    return _Field(field=field, exclude=match.group(1) == "-", value=_unquote(match.group(3)))


def parse_search(q: str) -> LensSearch:
    clauses: Final = tuple(clause for clause in map(_clause, _TOKEN.findall(q)) if clause.value)
    return LensSearch(
        text=tuple(f"%{like_literal(clause.value)}%" for clause in clauses if isinstance(clause, _Text)),
        filters=tuple(
            LensFilter(field=clause.field, pattern=like_literal(clause.value).replace("*", "%"), exclude=clause.exclude)
            for clause in clauses
            if isinstance(clause, _Field)
        ),
    )


def _any_matches(values: str, placeholder: str) -> str:
    return f"EXISTS (SELECT 1 FROM unnest(array_remove({values}, '')) AS v WHERE v ILIKE {placeholder})"


def search_predicate(search: LensSearch) -> tuple[str, tuple[str, ...]]:
    terms: Final = tuple((FREE_TEXT, pattern, False) for pattern in search.text) + tuple(
        (FIELD_VALUES[f.field], f.pattern, f.exclude) for f in search.filters
    )
    predicates: Final = tuple(
        f"{'NOT ' if exclude else ''}{_any_matches(values, f'${index}')}"
        for index, (values, _, exclude) in enumerate(terms, start=1)
    )
    return " AND ".join(predicates) or "TRUE", tuple(pattern for _, pattern, _ in terms)
