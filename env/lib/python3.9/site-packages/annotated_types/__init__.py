import math
import sys
from dataclasses import dataclass
from datetime import timezone
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, SupportsFloat, SupportsIndex, TypeVar, Union

if sys.version_info < (3, 8):
    from typing_extensions import Protocol, runtime_checkable
else:
    from typing import Protocol, runtime_checkable

if sys.version_info < (3, 9):
    from typing_extensions import Annotated, Literal
else:
    from typing import Annotated, Literal

if sys.version_info < (3, 10):
    EllipsisType = type(Ellipsis)
    KW_ONLY = {}
    SLOTS = {}
else:
    from types import EllipsisType

    KW_ONLY = {"kw_only": True}
    SLOTS = {"slots": True}


__all__ = (
    'BaseMetadata',
    'GroupedMetadata',
    'Gt',
    'Ge',
    'Lt',
    'Le',
    'Interval',
    'MultipleOf',
    'MinLen',
    'MaxLen',
    'Len',
    'Timezone',
    'Predicate',
    'LowerCase',
    'UpperCase',
    'IsDigits',
    'IsFinite',
    'IsNotFinite',
    'IsNan',
    'IsNotNan',
    'IsInfinite',
    'IsNotInfinite',
    'doc',
    'DocInfo',
    '__version__',
)

__version__ = '0.6.0'


T = TypeVar('T')


# arguments that start with __ are considered
# positional only
# see https://peps.python.org/pep-0484/#positional-only-arguments


class SupportsGt(Protocol):
    def __gt__(self: T, __other: T) -> bool:
        ...


class SupportsGe(Protocol):
    def __ge__(self: T, __other: T) -> bool:
        ...


class SupportsLt(Protocol):
    def __lt__(self: T, __other: T) -> bool:
        ...


class SupportsLe(Protocol):
    def __le__(self: T, __other: T) -> bool:
        ...


class SupportsMod(Protocol):
    def __mod__(self: T, __other: T) -> T:
        ...


class SupportsDiv(Protocol):
    def __div__(self: T, __other: T) -> T:
        ...


class BaseMetadata:
    """Base class for all metadata.

    This exists mainly so that implementers
    can do `isinstance(..., BaseMetadata)` while traversing field annotations.
    """

    __slots__ = ()


@dataclass(frozen=True, **SLOTS)
class Gt(BaseMetadata):
    """Gt(gt=x) implies that the value must be greater than x.

    It can be used with any type that supports the ``>`` operator,
    including numbers, dates and times, strings, sets, and so on.
    """

    gt: SupportsGt


@dataclass(frozen=True, **SLOTS)
class Ge(BaseMetadata):
    """Ge(ge=x) implies that the value must be greater than or equal to x.

    It can be used with any type that supports the ``>=`` operator,
    including numbers, dates and times, strings, sets, and so on.
    """

    ge: SupportsGe


@dataclass(frozen=True, **SLOTS)
class Lt(BaseMetadata):
    """Lt(lt=x) implies that the value must be less than x.

    It can be used with any type that supports the ``<`` operator,
    including numbers, dates and times, strings, sets, and so on.
    """

    lt: SupportsLt


@dataclass(frozen=True, **SLOTS)
class Le(BaseMetadata):
    """Le(le=x) implies that the value must be less than or equal to x.

    It can be used with any type that supports the ``<=`` operator,
    including numbers, dates and times, strings, sets, and so on.
    """

    le: SupportsLe


@runtime_checkable
class GroupedMetadata(Protocol):
    """A grouping of multiple BaseMetadata objects.

    `GroupedMetadata` on its own is not metadata and has no meaning.
    All it the the constraint and metadata should be fully expressable
    in terms of the `BaseMetadata`'s returned by `GroupedMetadata.__iter__()`.

    Concrete implementations should override `GroupedMetadata.__iter__()`
    to add their own metadata.
    For example:

    >>> @dataclass
    >>> class Field(GroupedMetadata):
    >>>     gt: float | None = None
    >>>     description: str | None = None
    ...
    >>>     def __iter__(self) -> Iterable[BaseMetadata]:
    >>>         if self.gt is not None:
    >>>             yield Gt(self.gt)
    >>>         if self.description is not None:
    >>>             yield Description(self.gt)

    Also see the implementation of `Interval` below for an example.

    Parsers should recognize this and unpack it so that it can be used
    both with and without unpacking:

    - `Annotated[int, Field(...)]` (parser must unpack Field)
    - `Annotated[int, *Field(...)]` (PEP-646)
    """  # noqa: trailing-whitespace

    @property
    def __is_annotated_types_grouped_metadata__(self) -> Literal[True]:
        return True

    def __iter__(self) -> Iterator[BaseMetadata]:
        ...

    if not TYPE_CHECKING:
        __slots__ = ()  # allow subclasses to use slots

        def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
            # Basic ABC like functionality without the complexity of an ABC
            super().__init_subclass__(*args, **kwargs)
            if cls.__iter__ is GroupedMetadata.__iter__:
                raise TypeError("Can't subclass GroupedMetadata without implementing __iter__")

        def __iter__(self) -> Iterator[BaseMetadata]:  # noqa: F811
            raise NotImplementedError  # more helpful than "None has no attribute..." type errors


@dataclass(frozen=True, **KW_ONLY, **SLOTS)
class Interval(GroupedMetadata):
    """Interval can express inclusive or exclusive bounds with a single object.

    It accepts keyword arguments ``gt``, ``ge``, ``lt``, and/or ``le``, which
    are interpreted the same way as the single-bound constraints.
    """

    gt: Union[SupportsGt, None] = None
    ge: Union[SupportsGe, None] = None
    lt: Union[SupportsLt, None] = None
    le: Union[SupportsLe, None] = None

    def __iter__(self) -> Iterator[BaseMetadata]:
        """Unpack an Interval into zero or more single-bounds."""
        if self.gt is not None:
            yield Gt(self.gt)
        if self.ge is not None:
            yield Ge(self.ge)
        if self.lt is not None:
            yield Lt(self.lt)
        if self.le is not None:
            yield Le(self.le)


@dataclass(frozen=True, **SLOTS)
class MultipleOf(BaseMetadata):
    """MultipleOf(multiple_of=x) might be interpreted in two ways:

    1. Python semantics, implying ``value % multiple_of == 0``, or
    2. JSONschema semantics, where ``int(value / multiple_of) == value / multiple_of``

    We encourage users to be aware of these two common interpretations,
    and libraries to carefully document which they implement.
    """

    multiple_of: Union[SupportsDiv, SupportsMod]


@dataclass(frozen=True, **SLOTS)
class MinLen(BaseMetadata):
    """
    MinLen() implies minimum inclusive length,
    e.g. ``len(value) >= min_length``.
    """

    min_length: Annotated[int, Ge(0)]


@dataclass(frozen=True, **SLOTS)
class MaxLen(BaseMetadata):
    """
    MaxLen() implies maximum inclusive length,
    e.g. ``len(value) <= max_length``.
    """

    max_length: Annotated[int, Ge(0)]


@dataclass(frozen=True, **SLOTS)
class Len(GroupedMetadata):
    """
    Len() implies that ``min_length <= len(value) <= max_length``.

    Upper bound may be omitted or ``None`` to indicate no upper length bound.
    """

    min_length: Annotated[int, Ge(0)] = 0
    max_length: Optional[Annotated[int, Ge(0)]] = None

    def __iter__(self) -> Iterator[BaseMetadata]:
        """Unpack a Len into zone or more single-bounds."""
        if self.min_length > 0:
            yield MinLen(self.min_length)
        if self.max_length is not None:
            yield MaxLen(self.max_length)


@dataclass(frozen=True, **SLOTS)
class Timezone(BaseMetadata):
    """Timezone(tz=...) requires a datetime to be aware (or ``tz=None``, naive).

    ``Annotated[datetime, Timezone(None)]`` must be a naive datetime.
    ``Timezone[...]`` (the ellipsis literal) expresses that the datetime must be
    tz-aware but any timezone is allowed.

    You may also pass a specific timezone string or timezone object such as
    ``Timezone(timezone.utc)`` or ``Timezone("Africa/Abidjan")`` to express that
    you only allow a specific timezone, though we note that this is often
    a symptom of poor design.
    """

    tz: Union[str, timezone, EllipsisType, None]


@dataclass(frozen=True, **SLOTS)
class Predicate(BaseMetadata):
    """``Predicate(func: Callable)`` implies `func(value)` is truthy for valid values.

    Users should prefer statically inspectable metadata, but if you need the full
    power and flexibility of arbitrary runtime predicates... here it is.

    We provide a few predefined predicates for common string constraints:
    ``IsLower = Predicate(str.islower)``, ``IsUpper = Predicate(str.isupper)``, and
    ``IsDigit = Predicate(str.isdigit)``. Users are encouraged to use methods which
    can be given special handling, and avoid indirection like ``lambda s: s.lower()``.

    Some libraries might have special logic to handle certain predicates, e.g. by
    checking for `str.isdigit` and using its presence to both call custom logic to
    enforce digit-only strings, and customise some generated external schema.

    We do not specify what behaviour should be expected for predicates that raise
    an exception.  For example `Annotated[int, Predicate(str.isdigit)]` might silently
    skip invalid constraints, or statically raise an error; or it might try calling it
    and then propogate or discard the resulting exception.
    """

    func: Callable[[Any], bool]


@dataclass
class Not:
    func: Callable[[Any], bool]

    def __call__(self, __v: Any) -> bool:
        return not self.func(__v)


_StrType = TypeVar("_StrType", bound=str)

LowerCase = Annotated[_StrType, Predicate(str.islower)]
"""
Return True if the string is a lowercase string, False otherwise.

A string is lowercase if all cased characters in the string are lowercase and there is at least one cased character in the string.
"""  # noqa: E501
UpperCase = Annotated[_StrType, Predicate(str.isupper)]
"""
Return True if the string is an uppercase string, False otherwise.

A string is uppercase if all cased characters in the string are uppercase and there is at least one cased character in the string.
"""  # noqa: E501
IsDigits = Annotated[_StrType, Predicate(str.isdigit)]
"""
Return True if the string is a digit string, False otherwise.

A string is a digit string if all characters in the string are digits and there is at least one character in the string.
"""  # noqa: E501
IsAscii = Annotated[_StrType, Predicate(str.isascii)]
"""
Return True if all characters in the string are ASCII, False otherwise.

ASCII characters have code points in the range U+0000-U+007F. Empty string is ASCII too.
"""

_NumericType = TypeVar('_NumericType', bound=Union[SupportsFloat, SupportsIndex])
IsFinite = Annotated[_NumericType, Predicate(math.isfinite)]
"""Return True if x is neither an infinity nor a NaN, and False otherwise."""
IsNotFinite = Annotated[_NumericType, Predicate(Not(math.isfinite))]
"""Return True if x is one of infinity or NaN, and False otherwise"""
IsNan = Annotated[_NumericType, Predicate(math.isnan)]
"""Return True if x is a NaN (not a number), and False otherwise."""
IsNotNan = Annotated[_NumericType, Predicate(Not(math.isnan))]
"""Return True if x is anything but NaN (not a number), and False otherwise."""
IsInfinite = Annotated[_NumericType, Predicate(math.isinf)]
"""Return True if x is a positive or negative infinity, and False otherwise."""
IsNotInfinite = Annotated[_NumericType, Predicate(Not(math.isinf))]
"""Return True if x is neither a positive or negative infinity, and False otherwise."""

try:
    from typing_extensions import DocInfo, doc  # type: ignore [attr-defined]
except ImportError:

    @dataclass(frozen=True, **SLOTS)
    class DocInfo:  # type: ignore [no-redef]
        """ "
        The return value of doc(), mainly to be used by tools that want to extract the
        Annotated documentation at runtime.
        """

        documentation: str
        """The documentation string passed to doc()."""

    def doc(
        documentation: str,
    ) -> DocInfo:
        """
        Add documentation to a type annotation inside of Annotated.

        For example:

        >>> def hi(name: Annotated[int, doc("The name of the user")]) -> None: ...
        """
        return DocInfo(documentation)
