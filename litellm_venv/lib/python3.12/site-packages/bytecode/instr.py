import dis
import enum
import opcode as _opcode
import sys
from abc import abstractmethod
from dataclasses import dataclass
from marshal import dumps as _dumps
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union

try:
    from typing import TypeGuard
except ImportError:
    from typing_extensions import TypeGuard  # type: ignore

import bytecode as _bytecode
from bytecode.utils import PY311, PY312, PY313, PY314

# --- Instruction argument tools and

MIN_INSTRUMENTED_OPCODE = getattr(_opcode, "MIN_INSTRUMENTED_OPCODE", 256)

# Instructions relying on a bit to modify its behavior.
# The lowest bit is used to encode custom behavior.
BITFLAG_OPCODES = (
    (
        _opcode.opmap["BUILD_INTERPOLATION"],
        _opcode.opmap["LOAD_GLOBAL"],
        _opcode.opmap["LOAD_ATTR"],
    )
    if PY314
    else (
        (_opcode.opmap["LOAD_GLOBAL"], _opcode.opmap["LOAD_ATTR"])
        if PY312
        else ((_opcode.opmap["LOAD_GLOBAL"],) if PY311 else ())
    )
)

BITFLAG2_OPCODES = (_opcode.opmap["LOAD_SUPER_ATTR"],) if PY312 else ()

# Binary op opcode which has a dedicated arg
BINARY_OPS = (_opcode.opmap["BINARY_OP"],) if PY311 else ()

# Intrinsic related opcodes
INTRINSIC_1OP = (_opcode.opmap["CALL_INTRINSIC_1"],) if PY312 else ()
INTRINSIC_2OP = (_opcode.opmap["CALL_INTRINSIC_2"],) if PY312 else ()
INTRINSIC = INTRINSIC_1OP + INTRINSIC_2OP

# Small integer related opcode
SMALL_INT_OPS = (_opcode.opmap["LOAD_SMALL_INT"],) if PY314 else ()

# Special method loading related opcodes
SPECIAL_OPS = (_opcode.opmap["LOAD_SPECIAL"],) if PY314 else ()

# Common constant loading related opcodes
COMMON_CONSTANT_OPS = (_opcode.opmap["LOAD_COMMON_CONSTANT"],) if PY314 else ()

# Value formatting related opcodes (only handle CONVERT_VALUE and BUILD_INTERPOLATION)
FORMAT_VALUE_OPS = (
    (
        _opcode.opmap["CONVERT_VALUE"],
        _opcode.opmap["BUILD_INTERPOLATION"],
    )
    if PY314
    else ((_opcode.opmap["CONVERT_VALUE"],) if PY313 else ())
)

HASJABS = () if PY313 else _opcode.hasjabs
if sys.version_info >= (3, 13):
    HASJREL = _opcode.hasjump
else:
    HASJREL = _opcode.hasjrel

#: Opcodes taking 2 arguments (highest 4 bits and lowest 4 bits)
DUAL_ARG_OPCODES: Tuple[int, ...] = ()
DUAL_ARG_OPCODES_SINGLE_OPS: Dict[int, Tuple[str, str]] = {}
if PY313:
    DUAL_ARG_OPCODES = (
        _opcode.opmap["LOAD_FAST_LOAD_FAST"],
        _opcode.opmap["STORE_FAST_LOAD_FAST"],
        _opcode.opmap["STORE_FAST_STORE_FAST"],
    )
    if PY314:
        DUAL_ARG_OPCODES = (
            *DUAL_ARG_OPCODES,
            _opcode.opmap["LOAD_FAST_BORROW_LOAD_FAST_BORROW"],
        )
    DUAL_ARG_OPCODES_SINGLE_OPS = {
        _opcode.opmap["LOAD_FAST_LOAD_FAST"]: ("LOAD_FAST", "LOAD_FAST"),
        _opcode.opmap["STORE_FAST_LOAD_FAST"]: ("STORE_FAST", "LOAD_FAST"),
        _opcode.opmap["STORE_FAST_STORE_FAST"]: ("STORE_FAST", "STORE_FAST"),
    }


# Used for COMPARE_OP opcode argument
@enum.unique
class Compare(enum.IntEnum):
    LT = 0
    LE = 1
    EQ = 2
    NE = 3
    GT = 4
    GE = 5
    if sys.version_info < (3, 9):
        IN = 6
        NOT_IN = 7
        IS = 8
        IS_NOT = 9
        EXC_MATCH = 10

    if PY312:

        def _get_mask(self):
            v = self & 0b1111
            if v == Compare.EQ:
                return 8
            elif v == Compare.NE:
                return 1 + 2 + 4
            elif v == Compare.LT:
                return 2
            elif v == Compare.LE:
                return 2 + 8
            elif v == Compare.GT:
                return 4
            elif v == Compare.GE:
                return 4 + 8

    if PY313:
        LT_CAST = 0 + 16
        LE_CAST = 1 + 16
        EQ_CAST = 2 + 16
        NE_CAST = 3 + 16
        GT_CAST = 4 + 16
        GE_CAST = 5 + 16


# Used for BINARY_OP under Python 3.11+
@enum.unique
class BinaryOp(enum.IntEnum):
    ADD = 0
    AND = 1
    FLOOR_DIVIDE = 2
    LSHIFT = 3
    MATRIX_MULTIPLY = 4
    MULTIPLY = 5
    REMAINDER = 6
    OR = 7
    POWER = 8
    RSHIFT = 9
    SUBTRACT = 10
    TRUE_DIVIDE = 11
    XOR = 12
    INPLACE_ADD = 13
    INPLACE_AND = 14
    INPLACE_FLOOR_DIVIDE = 15
    INPLACE_LSHIFT = 16
    INPLACE_MATRIX_MULTIPLY = 17
    INPLACE_MULTIPLY = 18
    INPLACE_REMAINDER = 19
    INPLACE_OR = 20
    INPLACE_POWER = 21
    INPLACE_RSHIFT = 22
    INPLACE_SUBTRACT = 23
    INPLACE_TRUE_DIVIDE = 24
    INPLACE_XOR = 25
    if PY314:
        SUBSCR = 26


@enum.unique
class Intrinsic1Op(enum.IntEnum):
    INTRINSIC_1_INVALID = 0
    INTRINSIC_PRINT = 1
    INTRINSIC_IMPORT_STAR = 2
    INTRINSIC_STOPITERATION_ERROR = 3
    INTRINSIC_ASYNC_GEN_WRAP = 4
    INTRINSIC_UNARY_POSITIVE = 5
    INTRINSIC_LIST_TO_TUPLE = 6
    INTRINSIC_TYPEVAR = 7
    INTRINSIC_PARAMSPEC = 8
    INTRINSIC_TYPEVARTUPLE = 9
    INTRINSIC_SUBSCRIPT_GENERIC = 10
    INTRINSIC_TYPEALIAS = 11


@enum.unique
class Intrinsic2Op(enum.IntEnum):
    INTRINSIC_2_INVALID = 0
    INTRINSIC_PREP_RERAISE_STAR = 1
    INTRINSIC_TYPEVAR_WITH_BOUND = 2
    INTRINSIC_TYPEVAR_WITH_CONSTRAINTS = 3
    INTRINSIC_SET_FUNCTION_TYPE_PARAMS = 4


@enum.unique
class FormatValue(enum.IntEnum):
    STR = 1
    REPR = 2
    ASCII = 3


@enum.unique
class SpecialMethod(enum.IntEnum):
    """Special method names used with LOAD_SPECIAL"""

    ENTER = 0
    EXIT = 1
    AENTER = 2
    AEXIT = 3


@enum.unique
class CommonConstant(enum.IntEnum):
    """Common constants names used with LOAD_COMMON_CONSTANT"""

    ASSERTION_ERROR = 0
    NOT_IMPLEMENTED_ERROR = 1
    BUILTIN_TUPLE = 2
    BUILTIN_ALL = 3
    BUILTIN_ANY = 4


# This make type checking happy but means it won't catch attempt to manipulate an unset
# statically. We would need guard on object attribute narrowed down through methods
class _UNSET(int):
    instance: Optional["_UNSET"] = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __eq__(self, other) -> bool:
        return self is other


for op in [
    "__abs__",
    "__add__",
    "__and__",
    "__bool__",
    "__ceil__",
    "__divmod__",
    "__float__",
    "__floor__",
    "__floordiv__",
    "__ge__",
    "__gt__",
    "__hash__",
    "__index__",
    "__int__",
    "__invert__",
    "__le__",
    "__lshift__",
    "__lt__",
    "__mod__",
    "__mul__",
    "__ne__",
    "__neg__",
    "__or__",
    "__pos__",
    "__pow__",
    "__radd__",
    "__rand__",
    "__rdivmod__",
    "__rfloordiv__",
    "__rlshift__",
    "__rmod__",
    "__rmul__",
    "__ror__",
    "__round__",
    "__rpow__",
    "__rrshift__",
    "__rshift__",
    "__rsub__",
    "__rtruediv__",
    "__rxor__",
    "__sub__",
    "__truediv__",
    "__trunc__",
    "__xor__",
]:
    setattr(_UNSET, op, lambda *args: NotImplemented)


UNSET = _UNSET()


def const_key(obj: Any) -> Union[bytes, Tuple[type, int]]:
    try:
        return _dumps(obj)
    except ValueError:
        # For other types, we use the object identifier as an unique identifier
        # to ensure that they are seen as unequal.
        return (type(obj), id(obj))


class Label:
    __slots__ = ()


#: Placeholder label temporarily used when performing some conversions
#: concrete -> bytecode
PLACEHOLDER_LABEL = Label()


class _Variable:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name: str = name

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        return self.name == other.name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return "<%s %r>" % (self.__class__.__name__, self.name)


class CellVar(_Variable):
    __slots__ = ()


class FreeVar(_Variable):
    __slots__ = ()


def _check_arg_int(arg: Any, name: str) -> TypeGuard[int]:
    if not isinstance(arg, int):
        raise TypeError(
            "operation %s argument must be an int, got %s" % (name, type(arg).__name__)
        )

    if not (0 <= arg <= 2147483647):
        raise ValueError(
            "operation %s argument must be in the range 0..2,147,483,647" % name
        )

    return True


if sys.version_info >= (3, 12):

    def opcode_has_argument(opcode: int) -> bool:
        return opcode in dis.hasarg

else:

    def opcode_has_argument(opcode: int) -> bool:
        return opcode >= dis.HAVE_ARGUMENT


# --- Instruction stack effect impact

# We split the stack effect between the manipulations done on the stack before
# executing the instruction (fetching the elements that are going to be used)
# and what is pushed back on the stack after the execution is complete.

# Stack effects that do not depend on the argument of the instruction
STATIC_STACK_EFFECTS: Dict[str, Tuple[int, int]] = {
    "ROT_TWO": (-2, 2),
    "ROT_THREE": (-3, 3),
    "ROT_FOUR": (-4, 4),
    "DUP_TOP": (-1, 2),
    "DUP_TOP_TWO": (-2, 4),
    "GET_LEN": (-1, 2),
    "GET_ITER": (-1, 1),
    "GET_YIELD_FROM_ITER": (-1, 1),
    "GET_AWAITABLE": (-1, 1),
    "GET_AITER": (-1, 1),
    "GET_ANEXT": (-1, 2),
    "LIST_TO_TUPLE": (-1, 1),
    "LIST_EXTEND": (-2, 1),
    "SET_UPDATE": (-2, 1),
    "DICT_UPDATE": (-2, 1),
    "DICT_MERGE": (-2, 1),
    "COMPARE_OP": (-2, 1),
    "IS_OP": (-2, 1),
    "CONTAINS_OP": (-2, 1),
    "IMPORT_NAME": (-2, 1),
    "ASYNC_GEN_WRAP": (-1, 1),
    "PUSH_EXC_INFO": (-1, 2),
    # Pop TOS and push TOS.__aexit__ and result of TOS.__aenter__()
    "BEFORE_ASYNC_WITH": (-1, 2),
    # Replace TOS based on TOS and TOS1
    "IMPORT_FROM": (-1, 2),
    "COPY_DICT_WITHOUT_KEYS": (-2, 2),
    # Call a function at position 7 (4 3.11+) on the stack and push the return value
    "WITH_EXCEPT_START": (-4, 5) if PY311 else (-7, 8),
    # Starting with Python 3.11 MATCH_CLASS does not push a boolean anymore
    "MATCH_CLASS": (-3, 1 if PY311 else 2),
    "MATCH_MAPPING": (-1, 2),
    "MATCH_SEQUENCE": (-1, 2),
    "MATCH_KEYS": (-2, 3 if PY311 else 4),
    "CHECK_EXC_MATCH": (-2, 2),  # (TOS1, TOS) -> (TOS1, bool)
    "CHECK_EG_MATCH": (-2, 2),  # (TOS, TOS1) -> non-matched, matched or TOS1, None)
    "PREP_RERAISE_STAR": (-2, 1),  # (TOS1, TOS) -> new exception group)
    **dict.fromkeys((o for o in _opcode.opmap if o.startswith("UNARY_")), (-1, 1)),
    **dict.fromkeys(
        (
            o
            for o in _opcode.opmap
            if o.startswith("BINARY_") or o.startswith("INPLACE_")
        ),
        (-2, 1),
    ),
    # Python 3.12 changes not covered by dis.stack_effect
    "BINARY_SLICE": (-3, 1),
    # "STORE_SLICE" handled by dis.stack_effect
    "LOAD_FROM_DICT_OR_GLOBALS": (-1, 1),
    "LOAD_FROM_DICT_OR_DEREF": (-1, 1),
    "LOAD_INTRISIC_1": (-1, 1),
    "LOAD_INTRISIC_2": (-2, 1),
    "SET_FUNCTION_ATTRIBUTE": (-2, 1),  # new in 3.13
    "CONVERT_VALUE": (-1, 1),  # new in 3.13
    "FORMAT_SIMPLE": (-1, 1),  # new in 3.13
    "FORMAT_SPEC": (-2, 1),  # new in 3.13
    "TO_BOOL": (-1, 1),  # new in 3.13
    "BUILD_TEMPLATE": (-2, 1),  # new in 3.14
}


DYNAMIC_STACK_EFFECTS: Dict[
    str, Callable[[int, Any, Optional[bool]], Tuple[int, int]]
] = {
    # PRECALL pops all arguments (as per its stack effect) and leaves
    # the callable and either self or NULL
    # CALL pops the 2 above items and push the return
    # (when PRECALL does not exist it pops more as encoded by the effect)
    "CALL": lambda effect, arg, jump: (
        -2 - arg if PY312 else -2,
        1,
    ),
    # 3.13 only
    "CALL_KW": lambda effect, arg, jump: (-3 - arg, 1),
    # 3.12 changed the behavior of LOAD_ATTR
    "LOAD_ATTR": lambda effect, arg, jump: (-1, 1 + effect),
    "LOAD_SUPER_ATTR": lambda effect, arg, jump: (-3, 3 + effect),
    "SWAP": lambda effect, arg, jump: (-arg, arg),
    "COPY": lambda effect, arg, jump: (-arg, arg + effect),
    "ROT_N": lambda effect, arg, jump: (-arg, arg),
    "SET_ADD": lambda effect, arg, jump: (-arg, arg - 1),
    "LIST_APPEND": lambda effect, arg, jump: (-arg, arg - 1),
    "MAP_ADD": lambda effect, arg, jump: (-arg, arg - 2),
    "FORMAT_VALUE": lambda effect, arg, jump: (effect - 1, 1),
    # FOR_ITER needs TOS to be an iterator, hence a prerequisite of 1 on the stack
    "FOR_ITER": lambda effect, arg, jump: (effect, 0) if jump else (-1, 2),
    "BUILD_INTERPOLATION": lambda effect, arg, jump: (-(2 + (arg & 1)), 1),
    **{
        # Instr(UNPACK_* , n) pops 1 and pushes n
        k: lambda effect, arg, jump: (-1, effect + 1)
        for k in (
            "UNPACK_SEQUENCE",
            "UNPACK_EX",
        )
    },
    **{
        k: lambda effect, arg, jump: (effect - 1, 1)
        for k in (
            "MAKE_FUNCTION",
            "CALL_FUNCTION",
            "CALL_FUNCTION_EX",
            "CALL_FUNCTION_KW",
            "CALL_METHOD",
            *(o for o in _opcode.opmap if o.startswith("BUILD_")),
        )
    },
}


# --- Instruction location


def _check_location(
    location: Optional[int], location_name: str, min_value: int
) -> None:
    if location is None:
        return
    if not isinstance(location, int):
        raise TypeError(f"{location_name} must be an int, got {type(location)}")
    if location < min_value:
        raise ValueError(
            f"invalid {location_name}, expected >= {min_value}, got {location}"
        )


@dataclass(frozen=True)
class InstrLocation:
    """Location information for an instruction."""

    #: Lineno at which the instruction corresponds.
    #: Optional so that a location of None in an instruction encode an unset value.
    lineno: Optional[int]

    #: End lineno at which the instruction corresponds (Python 3.11+ only)
    end_lineno: Optional[int]

    #: Column offset at which the instruction corresponds (Python 3.11+ only)
    col_offset: Optional[int]

    #: End column offset at which the instruction corresponds (Python 3.11+ only)
    end_col_offset: Optional[int]

    __slots__ = ["col_offset", "end_col_offset", "end_lineno", "lineno"]

    def __init__(
        self,
        lineno: Optional[int],
        end_lineno: Optional[int],
        col_offset: Optional[int],
        end_col_offset: Optional[int],
    ) -> None:
        # Needed because we want the class to be frozen
        object.__setattr__(self, "lineno", lineno)
        object.__setattr__(self, "end_lineno", end_lineno)
        object.__setattr__(self, "col_offset", col_offset)
        object.__setattr__(self, "end_col_offset", end_col_offset)
        # In Python 3.11 0 is a valid lineno for some instructions (RESUME for example)
        _check_location(lineno, "lineno", 0 if PY311 else 1)
        _check_location(end_lineno, "end_lineno", 1)
        _check_location(col_offset, "col_offset", 0)
        _check_location(end_col_offset, "end_col_offset", 0)
        if end_lineno:
            if lineno is None:
                raise ValueError("End lineno specified with no lineno.")
            elif lineno > end_lineno:
                raise ValueError(
                    f"End lineno {end_lineno} cannot be smaller than lineno {lineno}."
                )

        if col_offset is not None or end_col_offset is not None:
            if lineno is None or end_lineno is None:
                raise ValueError(
                    "Column offsets were specified but lineno information are "
                    f"incomplete. Lineno: {lineno}, end lineno: {end_lineno}."
                )
            if end_col_offset is not None:
                if col_offset is None:
                    raise ValueError(
                        "End column offset specified with no column offset."
                    )
                # Column offset must be increasing inside a signle line but
                # have no relations between different lines.
                elif lineno == end_lineno and col_offset > end_col_offset:
                    raise ValueError(
                        f"End column offset {end_col_offset} cannot be smaller than "
                        f"column offset: {col_offset}."
                    )
            else:
                raise ValueError(
                    "No end column offset was specified but a column offset was given."
                )

    @classmethod
    def from_positions(cls, position: "dis.Positions") -> "InstrLocation":  # type: ignore
        return InstrLocation(
            position.lineno,
            position.end_lineno,
            position.col_offset,
            position.end_col_offset,
        )


class SetLineno:
    __slots__ = ("_lineno",)

    def __init__(self, lineno: int) -> None:
        # In Python 3.11 0 is a valid lineno for some instructions (RESUME for example)
        _check_location(lineno, "lineno", 0 if PY311 else 1)
        self._lineno: int = lineno

    @property
    def lineno(self) -> int:
        return self._lineno

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SetLineno):
            return False
        return self._lineno == other._lineno


# --- Pseudo instructions used to represent exception handling (3.11+)


class TryBegin:
    __slots__ = ("push_lasti", "stack_depth", "target")

    def __init__(
        self,
        target: Union[Label, "_bytecode.BasicBlock"],
        push_lasti: bool,
        stack_depth: Union[int, _UNSET] = UNSET,
    ) -> None:
        self.target: Union[Label, "_bytecode.BasicBlock"] = target
        self.push_lasti: bool = push_lasti
        self.stack_depth: Union[int, _UNSET] = stack_depth

    def copy(self) -> "TryBegin":
        return TryBegin(self.target, self.push_lasti, self.stack_depth)


class TryEnd:
    __slots__ = "entry"

    def __init__(self, entry: TryBegin) -> None:
        self.entry: TryBegin = entry

    def copy(self) -> "TryEnd":
        return TryEnd(self.entry)


T = TypeVar("T", bound="BaseInstr")
A = TypeVar("A", bound=object)


class BaseInstr(Generic[A]):
    """Abstract instruction."""

    __slots__ = ("_arg", "_location", "_name", "_opcode")

    # Work around an issue with the default value of arg
    def __init__(
        self,
        name: str,
        arg: A = UNSET,  # type: ignore
        *,
        lineno: Union[int, None, _UNSET] = UNSET,
        location: Optional[InstrLocation] = None,
    ) -> None:
        self._set(name, arg)
        if location:
            self._location = location
        elif lineno is UNSET:
            self._location = None
        else:
            self._location = InstrLocation(lineno, None, None, None)

    # Work around an issue with the default value of arg
    def set(self, name: str, arg: A = UNSET) -> None:  # type: ignore
        """Modify the instruction in-place.

        Replace name and arg attributes. Don't modify lineno.

        """
        self._set(name, arg)

    def require_arg(self) -> bool:
        """Does the instruction require an argument?"""
        return opcode_has_argument(self._opcode)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._set(name, self._arg)

    @property
    def opcode(self) -> int:
        return self._opcode

    @opcode.setter
    def opcode(self, op: int) -> None:
        if not isinstance(op, int):
            raise TypeError("operator code must be an int")
        if 0 <= op <= 255:
            name = _opcode.opname[op]
            valid = name != "<%r>" % op
        else:
            valid = False
        if not valid:
            raise ValueError("invalid operator code")

        self._set(name, self._arg)

    @property
    def arg(self) -> A:
        return self._arg

    @arg.setter
    def arg(self, arg: A):
        self._set(self._name, arg)

    @property
    def lineno(self) -> Union[int, _UNSET, None]:
        return self._location.lineno if self._location is not None else UNSET

    @lineno.setter
    def lineno(self, lineno: Union[int, _UNSET, None]) -> None:
        loc = self._location
        if loc and (
            loc.end_lineno is not None
            or loc.col_offset is not None
            or loc.end_col_offset is not None
        ):
            raise RuntimeError(
                "The lineno of an instruction with detailed location information "
                "cannot be set."
            )

        if lineno is UNSET:
            self._location = None
        else:
            self._location = InstrLocation(lineno, None, None, None)

    @property
    def location(self) -> Optional[InstrLocation]:
        return self._location

    @location.setter
    def location(self, location: Optional[InstrLocation]) -> None:
        if location and not isinstance(location, InstrLocation):
            raise TypeError(
                "The instr location must be an instance of InstrLocation or None."
            )
        self._location = location

    def stack_effect(self, jump: Optional[bool] = None) -> int:
        if not self.require_arg():
            arg = None
        # 3.11 where LOAD_GLOBAL arg encode whether or we push a null
        # 3.12 does the same for LOAD_ATTR
        # 3.14 does this for BUILD_INTERPOLATION
        elif self._opcode in BITFLAG_OPCODES and isinstance(self._arg, tuple):
            assert len(self._arg) == 2
            arg = self._arg[0]
        # 3.12 does a similar trick for LOAD_SUPER_ATTR
        elif self._opcode in BITFLAG2_OPCODES and isinstance(self._arg, tuple):
            assert len(self._arg) == 3
            arg = self._arg[0]
        elif not isinstance(self._arg, int) or self._opcode in _opcode.hasconst:
            # Argument is either a non-integer or an integer constant,
            # not oparg.
            arg = 0
        else:
            arg = self._arg

        return dis.stack_effect(self._opcode, arg, jump=jump)

    def pre_and_post_stack_effect(self, jump: Optional[bool] = None) -> Tuple[int, int]:
        # Allow to check that execution will not cause a stack underflow
        _effect = self.stack_effect(jump=jump)

        n = self.name
        if n in STATIC_STACK_EFFECTS:
            return STATIC_STACK_EFFECTS[n]
        elif n in DYNAMIC_STACK_EFFECTS:
            return DYNAMIC_STACK_EFFECTS[n](_effect, self.arg, jump)
        else:
            # For instruction with no special value we simply consider the effect apply
            # before execution
            return (_effect, 0)

    def copy(self: T) -> T:
        return self.__class__(self._name, self._arg, location=self._location)

    def has_jump(self) -> bool:
        return self._has_jump(self._opcode)

    def is_cond_jump(self) -> bool:
        """Is a conditional jump?"""
        # Ex: POP_JUMP_IF_TRUE, JUMP_IF_FALSE_OR_POP
        # IN 3.11+ the JUMP and the IF are no necessary adjacent in the name.
        name = self._name
        return "JUMP_" in name and "IF_" in name

    def is_uncond_jump(self) -> bool:
        """Is an unconditional jump?"""
        # JUMP_BACKWARD has been introduced in 3.11+
        # JUMP_ABSOLUTE was removed in 3.11+
        return self.name in {
            "JUMP_FORWARD",
            "JUMP_ABSOLUTE",
            "JUMP_BACKWARD",
            "JUMP_BACKWARD_NO_INTERRUPT",
        }

    def is_abs_jump(self) -> bool:
        """Is an absolute jump."""
        return self._opcode in HASJABS

    def is_forward_rel_jump(self) -> bool:
        """Is a forward relative jump."""
        return self._opcode in HASJREL and "BACKWARD" not in self._name

    def is_backward_rel_jump(self) -> bool:
        """Is a backward relative jump."""
        return self._opcode in HASJREL and "BACKWARD" in self._name

    def is_final(self) -> bool:
        if self._name in {
            "RETURN_VALUE",
            "RETURN_CONST",
            "RAISE_VARARGS",
            "RERAISE",
            "BREAK_LOOP",
            "CONTINUE_LOOP",
        }:
            return True
        if self.is_uncond_jump():
            return True
        return False

    def __repr__(self) -> str:
        if self._arg is not UNSET:
            return "<%s arg=%r location=%s>" % (self._name, self._arg, self._location)
        else:
            return "<%s location=%s>" % (self._name, self._location)

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        return self._cmp_key() == other._cmp_key()

    # --- Private API

    _name: str

    _location: Optional[InstrLocation]

    _opcode: int

    _arg: A

    def _set(self, name: str, arg: A) -> None:
        if not isinstance(name, str):
            raise TypeError("operation name must be a str")
        try:
            opcode = _opcode.opmap[name]
        except KeyError:
            raise ValueError(f"invalid operation name: {name}")  # noqa

        if opcode >= MIN_INSTRUMENTED_OPCODE:
            raise ValueError(
                f"operation {name} is an instrumented or pseudo opcode. "
                "Only base opcodes are supported"
            )

        self._check_arg(name, opcode, arg)

        self._name = name
        self._opcode = opcode
        self._arg = arg

    @staticmethod
    def _has_jump(opcode) -> bool:
        return opcode in _opcode.hasjrel or opcode in _opcode.hasjabs

    @abstractmethod
    def _check_arg(self, name: str, opcode: int, arg: A) -> None:
        pass

    @abstractmethod
    def _cmp_key(self) -> Tuple[Optional[InstrLocation], str, Any]:
        pass


InstrArg = Union[
    int,
    str,
    Label,
    CellVar,
    FreeVar,
    "_bytecode.BasicBlock",
    Compare,
    FormatValue,
    BinaryOp,
    Intrinsic1Op,
    Intrinsic2Op,
    CommonConstant,
    SpecialMethod,
    Tuple[bool, str],
    Tuple[bool, bool, str],
    Tuple[bool, FormatValue],
    Tuple[Union[str, CellVar, FreeVar], Union[str, CellVar, FreeVar]],
]


class Instr(BaseInstr[InstrArg]):
    __slots__ = ()

    def _cmp_key(self) -> Tuple[Optional[InstrLocation], str, Any]:
        arg: Any = self._arg
        if self._opcode in _opcode.hasconst:
            arg = const_key(arg)
        return (self._location, self._name, arg)

    def _check_arg(self, name: str, opcode: int, arg: InstrArg) -> None:  # noqa: C901
        if name == "EXTENDED_ARG":
            raise ValueError(
                "only concrete instruction can contain EXTENDED_ARG, "
                "highlevel instruction can represent arbitrary argument without it"
            )

        if opcode_has_argument(opcode):
            if arg is UNSET:
                raise ValueError("operation %s requires an argument" % name)
        else:
            if arg is not UNSET:
                raise ValueError("operation %s has no argument" % name)

        if self._has_jump(opcode):
            if not isinstance(arg, (Label, _bytecode.BasicBlock)):
                raise TypeError(
                    "operation %s argument type must be "
                    "Label or BasicBlock, got %s" % (name, type(arg).__name__)
                )

        elif opcode in _opcode.hasfree:
            if not isinstance(arg, (CellVar, FreeVar)):
                raise TypeError(
                    "operation %s argument must be CellVar "
                    "or FreeVar, got %s" % (name, type(arg).__name__)
                )

        elif opcode in _opcode.haslocal or opcode in _opcode.hasname:
            if opcode in BITFLAG_OPCODES:
                if not (
                    isinstance(arg, tuple)
                    and len(arg) == 2
                    and isinstance(arg[0], bool)
                    and isinstance(arg[1], str)
                ):
                    raise TypeError(
                        "operation %s argument must be a tuple[bool, str | FormatValue], "
                        "got %s (value=%s)" % (name, type(arg).__name__, str(arg))
                    )

            elif opcode in BITFLAG2_OPCODES:
                if not (
                    isinstance(arg, tuple)
                    and len(arg) == 3
                    and isinstance(arg[0], bool)
                    and isinstance(arg[1], bool)
                    and isinstance(arg[2], str)
                ):
                    raise TypeError(
                        "operation %s argument must be a tuple[bool, bool, str], "
                        "got %s (value=%s)" % (name, type(arg).__name__, str(arg))
                    )

            elif opcode in DUAL_ARG_OPCODES:
                if not (
                    isinstance(arg, tuple)
                    and len(arg) == 2
                    and isinstance(arg[0], (str, CellVar))
                    and isinstance(arg[1], (str, CellVar))
                ):
                    raise TypeError(
                        "operation %s argument must be a tuple[str, str], "
                        "got %s (value=%s)" % (name, type(arg).__name__, str(arg))
                    )

            elif (
                PY313
                and opcode in _opcode.haslocal
                and isinstance(arg, (CellVar, FreeVar))
            ):
                # Cell and free vars can be accessed using locals in Python 3.13+
                pass

            elif not isinstance(arg, str):
                raise TypeError(
                    "operation %s argument must be a str, "
                    "got %s" % (name, type(arg).__name__)
                )

        elif opcode in _opcode.hasconst:
            if isinstance(arg, Label):
                raise ValueError("label argument cannot be used in %s operation" % name)
            if isinstance(arg, _bytecode.BasicBlock):
                raise ValueError("block argument cannot be used in %s operation" % name)

        elif opcode in _opcode.hascompare:
            if not isinstance(arg, Compare):
                raise TypeError(
                    "operation %s argument type must be "
                    "Compare, got %s" % (name, type(arg).__name__)
                )

        elif opcode in BINARY_OPS:
            if not isinstance(arg, BinaryOp):
                if isinstance(arg, int):
                    try:
                        arg = BinaryOp(arg)
                    except Exception as e:
                        raise TypeError(
                            "operation %s argument type must be "
                            "coercible to BinaryOp, got %s" % (name, type(arg).__name__)
                        ) from e
                else:
                    raise TypeError(
                        "operation %s argument type must be "
                        "BinaryOp, got %s" % (name, type(arg).__name__)
                    )

        # We do not enforce constant immortality since which constants are
        # immortal may differ between recompilation and execution.

        elif opcode in SMALL_INT_OPS:
            if not isinstance(arg, int):
                raise TypeError(
                    "operation %s argument type must be "
                    "int, got %s" % (name, type(arg).__name__)
                )
            if arg < 0 or arg > 255:
                raise ValueError(
                    "operation %s argument type must be an "
                    "int between 0 and 255, got %s" % (name, arg)
                )

        elif opcode in INTRINSIC_1OP:
            if not isinstance(arg, Intrinsic1Op):
                raise TypeError(
                    "operation %s argument type must be "
                    "Intrinsic1Op, got %s" % (name, type(arg).__name__)
                )

        elif opcode in INTRINSIC_2OP:
            if not isinstance(arg, Intrinsic2Op):
                raise TypeError(
                    "operation %s argument type must be "
                    "Intrinsic2Op, got %s" % (name, type(arg).__name__)
                )

        elif opcode in SPECIAL_OPS:
            if not isinstance(arg, SpecialMethod):
                raise TypeError(
                    "operation %s argument type must be "
                    "SpecialMethod, got %s" % (name, type(arg).__name__)
                )
        elif opcode in COMMON_CONSTANT_OPS:
            if not isinstance(arg, CommonConstant):
                raise TypeError(
                    "operation %s argument type must be "
                    "CommonConstant, got %s" % (name, type(arg).__name__)
                )

        elif opcode in FORMAT_VALUE_OPS:
            if opcode in BITFLAG_OPCODES:
                if not (
                    isinstance(arg, tuple)
                    and len(arg) == 2
                    and isinstance(arg[0], bool)
                    and isinstance(arg[1], FormatValue)
                ):
                    raise TypeError(
                        "operation %s argument must be a tuple[bool, FormatValue], "
                        "got %s (value=%s)" % (name, type(arg).__name__, str(arg))
                    )
            elif not isinstance(arg, FormatValue):
                if isinstance(arg, int):
                    try:
                        arg = FormatValue(arg)
                    except Exception as e:
                        raise TypeError(
                            "operation %s argument must be a FormatValue] "
                            "got %s (value=%s)" % (name, type(arg).__name__, str(arg))
                        ) from e
                else:
                    raise TypeError(
                        "operation %s argument must be a FormatValue] "
                        "got %s (value=%s)" % (name, type(arg).__name__, str(arg))
                    )

        elif opcode_has_argument(opcode):
            _check_arg_int(arg, name)
