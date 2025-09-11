import opcode as _opcode
from enum import IntFlag
from typing import Optional, Union

# alias to keep the 'bytecode' variable free
import bytecode as _bytecode

from .instr import DUAL_ARG_OPCODES, CellVar, FreeVar
from .utils import PY311, PY312, PY313, PY314


class CompilerFlags(IntFlag):
    """Possible values of the co_flags attribute of Code object.

    Note: We do not rely on inspect values here as some of them are missing and
    furthermore would be version dependent.

    """

    OPTIMIZED = 0x00001
    NEWLOCALS = 0x00002
    VARARGS = 0x00004
    VARKEYWORDS = 0x00008
    NESTED = 0x00010
    GENERATOR = 0x00020
    NOFREE = 0x00040
    # New in Python 3.5
    # Used for coroutines defined using async def ie native coroutine
    COROUTINE = 0x00080
    # Used for coroutines defined as a generator and then decorated using
    # types.coroutine
    ITERABLE_COROUTINE = 0x00100
    # New in Python 3.6
    # Generator defined in an async def function
    ASYNC_GENERATOR = 0x00200

    FUTURE_GENERATOR_STOP = 0x800000
    FUTURE_ANNOTATIONS = 0x1000000


UNOPTIMIZED_OPCODES = (
    _opcode.opmap["STORE_NAME"],
    _opcode.opmap["LOAD_NAME"],
    _opcode.opmap["DELETE_NAME"],
)

ASYNC_OPCODES = (
    _opcode.opmap["GET_AWAITABLE"],
    _opcode.opmap["GET_AITER"],
    _opcode.opmap["GET_ANEXT"],
    *((_opcode.opmap["BEFORE_ASYNC_WITH"],) if not PY314 else ()),  # Removed in 3.14+
    *((_opcode.opmap["SETUP_ASYNC_WITH"],) if not PY311 else ()),  # Removed in 3.11+
    _opcode.opmap["END_ASYNC_FOR"],
    *((_opcode.opmap["ASYNC_GEN_WRAP"],) if PY311 and not PY312 else ()),  # New in 3.11
)

YIELD_VALUE_OPCODE = _opcode.opmap["YIELD_VALUE"]
GENERATOR_LIKE_OPCODES = (
    *((_opcode.opmap["YIELD_FROM"],) if not PY311 else ()),  # Removed in 3.11+
    *((_opcode.opmap["RETURN_GENERATOR"],) if PY311 else ()),  # Added in 3.11+
)


def infer_flags(
    bytecode: Union[
        "_bytecode.Bytecode", "_bytecode.ConcreteBytecode", "_bytecode.ControlFlowGraph"
    ],
    is_async: Optional[bool] = None,
):
    """Infer the proper flags for a bytecode based on the instructions.

    Because the bytecode does not have enough context to guess if a function
    is asynchronous the algorithm tries to be conservative and will never turn
    a previously async code into a sync one.

    Parameters
    ----------
    bytecode : Bytecode | ConcreteBytecode | ControlFlowGraph
        Bytecode for which to infer the proper flags
    is_async : bool | None, optional
        Force the code to be marked as asynchronous if True, prevent it from
        being marked as asynchronous if False and simply infer the best
        solution based on the opcode and the existing flag if None.

    """
    flags = CompilerFlags(0)
    if not isinstance(
        bytecode,
        (_bytecode.Bytecode, _bytecode.ConcreteBytecode, _bytecode.ControlFlowGraph),
    ):
        msg = (
            "Expected a Bytecode, ConcreteBytecode or ControlFlowGraph instance not %s"
        )
        raise ValueError(msg % bytecode)

    instructions = (
        bytecode._get_instructions()
        if isinstance(bytecode, _bytecode.ControlFlowGraph)
        else bytecode
    )

    # Iterate over the instructions and inspect the arguments
    is_concrete = isinstance(bytecode, _bytecode.ConcreteBytecode)
    optimized = True
    has_free = False if not is_concrete else bytecode.cellvars and bytecode.freevars
    known_async = False
    known_generator = False
    possible_generator = False
    instr_iter = iter(instructions)
    for instr in instr_iter:
        if isinstance(
            instr,
            (
                _bytecode.SetLineno,
                _bytecode.Label,
                _bytecode.TryBegin,
                _bytecode.TryEnd,
            ),
        ):
            continue
        opcode = instr.opcode
        if opcode in UNOPTIMIZED_OPCODES:
            optimized = False
        elif opcode in ASYNC_OPCODES:
            known_async = True
        elif opcode == YIELD_VALUE_OPCODE:
            if PY311:
                while isinstance(
                    ni := next(instr_iter),
                    (
                        _bytecode.SetLineno,
                        _bytecode.Label,
                        _bytecode.TryBegin,
                        _bytecode.TryEnd,
                    ),
                ):
                    pass
                assert ni.name == "RESUME"
                if (ni.arg & 3) != 3:
                    known_generator = True
                else:
                    known_async = True
            else:
                known_generator = True
        elif opcode in GENERATOR_LIKE_OPCODES:
            possible_generator = True
        elif opcode in _opcode.hasfree:
            has_free = True
        elif (
            not is_concrete
            and opcode in DUAL_ARG_OPCODES
            and (isinstance(instr.arg[0], CellVar) or isinstance(instr.arg[1], CellVar))
        ):
            has_free = True
        elif (
            PY313
            and opcode in _opcode.haslocal
            and isinstance(instr.arg, (CellVar, FreeVar))
        ):
            has_free = True

    # Identify optimized code
    if optimized:
        flags |= CompilerFlags.OPTIMIZED

    # Check for free variables
    if not has_free:
        flags |= CompilerFlags.NOFREE

    # Copy flags for which we cannot infer the right value
    flags |= bytecode.flags & (
        CompilerFlags.NEWLOCALS
        | CompilerFlags.VARARGS
        | CompilerFlags.VARKEYWORDS
        | CompilerFlags.NESTED
    )

    # If performing inference or forcing an async behavior, first inspect
    # the flags since this is the only way to identify iterable coroutines
    if is_async in (None, True):
        if (
            bytecode.flags & CompilerFlags.COROUTINE
            or bytecode.flags & CompilerFlags.ASYNC_GENERATOR
        ):
            if known_generator:
                flags |= CompilerFlags.ASYNC_GENERATOR
            else:
                flags |= CompilerFlags.COROUTINE
        elif bytecode.flags & CompilerFlags.ITERABLE_COROUTINE:
            if known_async:
                msg = (
                    "The ITERABLE_COROUTINE flag is set but bytecode that"
                    "can only be used in async functions have been "
                    "detected. Please unset that flag before performing "
                    "inference."
                )
                raise ValueError(msg)
            flags |= CompilerFlags.ITERABLE_COROUTINE

        # If the code was not asynchronous before determine if it should now be
        # asynchronous based on the opcode and the is_async argument.
        else:
            if known_async:
                # YIELD_FROM is not allowed in async generator
                if known_generator:
                    flags |= CompilerFlags.ASYNC_GENERATOR
                else:
                    flags |= CompilerFlags.COROUTINE

            elif known_generator or possible_generator:
                if is_async:
                    if known_generator:
                        flags |= CompilerFlags.ASYNC_GENERATOR
                    else:
                        flags |= CompilerFlags.COROUTINE
                else:
                    flags |= CompilerFlags.GENERATOR

            elif is_async:
                flags |= CompilerFlags.COROUTINE

    # If the code should not be asynchronous, check first it is possible and
    # next set the GENERATOR flag if relevant
    else:
        if known_async:
            raise ValueError(
                "The is_async argument is False but bytecodes "
                "that can only be used in async functions have "
                "been detected."
            )

        if known_generator or possible_generator:
            flags |= CompilerFlags.GENERATOR

    flags |= bytecode.flags & CompilerFlags.FUTURE_GENERATOR_STOP

    return flags
