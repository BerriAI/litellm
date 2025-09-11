import dis
import inspect
import itertools
import opcode as _opcode
import struct
import sys
import types
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# alias to keep the 'bytecode' variable free
import bytecode as _bytecode
from bytecode.flags import CompilerFlags
from bytecode.instr import (
    _UNSET,
    BINARY_OPS,
    BITFLAG2_OPCODES,
    BITFLAG_OPCODES,
    COMMON_CONSTANT_OPS,
    DUAL_ARG_OPCODES,
    DUAL_ARG_OPCODES_SINGLE_OPS,
    FORMAT_VALUE_OPS,
    INTRINSIC,
    INTRINSIC_1OP,
    INTRINSIC_2OP,
    PLACEHOLDER_LABEL,
    SPECIAL_OPS,
    UNSET,
    BaseInstr,
    BinaryOp,
    CellVar,
    CommonConstant,
    Compare,
    FormatValue,
    FreeVar,
    Instr,
    InstrArg,
    InstrLocation,
    Intrinsic1Op,
    Intrinsic2Op,
    Label,
    SetLineno,
    SpecialMethod,
    TryBegin,
    TryEnd,
    _check_arg_int,
    const_key,
    opcode_has_argument,
)
from bytecode.utils import PY310, PY311, PY312, PY313

# - jumps use instruction
# - lineno use bytes (dis.findlinestarts(code))
# - dis displays bytes
OFFSET_AS_INSTRUCTION = PY310


def _set_docstring(code: _bytecode.BaseBytecode, consts: Sequence) -> None:
    if not consts:
        return
    first_const = consts[0]
    if isinstance(first_const, str) or first_const is None:
        code.docstring = first_const


T = TypeVar("T", bound="ConcreteInstr")


class ConcreteInstr(BaseInstr[int]):
    """Concrete instruction.

    arg must be an integer in the range 0..2147483647.

    It has a read-only size attribute.

    """

    # For ConcreteInstr the argument is always an integer
    _arg: int

    __slots__ = ("_extended_args", "_size")

    def __init__(
        self,
        name: str,
        arg: int = UNSET,
        *,
        lineno: Union[int, None, _UNSET] = UNSET,
        location: Optional[InstrLocation] = None,
        extended_args: Optional[int] = None,
    ):
        # Allow to remember a potentially meaningless EXTENDED_ARG emitted by
        # Python to properly compute the size and avoid messing up the jump
        # targets
        self._extended_args = extended_args
        super().__init__(name, arg, lineno=lineno, location=location)

    def _check_arg(self, name: str, opcode: int, arg: int) -> None:
        if opcode_has_argument(opcode):
            if arg is UNSET:
                raise ValueError("operation %s requires an argument" % name)

            _check_arg_int(arg, name)
        # opcode == 0 corresponds to CACHE instruction in 3.11+ and was unused before
        elif opcode == 0:
            arg = arg if arg is not UNSET else 0
            _check_arg_int(arg, name)
        else:
            if arg is not UNSET:
                raise ValueError("operation %s has no argument" % name)

    def _set(
        self,
        name: str,
        arg: int,
    ) -> None:
        super()._set(name, arg)
        size = 2
        if arg is not UNSET:
            while arg > 0xFF:
                size += 2
                arg >>= 8
        if self._extended_args is not None:
            size = 2 + 2 * self._extended_args
        self._size = size

    @property
    def size(self) -> int:
        return self._size

    def _cmp_key(self) -> Tuple[Optional[InstrLocation], str, int]:
        return (self._location, self._name, self._arg)

    def get_jump_target(self, instr_offset: int) -> Optional[int]:
        # When a jump arg is zero the jump always points to the first non-CACHE
        # opcode following the jump. The passed in offset is the offset at
        # which the jump opcode starts. So to compute the target, we add to it
        # the instruction size (accounting for extended args) and the
        # number of caches expected to follow the jump instruction.
        s = (
            (self._size // 2) if OFFSET_AS_INSTRUCTION else self._size
        ) + self.use_cache_opcodes()
        if self.is_forward_rel_jump():
            return instr_offset + s + self._arg
        if self.is_backward_rel_jump():
            return instr_offset + s - self._arg
        if self.is_abs_jump():
            return self._arg
        return None

    def assemble(self) -> bytes:
        if self._arg is UNSET:
            return bytes((self._opcode, 0))

        arg = self._arg
        b = [self._opcode, arg & 0xFF]
        while arg > 0xFF:
            arg >>= 8
            b[:0] = [_opcode.EXTENDED_ARG, arg & 0xFF]

        if self._extended_args:
            while len(b) < self._size:
                b[:0] = [_opcode.EXTENDED_ARG, 0x00]

        return bytes(b)

    @classmethod
    def disassemble(cls: Type[T], lineno: Optional[int], code: bytes, offset: int) -> T:
        index = 2 * offset if OFFSET_AS_INSTRUCTION else offset
        op = code[index]
        if opcode_has_argument(op):
            arg = code[index + 1]
        else:
            arg = UNSET
        name = _opcode.opname[op]
        return cls(name, arg, lineno=lineno)

    def use_cache_opcodes(self) -> int:
        if sys.version_info >= (3, 13):
            return (
                dis._inline_cache_entries[self._name]  # type: ignore[attr-defined]
                if self._name in dis._inline_cache_entries  # type: ignore[attr-defined]
                else 0
            )
        elif sys.version_info >= (3, 11):
            return dis._inline_cache_entries[self._opcode]  # type: ignore
        else:
            return 0


class ExceptionTableEntry:
    """Entry for a given line in the exception table.

    All offset are expressed in instructions not in bytes.

    """

    #: Offset in instruction between the beginning of the bytecode and the beginning
    #: of this entry.
    start_offset: int

    #: Offset in instruction between the beginning of the bytecode and the end
    #: of this entry. This offset is inclusive meaning that the instruction it points
    #: to is included in the try/except handling.
    stop_offset: int

    #: Offset in instruction to the first instruction of the exception handling block.
    target: int

    #: Minimal stack depth in the block delineated by start and stop
    #: offset of the exception table entry. Used to restore the stack (by
    #: popping items) when entering the exception handling block.
    stack_depth: int

    #: Should the offset, at which an exception was raised, be pushed on the stack
    #: before the exception itself (which is pushed as a single value)).
    push_lasti: bool

    __slots__ = ("push_lasti", "stack_depth", "start_offset", "stop_offset", "target")

    def __init__(
        self,
        start_offset: int,
        stop_offset: int,
        target: int,
        stack_depth: int,
        push_lasti: bool,
    ) -> None:
        self.start_offset = start_offset
        self.stop_offset = stop_offset
        self.target = target
        self.stack_depth = stack_depth
        self.push_lasti = push_lasti

    def __repr__(self) -> str:
        return (
            "ExceptionTableEntry("
            f"start_offset={self.start_offset}, "
            f"stop_offset={self.stop_offset}, "
            f"target={self.target}, "
            f"stack_depth={self.stack_depth}, "
            f"push_lasti={self.push_lasti}"
        )


class ConcreteBytecode(_bytecode._BaseBytecodeList[Union[ConcreteInstr, SetLineno]]):
    #: List of "constant" objects for the bytecode
    consts: List

    #: List of names used by local variables.
    names: List[str]

    #: List of names used by input variables.
    varnames: List[str]

    #: Table describing portion of the bytecode in which exceptions are caught and
    #: where there are handled.
    #: Used only in Python 3.11+
    exception_table: List[ExceptionTableEntry]

    def __init__(
        self,
        instructions=(),
        *,
        consts: tuple = (),
        names: Tuple[str, ...] = (),
        varnames: Iterable[str] = (),
        exception_table: Optional[List[ExceptionTableEntry]] = None,
    ):
        super().__init__()
        self.consts = list(consts)
        self.names = list(names)
        self.varnames = list(varnames)
        self.exception_table = exception_table or []
        for instr in instructions:
            self._check_instr(instr)
        self.extend(instructions)

    def __iter__(self) -> Iterator[Union[ConcreteInstr, SetLineno]]:
        instructions = super().__iter__()
        for instr in instructions:
            self._check_instr(instr)
            yield instr

    def _check_instr(self, instr: Any) -> None:
        if not isinstance(instr, (ConcreteInstr, SetLineno)):
            raise ValueError(
                "ConcreteBytecode must only contain "
                "ConcreteInstr and SetLineno objects, "
                "but %s was found" % type(instr).__name__
            )

    def _copy_attr_from(self, bytecode):
        super()._copy_attr_from(bytecode)
        if isinstance(bytecode, ConcreteBytecode):
            self.consts = bytecode.consts
            self.names = bytecode.names
            self.varnames = bytecode.varnames

    def __repr__(self) -> str:
        return "<ConcreteBytecode instr#=%s>" % len(self)

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False

        const_keys1 = list(map(const_key, self.consts))
        const_keys2 = list(map(const_key, other.consts))
        if const_keys1 != const_keys2:
            return False

        if self.names != other.names:
            return False
        if self.varnames != other.varnames:
            return False

        return super().__eq__(other)

    @staticmethod
    def from_code(
        code: types.CodeType, *, extended_arg: bool = False
    ) -> "ConcreteBytecode":
        instructions: MutableSequence[Union[SetLineno, ConcreteInstr]]
        # For Python 3.11+ we use dis to extract the detailed location information at
        # reduced maintenance cost.
        if PY311:
            instructions = []
            for i in dis.get_instructions(code, show_caches=True):
                loc = InstrLocation.from_positions(i.positions) if i.positions else None
                # dis.get_instructions automatically handle extended arg which
                # we do not want, so we fold back arguments to be between 0 and 255
                instructions.append(
                    ConcreteInstr(
                        i.opname,
                        i.arg % 256 if i.arg is not None else UNSET,
                        location=loc,
                    )
                )
                # cache_info only exist on 3.13+
                for _, size, _ in (i.cache_info or ()) if PY313 else ():  # type: ignore
                    for _ in range(size):
                        instructions.append(ConcreteInstr("CACHE", 0, location=loc))
        else:
            if PY310:
                line_starts = {offset: lineno for offset, _, lineno in code.co_lines()}
            else:
                line_starts = dict(dis.findlinestarts(code))

            # find block starts
            instructions = []
            offset = 0
            lineno: Optional[int] = code.co_firstlineno
            while offset < (len(code.co_code) // (2 if OFFSET_AS_INSTRUCTION else 1)):
                lineno_off = (2 * offset) if OFFSET_AS_INSTRUCTION else offset
                if lineno_off in line_starts:
                    lineno = line_starts[lineno_off]

                instr = ConcreteInstr.disassemble(lineno, code.co_code, offset)

                instructions.append(instr)
                offset += (instr.size // 2) if OFFSET_AS_INSTRUCTION else instr.size

        bytecode = ConcreteBytecode()

        # HINT : in some cases Python generate useless EXTENDED_ARG opcode
        # with a value of zero. Such opcodes do not increases the size of the
        # following opcode the way a normal EXTENDED_ARG does. As a
        # consequence, they need to be tracked manually as otherwise the
        # offsets in jump targets can end up being wrong.
        if not extended_arg:
            # The list is modified in place
            bytecode._remove_extended_args(instructions)

        bytecode.name = code.co_name
        bytecode.filename = code.co_filename
        bytecode.flags = CompilerFlags(code.co_flags)
        bytecode.argcount = code.co_argcount
        bytecode.posonlyargcount = code.co_posonlyargcount
        bytecode.kwonlyargcount = code.co_kwonlyargcount
        bytecode.first_lineno = code.co_firstlineno
        bytecode.names = list(code.co_names)
        bytecode.consts = list(code.co_consts)
        bytecode.varnames = list(code.co_varnames)
        bytecode.freevars = list(code.co_freevars)
        bytecode.cellvars = list(code.co_cellvars)
        _set_docstring(bytecode, code.co_consts)
        if PY311:
            bytecode.exception_table = bytecode._parse_exception_table(
                code.co_exceptiontable
            )
            bytecode.qualname = code.co_qualname
        else:
            bytecode.qualname = bytecode.qualname

        bytecode[:] = instructions
        return bytecode

    @staticmethod
    def _normalize_lineno(
        instructions: Sequence[Union[ConcreteInstr, SetLineno]], first_lineno: int
    ) -> Iterator[Tuple[int, ConcreteInstr]]:
        lineno = first_lineno
        # For each instruction compute an "inherited" lineno used:
        # - on 3.8 and 3.9 for which a lineno is mandatory
        # - to infer a lineno on 3.10+ if no lineno was provided
        for instr in instructions:
            i_lineno = instr.lineno
            # if instr.lineno is not set, it's inherited from the previous
            # instruction, or from self.first_lineno
            if i_lineno is not None and i_lineno is not UNSET:
                lineno = i_lineno

            if isinstance(instr, ConcreteInstr):
                yield (lineno, instr)

    def _assemble_code(
        self,
    ) -> Tuple[bytes, List[Tuple[int, int, int, Optional[InstrLocation]]]]:
        offset = 0
        code_str = []
        linenos = []
        for lineno, instr in self._normalize_lineno(self, self.first_lineno):
            code_str.append(instr.assemble())
            i_size = instr.size
            linenos.append(
                (
                    (offset * 2) if OFFSET_AS_INSTRUCTION else offset,
                    i_size,
                    lineno,
                    instr.location,
                )
            )
            offset += (i_size // 2) if OFFSET_AS_INSTRUCTION else i_size

        return (b"".join(code_str), linenos)

    # Used on 3.8 and 3.9
    @staticmethod
    def _assemble_lnotab(
        first_lineno: int, linenos: List[Tuple[int, int, int, Optional[InstrLocation]]]
    ) -> bytes:
        lnotab = []
        old_offset = 0
        old_lineno = first_lineno
        for offset, _, lineno, _ in linenos:
            dlineno = lineno - old_lineno
            if dlineno == 0:
                continue
            old_lineno = lineno

            doff = offset - old_offset
            old_offset = offset

            while doff > 255:
                lnotab.append(b"\xff\x00")
                doff -= 255

            while dlineno < -128:
                lnotab.append(struct.pack("Bb", doff, -128))
                doff = 0
                dlineno -= -128

            while dlineno > 127:
                lnotab.append(struct.pack("Bb", doff, 127))
                doff = 0
                dlineno -= 127

            assert 0 <= doff <= 255
            assert -128 <= dlineno <= 127

            lnotab.append(struct.pack("Bb", doff, dlineno))

        return b"".join(lnotab)

    @staticmethod
    def _pack_linetable(
        linetable: List[bytes], doff: int, dlineno: Optional[int]
    ) -> None:
        if dlineno is not None:
            # Ensure linenos are between -126 and +126, by using 127 lines jumps with
            # a 0 byte offset
            while dlineno < -127:
                linetable.append(struct.pack("Bb", 0, -127))
                dlineno -= -127

            while dlineno > 127:
                linetable.append(struct.pack("Bb", 0, 127))
                dlineno -= 127

            assert -127 <= dlineno <= 127
        else:
            dlineno = -128

        # Ensure offsets are less than 255.
        # If an offset is larger, we first mark the line change with an offset of 254
        # then use as many 254 offset with no line change to reduce the offset to
        # less than 254.
        if doff > 254:
            linetable.append(struct.pack("Bb", 254, dlineno))
            doff -= 254

            while doff > 254:
                linetable.append(b"\xfe\x00")
                doff -= 254
            linetable.append(struct.pack("Bb", doff, 0))

        else:
            linetable.append(struct.pack("Bb", doff, dlineno))

        assert 0 <= doff <= 254

    # Used on 3.10
    def _assemble_linestable(
        self,
        first_lineno: int,
        linenos: Iterable[Tuple[int, int, int, Optional[InstrLocation]]],
    ) -> bytes:
        if not linenos:
            return b""

        linetable: List[bytes] = []
        old_offset = 0

        iter_in = iter(linenos)

        offset, i_size, old_lineno, old_location = next(iter_in)
        if old_location is not None:
            old_dlineno = (
                old_location.lineno - first_lineno
                if old_location.lineno is not None
                else None
            )
        else:
            old_dlineno = old_lineno - first_lineno

        # i_size is used after we exit the loop
        for offset, i_size, lineno, location in iter_in:  # noqa
            if location is not None:
                dlineno = (
                    location.lineno - old_lineno
                    if location.lineno is not None
                    else None
                )
            else:
                dlineno = lineno - old_lineno

            if dlineno == 0 or (old_dlineno is None and dlineno is None):
                continue
            old_lineno = lineno

            doff = offset - old_offset
            old_offset = offset

            self._pack_linetable(linetable, doff, old_dlineno)
            old_dlineno = dlineno

        # Pack the line of the last instruction.
        doff = offset + i_size - old_offset
        self._pack_linetable(linetable, doff, old_dlineno)

        return b"".join(linetable)

    # The formats are describes in CPython/Objects/locations.md
    @staticmethod
    def _encode_location_varint(varint: int) -> bytearray:
        encoded = bytearray()
        # We encode on 6 bits
        while True:
            encoded.append(varint & 0x3F)
            varint >>= 6
            if varint:
                encoded[-1] |= 0x40  # bit 6 is set except on the last entry
            else:
                break
        return encoded

    def _encode_location_svarint(self, svarint: int) -> bytearray:
        if svarint < 0:
            return self._encode_location_varint(((-svarint) << 1) | 1)
        else:
            return self._encode_location_varint(svarint << 1)

    # Python 3.11+ location format encoding
    @staticmethod
    def _pack_location_header(code: int, size: int) -> int:
        return (1 << 7) + (code << 3) + (size - 1 if size <= 8 else 7)

    def _pack_location(
        self, size: int, lineno: int, location: Optional[InstrLocation]
    ) -> bytearray:
        packed = bytearray()

        l_lineno: Optional[int]
        # The location was not set so we infer a line.
        if location is None:
            l_lineno, end_lineno, col_offset, end_col_offset = (
                lineno,
                None,
                None,
                None,
            )
        else:
            l_lineno, end_lineno, col_offset, end_col_offset = (
                location.lineno,
                location.end_lineno,
                location.col_offset,
                location.end_col_offset,
            )

        # We have no location information so the code is 15
        if l_lineno is None:
            packed.append(self._pack_location_header(15, size))

        # No column info, code 13
        elif col_offset is None:
            if end_lineno is not None and end_lineno != l_lineno:
                raise ValueError(
                    "An instruction cannot have no column offset and span "
                    f"multiple lines (lineno: {l_lineno}, end lineno: {end_lineno}"
                )
            packed.extend(
                (
                    self._pack_location_header(13, size),
                    *self._encode_location_svarint(l_lineno - lineno),
                )
            )

        # We enforce the end_lineno to be defined
        else:
            assert end_lineno is not None
            assert end_col_offset is not None

            # Short forms
            if (
                end_lineno == l_lineno
                and l_lineno - lineno == 0
                and col_offset < 72
                and (end_col_offset - col_offset) <= 15
            ):
                packed.extend(
                    (
                        self._pack_location_header(col_offset // 8, size),
                        ((col_offset % 8) << 4) + (end_col_offset - col_offset),
                    )
                )

            # One line form
            elif (
                end_lineno == l_lineno
                and l_lineno - lineno in (1, 2)
                and col_offset < 256
                and end_col_offset < 256
            ):
                packed.extend(
                    (
                        self._pack_location_header(10 + l_lineno - lineno, size),
                        col_offset,
                        end_col_offset,
                    )
                )

            # Long form
            else:
                packed.extend(
                    (
                        self._pack_location_header(14, size),
                        *self._encode_location_svarint(l_lineno - lineno),
                        *self._encode_location_varint(end_lineno - l_lineno),
                        # When decoding in codeobject.c::advance_with_locations
                        # we remove 1 from the offset ...
                        *self._encode_location_varint(col_offset + 1),
                        *self._encode_location_varint(end_col_offset + 1),
                    )
                )

        return packed

    def _push_locations(
        self,
        locations: List[bytearray],
        size: int,
        lineno: int,
        location: InstrLocation,
    ) -> int:
        # We need the size in instruction not in bytes
        size //= 2

        # Repeatedly add element since we cannot cover more than 8 code
        # elements. We recompute each time since in practice we will
        # rarely loop.
        while True:
            locations.append(self._pack_location(size, lineno, location))
            # Update the lineno since if we need more than one entry the
            # reference for the delta of the lineno change
            lineno = location.lineno if location.lineno is not None else lineno
            size -= 8
            if size < 1:
                break

        return lineno

    def _assemble_locations(
        self,
        first_lineno: int,
        linenos: Iterable[Tuple[int, int, int, Optional[InstrLocation]]],
    ) -> bytes:
        if not linenos:
            return b""

        locations: List[bytearray] = []

        iter_in = iter(linenos)

        _, size, lineno, old_location = next(iter_in)
        # Infer the line if location is None
        old_location = old_location or InstrLocation(lineno, None, None, None)
        lineno = first_lineno

        # We track the last set lineno to be able to compute deltas
        for _, i_size, _, location in iter_in:
            # Infer the location if location is None
            location = location or old_location

            # Group together instruction with equivalent locations
            if old_location.lineno is not None and old_location == location:
                size += i_size
                continue

            lineno = self._push_locations(locations, size, lineno, old_location)

            size = i_size
            old_location = location

        # Pack the line of the last instruction.
        self._push_locations(locations, size, lineno, old_location)

        return b"".join(locations)

    @staticmethod
    def _remove_extended_args(
        instructions: MutableSequence[Union[SetLineno, ConcreteInstr]],
    ) -> None:
        # replace jump targets with blocks
        # HINT : in some cases Python generate useless EXTENDED_ARG opcode
        # with a value of zero. Such opcodes do not increases the size of the
        # following opcode the way a normal EXTENDED_ARG does. As a
        # consequence, they need to be tracked manually as otherwise the
        # offsets in jump targets can end up being wrong.
        nb_extended_args = 0
        extended_arg = None
        index = 0
        while index < len(instructions):
            instr = instructions[index]

            # Skip SetLineno meta instruction
            if isinstance(instr, SetLineno):
                index += 1
                continue

            if instr.name == "EXTENDED_ARG":
                nb_extended_args += 1
                if extended_arg is not None:
                    extended_arg = (extended_arg << 8) + instr.arg
                else:
                    extended_arg = instr.arg

                del instructions[index]
                continue

            if extended_arg is not None:
                arg = UNSET if instr.name == "NOP" else (extended_arg << 8) + instr.arg
                extended_arg = None

                instr = ConcreteInstr(
                    instr.name,
                    arg,
                    location=instr.location,
                    extended_args=nb_extended_args,
                )
                instructions[index] = instr
                nb_extended_args = 0

            index += 1

        if extended_arg is not None:
            raise ValueError("EXTENDED_ARG at the end of the code")

    # Taken and adapted from exception_handling_notes.txt in cpython/Objects
    @staticmethod
    def _parse_varint(except_table_iterator: Iterator[int]) -> int:
        b = next(except_table_iterator)
        val = b & 63
        while b & 64:
            val <<= 6
            b = next(except_table_iterator)
            val |= b & 63
        return val

    def _parse_exception_table(
        self, exception_table: bytes
    ) -> List[ExceptionTableEntry]:
        assert PY311
        table = []
        iterator = iter(exception_table)
        try:
            while True:
                start = self._parse_varint(iterator)
                length = self._parse_varint(iterator)
                end = start + length - 1  # Present as inclusive
                target = self._parse_varint(iterator)
                dl = self._parse_varint(iterator)
                depth = dl >> 1
                lasti = bool(dl & 1)
                table.append(ExceptionTableEntry(start, end, target, depth, lasti))
        except StopIteration:
            return table

    @staticmethod
    def _encode_varint(value: int, set_begin_marker: bool = False) -> Iterator[int]:
        # Encode value as a varint on 7 bits (MSB should come first) and set
        # the begin marker if requested.
        temp: List[int] = []
        assert value >= 0
        while value:
            temp.append(value & 63 | (64 if temp else 0))
            value >>= 6
        temp = temp or [0]
        if set_begin_marker:
            temp[-1] |= 128
        return reversed(temp)

    def _assemble_exception_table(self) -> bytes:
        table = bytearray()
        for entry in self.exception_table or []:
            size = entry.stop_offset - entry.start_offset + 1
            depth = (entry.stack_depth << 1) + entry.push_lasti
            table.extend(self._encode_varint(entry.start_offset, True))
            table.extend(self._encode_varint(size))
            table.extend(self._encode_varint(entry.target))
            table.extend(self._encode_varint(depth))

        return bytes(table)

    def compute_stacksize(self, *, check_pre_and_post: bool = True) -> int:
        bytecode = self.to_bytecode()
        cfg = _bytecode.ControlFlowGraph.from_bytecode(bytecode)
        return cfg.compute_stacksize(check_pre_and_post=check_pre_and_post)

    def to_code(
        self,
        stacksize: Optional[int] = None,
        *,
        check_pre_and_post: bool = True,
        compute_exception_stack_depths: bool = True,
    ) -> types.CodeType:
        # Prevent reconverting the concrete bytecode to bytecode and cfg to do the
        # calculation if we need to do it.
        if stacksize is None or (PY311 and compute_exception_stack_depths):
            cfg = _bytecode.ControlFlowGraph.from_bytecode(self.to_bytecode())
            stacksize = cfg.compute_stacksize(
                check_pre_and_post=check_pre_and_post,
                compute_exception_stack_depths=compute_exception_stack_depths,
            )
            self = cfg.to_bytecode().to_concrete_bytecode(
                compute_exception_stack_depths=False
            )

        # Assemble the code string after round tripping to CFG if necessary.
        code_str, linenos = self._assemble_code()

        lnotab = (
            self._assemble_locations(self.first_lineno, linenos)
            if PY311
            else (
                self._assemble_linestable(self.first_lineno, linenos)
                if PY310
                else self._assemble_lnotab(self.first_lineno, linenos)
            )
        )
        nlocals = len(self.varnames)

        if sys.version_info >= (3, 11):
            return types.CodeType(
                self.argcount,
                self.posonlyargcount,
                self.kwonlyargcount,
                nlocals,
                stacksize,
                int(self.flags),
                code_str,
                tuple(self.consts),
                tuple(self.names),
                tuple(self.varnames),
                self.filename,
                self.name,
                self.qualname,
                self.first_lineno,
                lnotab,
                self._assemble_exception_table(),
                tuple(self.freevars),
                tuple(self.cellvars),
            )
        else:
            return types.CodeType(
                self.argcount,
                self.posonlyargcount,
                self.kwonlyargcount,
                nlocals,
                stacksize,
                int(self.flags),
                code_str,
                tuple(self.consts),
                tuple(self.names),
                tuple(self.varnames),
                self.filename,
                self.name,
                self.first_lineno,
                lnotab,
                tuple(self.freevars),
                tuple(self.cellvars),
            )

    def to_bytecode(
        self,
        prune_caches: bool = True,
        conserve_exception_block_stackdepth: bool = False,
    ) -> _bytecode.Bytecode:
        # On 3.11 we generate pseudo-instruction from the exception table

        # Copy instruction and remove extended args if any (in-place)
        c_instructions = self[:]
        self._remove_extended_args(c_instructions)

        # Find jump targets
        jump_targets: Set[int] = set()
        offset = 0
        for c_instr in c_instructions:
            if isinstance(c_instr, SetLineno):
                continue
            target = c_instr.get_jump_target(offset)
            if target is not None:
                jump_targets.add(target)
            offset += (c_instr.size // 2) if OFFSET_AS_INSTRUCTION else c_instr.size

        # On 3.11+ we need to also look at the exception table for jump targets
        for ex_entry in self.exception_table:
            jump_targets.add(ex_entry.target)

        # Create look up dict to find entries based on either exception handling
        # block exit or entry offsets. Several blocks can end on the same instruction
        # so we store a list of entry per offset.
        ex_start: Dict[int, ExceptionTableEntry] = {}
        ex_end: Dict[int, List[ExceptionTableEntry]] = {}
        for entry in self.exception_table:
            # Ensure we do not have more than one entry with identical starting
            # offsets
            assert entry.start_offset not in ex_start
            ex_start[entry.start_offset] = entry
            ex_end.setdefault(entry.stop_offset, []).append(entry)

        # Create labels and instructions
        jumps: List[Tuple[int, int]] = []
        instructions: List[Union[Instr, Label, TryBegin, TryEnd, SetLineno]] = []
        labels = {}
        tb_instrs: Dict[ExceptionTableEntry, TryBegin] = {}
        offset = 0

        # In Python 3.11+ cell and varnames can be shared and are indexed in a single
        # array.
        # As a consequence, the instruction argument can be either:
        # - < len(varnames): the name is shared an we can directly use
        #   the index to access the name in cellvars
        # - > len(varnames): the name is not shared and is offset by the
        #   number unshared varname.
        # Free vars are never shared and correspond to index larger than the
        # largest cell var.
        # See PyCode_NewWithPosOnlyArgs
        if PY311:
            cells_lookup = self.varnames + [
                CellVar(n) for n in self.cellvars if n not in self.varnames
            ]
            ncells = len(cells_lookup)
        else:
            ncells = len(self.cellvars)
            cells_lookup = [CellVar(n) for n in self.cellvars]

        # In Python 3.13+ LOAD_FAST can be used to retrieve cell values
        locals_lookup: Sequence[Union[str, CellVar, FreeVar]]
        if PY313:
            locals_lookup = cells_lookup + [
                FreeVar(n) for n in self.freevars if n not in self.varnames
            ]
        else:
            locals_lookup = self.varnames

        for lineno, c_instr in self._normalize_lineno(
            c_instructions, self.first_lineno
        ):
            if offset in jump_targets:
                label = Label()
                labels[offset] = label
                instructions.append(label)

            # Handle TryBegin pseudo instructions
            if offset in ex_start:
                entry = ex_start[offset]
                # Check if the try begin was already created by an entry
                # with a end offset less or equal to the start offset.
                if entry not in tb_instrs:
                    tb_instr = TryBegin(
                        Label(),
                        entry.push_lasti,
                        entry.stack_depth
                        if conserve_exception_block_stackdepth
                        else UNSET,
                    )
                    # Per entry store the pseudo instruction associated
                    tb_instrs[entry] = tb_instr
                    instructions.append(tb_instr)

            jump_target = c_instr.get_jump_target(offset)
            size = c_instr.size
            # If an instruction uses extended args, those appear before the instruction
            # causing the instruction to appear at offset that accounts for extended
            # args. So we first update the offset to account for extended args, then
            # record the instruction offset and then add the instruction itself to the
            # offset.
            offset += (size // 2 - 1) if OFFSET_AS_INSTRUCTION else (size - 2)
            current_instr_offset = offset
            offset += 1 if OFFSET_AS_INSTRUCTION else 2

            # on Python 3.11+ remove CACHE opcodes if we are requested to do so.
            # We are careful to first advance the offset and check that the CACHE
            # is not a jump target. It should never be the case but we double check.
            if prune_caches and c_instr.name == "CACHE":
                assert jump_target is None

            # We may need to insert a TryEnd after a CACHE so we need to run the
            # through the last block.
            else:
                opcode = c_instr._opcode
                arg: InstrArg
                c_arg = c_instr.arg
                # FIXME: better error reporting
                if opcode in _opcode.hasconst:
                    arg = self.consts[c_arg]
                elif opcode in _opcode.haslocal:
                    if opcode in DUAL_ARG_OPCODES:
                        arg = (locals_lookup[c_arg >> 4], locals_lookup[c_arg & 15])
                    else:
                        arg = locals_lookup[c_arg]
                elif opcode in _opcode.hasname:
                    if opcode in BITFLAG_OPCODES:
                        arg = (
                            bool(c_arg & 1),
                            self.names[c_arg >> 1],
                        )
                    elif opcode in BITFLAG2_OPCODES:
                        arg = (bool(c_arg & 1), bool(c_arg & 2), self.names[c_arg >> 2])
                    else:
                        arg = self.names[c_arg]
                elif opcode in _opcode.hasfree:
                    if c_arg < ncells:
                        n_or_cell = cells_lookup[c_arg]
                        arg = (
                            n_or_cell
                            if isinstance(n_or_cell, CellVar)
                            else CellVar(n_or_cell)
                        )
                    else:
                        name = self.freevars[c_arg - ncells]
                        arg = FreeVar(name)
                elif opcode in _opcode.hascompare:
                    arg = Compare(
                        (c_arg >> 5) + ((1 << 4) if (c_arg & 16) else 0)
                        if PY313
                        else ((c_arg >> 4) if PY312 else c_arg)
                    )
                elif opcode in INTRINSIC_1OP:
                    arg = Intrinsic1Op(c_arg)
                elif opcode in INTRINSIC_2OP:
                    arg = Intrinsic2Op(c_arg)
                elif opcode in BINARY_OPS:
                    arg = BinaryOp(c_arg)
                elif opcode in COMMON_CONSTANT_OPS:
                    arg = CommonConstant(c_arg)
                elif opcode in SPECIAL_OPS:
                    arg = SpecialMethod(c_arg)
                elif opcode in FORMAT_VALUE_OPS:
                    if opcode in BITFLAG_OPCODES:
                        arg = (
                            bool(c_arg & 1),
                            FormatValue(c_arg >> 1),
                        )
                    else:
                        arg = FormatValue(c_arg)
                else:
                    arg = c_arg

                location = c_instr.location or InstrLocation(lineno, None, None, None)

                if jump_target is not None:
                    arg = PLACEHOLDER_LABEL
                    instr_index = len(instructions)
                    jumps.append((instr_index, jump_target))

                instructions.append(Instr(c_instr.name, arg, location=location))

            # We now insert the TryEnd entries
            if current_instr_offset in ex_end:
                entries = ex_end[current_instr_offset]
                for entry in reversed(entries):
                    try:
                        instructions.append(TryEnd(tb_instrs[entry]))
                    except KeyError:
                        # The end offset is behind the start offset, so we
                        # need to create
                        tb_instr = TryBegin(
                            Label(),
                            entry.push_lasti,
                            entry.stack_depth
                            if conserve_exception_block_stackdepth
                            else UNSET,
                        )
                        # Per entry store the pseudo instruction associated
                        tb_instrs[entry] = tb_instr
                        instructions.append(tb_instr)
                        instructions.append(TryEnd(tb_instr))

        # Replace jump targets with labels
        for index, jump_target in jumps:
            instr = instructions[index]
            assert isinstance(instr, Instr) and instr.arg is PLACEHOLDER_LABEL
            # FIXME: better error reporting on missing label
            instr.arg = labels[jump_target]

        # Set the label for TryBegin
        for entry, tb in tb_instrs.items():
            tb.target = labels[entry.target]

        bytecode = _bytecode.Bytecode()
        bytecode._copy_attr_from(self)

        nargs = bytecode.argcount + bytecode.kwonlyargcount
        nargs += bytecode.posonlyargcount
        if bytecode.flags & inspect.CO_VARARGS:
            nargs += 1
        if bytecode.flags & inspect.CO_VARKEYWORDS:
            nargs += 1
        bytecode.argnames = self.varnames[:nargs]
        _set_docstring(bytecode, self.consts)

        bytecode.extend(instructions)
        return bytecode


class _ConvertBytecodeToConcrete:
    # FIXME document attributes

    #: Default number of passes of compute_jumps() before giving up.  Refer to
    #: assemble_jump_offsets() in compile.c for background.
    _compute_jumps_passes = 10

    def __init__(self, code: _bytecode.Bytecode) -> None:
        assert isinstance(code, _bytecode.Bytecode)
        self.bytecode = code

        # temporary variables
        self.instructions: List[ConcreteInstr] = []
        self.jumps: List[Tuple[int, Label, ConcreteInstr]] = []
        self.labels: Dict[Label, int] = {}
        self.exception_handling_blocks: Dict[TryBegin, ExceptionTableEntry] = {}
        self.required_caches = 0
        self.seen_manual_cache = False

        # used to build ConcreteBytecode() object
        self.consts_indices: Dict[Union[bytes, Tuple[type, int]], int] = {}
        self.consts_list: List[Any] = []
        self.names: List[str] = []
        self.varnames: List[str] = []

    def add_const(self, value: Any) -> int:
        key = const_key(value)
        if key in self.consts_indices:
            return self.consts_indices[key]
        index = len(self.consts_indices)
        self.consts_indices[key] = index
        self.consts_list.append(value)
        return index

    @staticmethod
    def add(names: List[str], name: str) -> int:
        try:
            index = names.index(name)
        except ValueError:
            index = len(names)
            names.append(name)
        return index

    def concrete_instructions(self) -> None:
        location = InstrLocation(self.bytecode.first_lineno, None, None, None)
        # Track instruction (index) using cell vars and free vars to be able to update
        # the index used once all the names are known.
        cell_instrs: List[int] = []
        free_instrs: List[int] = []

        # On 3.13+, try to use small indexes for names used in dual arg opcode
        # to improve the chances to be able to use them (since we cannot use
        # only the 15 first names.
        if PY313:
            for binstr in self.bytecode:
                if isinstance(binstr, Instr) and binstr._opcode in DUAL_ARG_OPCODES:
                    assert isinstance(binstr.arg, tuple)
                    for parg in binstr.arg:
                        assert isinstance(parg, str)
                        self.add(self.varnames, parg)

        # We use None as a sentinel to ensure caches for the last instruction are
        # properly generated.
        for instr in itertools.chain(self.bytecode, [None]):
            # Enforce proper use of CACHE opcode on Python 3.11+ by checking we get the
            # number we expect or directly generate the needed ones.
            if isinstance(instr, Instr) and instr.name == "CACHE":
                if not self.required_caches:
                    raise RuntimeError("Found a CACHE opcode when none was expected.")
                self.seen_manual_cache = True
                self.required_caches -= 1

            elif self.required_caches:
                if not self.seen_manual_cache:
                    # We preserve the location of the instruction requiring the
                    # presence of cache instructions
                    self.instructions.extend(
                        [
                            ConcreteInstr(
                                "CACHE", 0, location=self.instructions[-1].location
                            )
                            for i in range(self.required_caches)
                        ]
                    )
                    self.required_caches = 0
                    self.seen_manual_cache = False
                else:
                    raise RuntimeError(
                        "Found some manual CACHE opcode but less than expected. "
                        f"Missing {self.required_caches} CACHE opcodes."
                    )

            if instr is None:
                continue

            if isinstance(instr, Label):
                self.labels[instr] = len(self.instructions)
                continue

            if isinstance(instr, SetLineno):
                location = InstrLocation(instr.lineno, None, None, None)
                continue

            if isinstance(instr, TryBegin):
                # We expect the stack depth to have be provided or computed earlier
                assert instr.stack_depth is not UNSET
                # NOTE here we store the index of the instruction at which the
                # exception table entry starts. This is not the final value we want,
                # we want the offset in the bytecode but that requires to compute
                # the jumps first to resolve any possible extended arg needed in a
                # jump.
                self.exception_handling_blocks[instr] = ExceptionTableEntry(
                    len(self.instructions), 0, 0, instr.stack_depth, instr.push_lasti
                )
                continue

            # Do not handle TryEnd before we insert possible CACHE opcode
            if isinstance(instr, TryEnd):
                entry = self.exception_handling_blocks[instr.entry]
                # The TryEnd is located after the last opcode in the exception entry
                # so we move the offset by one. We choose one so that the end does
                # encompass a possible EXTENDED_ARG
                entry.stop_offset = len(self.instructions) - 1
                continue

            assert isinstance(instr, Instr)

            if instr.location is not UNSET and instr.location is not None:
                location = instr.location

            instr_name = instr.name
            opcode = instr._opcode
            arg = instr.arg
            is_jump = False
            if isinstance(arg, Label):
                label = arg
                # fake value, real value is set in compute_jumps()
                c_arg = 0
                is_jump = True
            elif opcode in _opcode.hasconst:
                c_arg = self.add_const(arg)
            elif opcode in _opcode.haslocal:
                if opcode in DUAL_ARG_OPCODES:
                    assert (
                        isinstance(arg, tuple)
                        and len(arg) == 2
                        and isinstance(arg[0], str)
                        and isinstance(arg[1], str)
                    )
                    arg1_index = self.add(self.varnames, arg[0])
                    arg2_index = self.add(self.varnames, arg[1])
                    if arg1_index > 16 or arg2_index > 16:
                        n1, n2 = DUAL_ARG_OPCODES_SINGLE_OPS[opcode]
                        c_instr = ConcreteInstr(n1, arg1_index, location=location)
                        self.instructions.append(c_instr)
                        instr_name = n2
                        c_arg = arg2_index
                    else:
                        c_arg = (arg1_index << 4) + arg2_index
                elif PY313 and isinstance(arg, CellVar):
                    cell_instrs.append(len(self.instructions))
                    c_arg = self.bytecode.cellvars.index(arg.name)
                elif PY313 and isinstance(arg, FreeVar):
                    free_instrs.append(len(self.instructions))
                    c_arg = self.bytecode.freevars.index(arg.name)
                else:
                    assert isinstance(arg, str)
                    c_arg = self.add(self.varnames, arg)
            elif opcode in _opcode.hasname:
                if opcode in BITFLAG_OPCODES:
                    assert (
                        isinstance(arg, tuple)
                        and len(arg) == 2
                        and isinstance(arg[0], bool)
                    ), arg
                    if isinstance(arg[1], str):
                        index = self.add(self.names, arg[1])
                    elif isinstance(arg, FormatValue):
                        index = int(arg)
                    else:
                        assert False, arg  # noqa
                    c_arg = int(arg[0]) + (index << 1)
                elif opcode in BITFLAG2_OPCODES:
                    assert (
                        isinstance(arg, tuple)
                        and len(arg) == 3
                        and isinstance(arg[0], bool)
                        and isinstance(arg[1], bool)
                        and isinstance(arg[2], str)
                    ), arg
                    index = self.add(self.names, arg[2])
                    c_arg = int(arg[0]) + 2 * int(arg[1]) + (index << 2)
                else:
                    assert isinstance(arg, str), f"Got {arg}, expected a str"
                    c_arg = self.add(self.names, arg)
            elif opcode in _opcode.hasfree:
                if isinstance(arg, CellVar):
                    cell_instrs.append(len(self.instructions))
                    c_arg = self.bytecode.cellvars.index(arg.name)
                else:
                    assert isinstance(arg, FreeVar)
                    free_instrs.append(len(self.instructions))
                    c_arg = self.bytecode.freevars.index(arg.name)
            elif opcode in _opcode.hascompare:
                if isinstance(arg, Compare):
                    # In Python 3.13 the 4 lowest bits are used for caching
                    # and the 5th one indicate a cast to bool
                    if PY313:
                        c_arg = (
                            arg._get_mask()
                            + ((arg.value & 0b1111) << 5)
                            + (arg.value & 16)
                        )
                    # In Python 3.12 the 4 lowest bits are used for caching
                    # See compare_masks in compile.c
                    elif PY312:
                        c_arg = arg._get_mask() + (arg.value << 4)
                    else:
                        c_arg = arg.value
            elif opcode in INTRINSIC:
                if isinstance(arg, (Intrinsic1Op, Intrinsic2Op)):
                    c_arg = arg.value
            else:
                assert isinstance(arg, int)
                c_arg = arg

            # The above should have performed all the necessary conversion
            c_instr = ConcreteInstr(instr_name, c_arg, location=location)
            if is_jump:
                self.jumps.append((len(self.instructions), label, c_instr))

            # If the instruction expect some cache
            if PY311:
                self.required_caches = c_instr.use_cache_opcodes()
                self.seen_manual_cache = False

            self.instructions.append(c_instr)

        # On Python 3.11 varnames and cells can share some names. Wind the shared
        # names and update the arg argument of instructions using cell vars.
        # We also track by how much to offset free vars which are stored in a
        # contiguous array after the cell vars
        if PY311:
            # Map naive cell index to shared index
            shared_name_indexes: Dict[int, int] = {}
            n_shared = 0
            n_unshared = 0
            for i, name in enumerate(self.bytecode.cellvars):
                if name in self.varnames:
                    shared_name_indexes[i] = self.varnames.index(name)
                    n_shared += 1
                else:
                    shared_name_indexes[i] = len(self.varnames) + n_unshared
                    n_unshared += 1

            for index in cell_instrs:
                c_instr = self.instructions[index]
                c_instr.arg = shared_name_indexes[c_instr.arg]

            free_offset = len(self.varnames) + len(self.bytecode.cellvars) - n_shared
        else:
            free_offset = len(self.bytecode.cellvars)

        for index in free_instrs:
            c_instr = self.instructions[index]
            c_instr.arg += free_offset

    def compute_jumps(self) -> bool:
        # For labels we need the offset before the instruction at a given index but for
        # exception table entries we need the offset of the instruction which can differ
        # in the presence of extended args...
        label_offsets = []
        instruction_offsets = []
        offset = 0
        for instr in self.instructions:
            label_offsets.append(offset)
            # If an instruction uses extended args, those appear before the instruction
            # causing the instruction to appear at offset that accounts for extended
            # args.
            offset += (
                (instr.size // 2 - 1) if OFFSET_AS_INSTRUCTION else (instr.size - 2)
            )
            instruction_offsets.append(offset)
            offset += 1 if OFFSET_AS_INSTRUCTION else 2
        # needed if a label is at the end
        label_offsets.append(offset)

        # Fix argument of jump instructions: resolve labels
        modified = False
        for index, label, instr in self.jumps:
            target_index = self.labels[label]
            target_offset = label_offsets[target_index]

            # For jump using cache opcodes, an argument of 0 jumps to the
            # first non cache instructions right after the jump instruction
            instr_offset = label_offsets[index] + instr.use_cache_opcodes()
            if instr.is_forward_rel_jump():
                target_offset -= instr_offset + (
                    instr.size // 2 if OFFSET_AS_INSTRUCTION else instr.size
                )
            elif instr.is_backward_rel_jump():
                target_offset = (
                    instr_offset
                    + (instr.size // 2 if OFFSET_AS_INSTRUCTION else instr.size)
                    - target_offset
                )

            old_size = instr.size
            # FIXME: better error report if target_offset is negative
            instr.arg = target_offset
            if instr.size != old_size:
                modified = True

        # If a jump required an extended arg hence invalidating the calculation
        # we return early before filling the exception table entries
        if modified:
            return modified

        # Resolve labels for exception handling entries
        for tb, entry in self.exception_handling_blocks.items():
            # Set the offset for the start and end offset from the instruction
            # index stored when assembling the concrete instructions.
            entry.start_offset = instruction_offsets[entry.start_offset]
            entry.stop_offset = instruction_offsets[entry.stop_offset]

            # Set the offset to the target instruction
            lb = tb.target
            assert isinstance(lb, Label)
            target_index = self.labels[lb]
            target_offset = label_offsets[target_index]
            entry.target = target_offset

        return False

    def to_concrete_bytecode(
        self,
        compute_jumps_passes: Optional[int] = None,
        compute_exception_stack_depths: bool = True,
    ) -> ConcreteBytecode:
        if PY311 and compute_exception_stack_depths:
            cfg = _bytecode.ControlFlowGraph.from_bytecode(self.bytecode)
            cfg.compute_stacksize(compute_exception_stack_depths=True)
            self.bytecode = cfg.to_bytecode()

        if compute_jumps_passes is None:
            compute_jumps_passes = self._compute_jumps_passes

        first_const = self.bytecode.docstring
        if first_const is not UNSET:
            self.add_const(first_const)

        self.varnames.extend(self.bytecode.argnames)

        self.concrete_instructions()
        for _ in range(0, compute_jumps_passes):
            modified = self.compute_jumps()
            if not modified:
                break
        else:
            raise RuntimeError(
                "compute_jumps() failed to converge after"
                " %d passes" % (compute_jumps_passes)
            )

        concrete = ConcreteBytecode(
            self.instructions,
            consts=tuple(self.consts_list),
            names=tuple(self.names),
            varnames=self.varnames,
            exception_table=list(self.exception_handling_blocks.values()),
        )
        concrete._copy_attr_from(self.bytecode)
        return concrete
