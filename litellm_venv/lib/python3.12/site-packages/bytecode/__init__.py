__all__ = [
    "BinaryOp",
    "Bytecode",
    "Compare",
    "CompilerFlags",
    "ConcreteBytecode",
    "ConcreteInstr",
    "ControlFlowGraph",
    "Instr",
    "Label",
    "SetLineno",
    "__version__",
]

from io import StringIO
from typing import List, Union

# import needed to use it in bytecode.py
from bytecode.bytecode import (
    BaseBytecode,
    Bytecode,
    _BaseBytecodeList,
    _InstrList,
)

# import needed to use it in bytecode.py
from bytecode.cfg import BasicBlock, ControlFlowGraph

# import needed to use it in bytecode.py
from bytecode.concrete import (
    ConcreteBytecode,
    ConcreteInstr,
    _ConvertBytecodeToConcrete,
)
from bytecode.flags import CompilerFlags

# import needed to use it in bytecode.py
from bytecode.instr import (
    UNSET,
    BinaryOp,
    CellVar,
    Compare,
    FreeVar,
    Instr,
    Intrinsic1Op,
    Intrinsic2Op,
    Label,
    SetLineno,
    TryBegin,
    TryEnd,
)
from bytecode.version import __version__


def format_bytecode(
    bytecode: Union[Bytecode, ConcreteBytecode, ControlFlowGraph],
    *,
    lineno: bool = False,
) -> str:
    try_begins: List[TryBegin] = []

    def format_line(index, line):
        nonlocal cur_lineno, prev_lineno
        if lineno:
            if cur_lineno != prev_lineno:
                line = "L.% 3s % 3s: %s" % (cur_lineno, index, line)
                prev_lineno = cur_lineno
            else:
                line = "      % 3s: %s" % (index, line)
        else:
            line = line
        return line

    def format_instr(instr, labels=None):
        text = instr.name
        arg = instr._arg
        if arg is not UNSET:
            if isinstance(arg, Label):
                try:
                    arg = "<%s>" % labels[arg]
                except KeyError:
                    arg = "<error: unknown label>"
            elif isinstance(arg, BasicBlock):
                try:
                    arg = "<%s>" % labels[id(arg)]
                except KeyError:
                    arg = "<error: unknown block>"
            else:
                arg = repr(arg)
            text = "%s %s" % (text, arg)
        return text

    def format_try_begin(instr: TryBegin, labels: dict) -> str:
        if isinstance(instr.target, Label):
            try:
                arg = "<%s>" % labels[instr.target]
            except KeyError:
                arg = "<error: unknown label>"
        else:
            try:
                arg = "<%s>" % labels[id(instr.target)]
            except KeyError:
                arg = "<error: unknown label>"
        line = "TryBegin %s -> %s [%s]" % (
            len(try_begins),
            arg,
            instr.stack_depth,
        ) + (" last_i" if instr.push_lasti else "")

        # Track the seen try begin
        try_begins.append(instr)

        return line

    def format_try_end(instr: TryEnd) -> str:
        i = try_begins.index(instr.entry) if instr.entry in try_begins else "<unknwon>"
        return "TryEnd (%s)" % i

    buffer = StringIO()

    indent = " " * 4

    cur_lineno = bytecode.first_lineno
    prev_lineno = None

    if isinstance(bytecode, ConcreteBytecode):
        offset = 0
        for c_instr in bytecode:
            fields = []
            if c_instr.lineno is not None:
                cur_lineno = c_instr.lineno
            if lineno:
                fields.append(format_instr(c_instr))
                line = "".join(fields)
                line = format_line(offset, line)
            else:
                fields.append("% 3s    %s" % (offset, format_instr(c_instr)))
                line = "".join(fields)
            buffer.write(line + "\n")

            if isinstance(c_instr, ConcreteInstr):
                offset += c_instr.size

        if bytecode.exception_table:
            buffer.write("\n")
            buffer.write("Exception table:\n")
            for entry in bytecode.exception_table:
                buffer.write(
                    f"{entry.start_offset} to {entry.stop_offset} -> "
                    f"{entry.target} [{entry.stack_depth}]"
                    + (" lasti" if entry.push_lasti else "")
                    + "\n"
                )

    elif isinstance(bytecode, Bytecode):
        labels: dict[Label, str] = {}
        for index, instr in enumerate(bytecode):
            if isinstance(instr, Label):
                labels[instr] = "label_instr%s" % index

        for index, instr in enumerate(bytecode):
            if isinstance(instr, Label):
                label = labels[instr]
                line = "%s:" % label
                if index != 0:
                    buffer.write("\n")
            elif isinstance(instr, TryBegin):
                line = indent + format_line(index, format_try_begin(instr, labels))
                indent += "  "
            elif isinstance(instr, TryEnd):
                indent = indent[:-2]
                line = indent + format_line(index, format_try_end(instr))
            else:
                if instr.lineno is not None:
                    cur_lineno = instr.lineno
                line = format_instr(instr, labels)
                line = indent + format_line(index, line)
            buffer.write(line + "\n")
        buffer.write("\n")

    elif isinstance(bytecode, ControlFlowGraph):
        cfg_labels = {}
        for block_index, block in enumerate(bytecode, 1):
            cfg_labels[id(block)] = "block%s" % block_index

        for block in bytecode:
            buffer.write("%s:\n" % cfg_labels[id(block)])
            seen_instr = False
            for index, instr in enumerate(block):
                if isinstance(instr, TryBegin):
                    line = indent + format_line(
                        index, format_try_begin(instr, cfg_labels)
                    )
                    indent += "  "
                elif isinstance(instr, TryEnd):
                    if seen_instr:
                        indent = indent[:-2]
                    line = indent + format_line(index, format_try_end(instr))
                else:
                    if isinstance(instr, Instr):
                        seen_instr = True
                    if instr.lineno is not None:
                        cur_lineno = instr.lineno
                    line = format_instr(instr, cfg_labels)
                    line = indent + format_line(index, line)
                buffer.write(line + "\n")
            if block.next_block is not None:
                buffer.write(indent + "-> %s\n" % cfg_labels[id(block.next_block)])
            buffer.write("\n")
    else:
        raise TypeError("unknown bytecode class")

    return buffer.getvalue()[:-1]


def dump_bytecode(
    bytecode: Union[Bytecode, ConcreteBytecode, ControlFlowGraph],
    *,
    lineno: bool = False,
) -> None:
    print(format_bytecode(bytecode, lineno=lineno))
