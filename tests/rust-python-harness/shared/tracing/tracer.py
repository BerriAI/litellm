"""Execution tracer for validating Python-to-Rust mapping."""
import sys
from dataclasses import dataclass, field
from typing import Set, List, Dict
from tabulate import tabulate


@dataclass
class CallRecord:
    module: str
    function: str
    depth: int


@dataclass
class ExecutionTrace:
    endpoint: str
    calls: List[CallRecord] = field(default_factory=list)
    call_counts: Dict[str, int] = field(default_factory=dict)


class ExecutionTracer:
    """Traces Python function calls to validate Rust mapping."""

    def __init__(self, target_modules: List[str]):
        self.target_modules = target_modules
        self.trace = ExecutionTrace(endpoint="unknown")
        self.depth = 0

    def _should_trace(self, frame) -> bool:
        module = frame.f_globals.get('__name__', '')
        return any(module.startswith(m) for m in self.target_modules)

    def trace_calls(self, frame, event, arg):
        if event == 'call' and self._should_trace(frame):
            module = frame.f_globals.get('__name__', '')
            func = frame.f_code.co_name
            key = f"{module}.{func}"

            record = CallRecord(module=module, function=func, depth=self.depth)
            self.trace.calls.append(record)
            self.trace.call_counts[key] = self.trace.call_counts.get(key, 0) + 1
            self.depth += 1
        elif event == 'return':
            self.depth = max(0, self.depth - 1)

        return self.trace_calls

    def start(self):
        sys.settrace(self.trace_calls)

    def stop(self):
        sys.settrace(None)
        return self.trace


def print_trace_table(trace: ExecutionTrace, rust_functions: Set[str]):
    """Print mapping validation results."""
    python_calls = {f"{c.module}.{c.function}" for c in trace.calls}
    missing = python_calls - rust_functions
    implemented = python_calls & rust_functions

    print(f"\n🔍 Mapping Validation: {trace.endpoint}")
    print(f"   Total: {len(python_calls)} | Rust: {len(implemented)} | Missing: {len(missing)} | Coverage: {len(implemented)/len(python_calls)*100:.1f}%\n")

    table_data = []
    for call in trace.calls[:20]:
        func_key = f"{call.module}.{call.function}"
        status = "✅" if func_key in rust_functions else "❌"
        table_data.append([status, call.function[:40], call.module[:50], trace.call_counts[func_key], call.depth])

    print(tabulate(table_data, headers=["Status", "Function", "Module", "Calls", "Depth"], tablefmt="grid"))

    if missing:
        print(f"\n❌ Missing Rust implementations (Top 10):\n")
        missing_sorted = sorted(missing, key=lambda f: trace.call_counts.get(f, 0), reverse=True)
        for func in missing_sorted[:10]:
            print(f"   {func} ({trace.call_counts[func]} calls)")
