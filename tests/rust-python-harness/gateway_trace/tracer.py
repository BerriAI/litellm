"""Execution tracer for gateway endpoints."""
import sys
from dataclasses import dataclass, field
from typing import Set, List, Dict
from tabulate import tabulate


@dataclass
class CallRecord:
    """Record of a single function call."""
    module: str
    function: str
    depth: int
    args: List[str]


@dataclass
class ExecutionTrace:
    """Complete execution trace."""
    endpoint: str
    calls: List[CallRecord] = field(default_factory=list)
    call_counts: Dict[str, int] = field(default_factory=dict)


class ExecutionTracer:
    """Traces function calls during execution."""

    def __init__(self, target_modules: List[str]):
        """
        Initialize tracer.

        Args:
            target_modules: List of module prefixes to trace (e.g., ['litellm.proxy'])
        """
        self.target_modules = target_modules
        self.trace = ExecutionTrace(endpoint="unknown")
        self.depth = 0

    def _should_trace(self, frame) -> bool:
        """Check if we should trace this frame."""
        module = frame.f_globals.get('__name__', '')
        return any(module.startswith(m) for m in self.target_modules)

    def trace_calls(self, frame, event, arg):
        """Trace function calls."""
        if event == 'call' and self._should_trace(frame):
            module = frame.f_globals.get('__name__', '')
            func = frame.f_code.co_name
            key = f"{module}.{func}"

            # Record call
            record = CallRecord(
                module=module,
                function=func,
                depth=self.depth,
                args=list(frame.f_locals.keys())[:5]
            )
            self.trace.calls.append(record)
            self.trace.call_counts[key] = self.trace.call_counts.get(key, 0) + 1
            self.depth += 1

        elif event == 'return':
            self.depth = max(0, self.depth - 1)

        return self.trace_calls

    def start(self):
        """Start tracing."""
        sys.settrace(self.trace_calls)

    def stop(self):
        """Stop tracing and return trace."""
        sys.settrace(None)
        return self.trace


def print_trace_table(trace: ExecutionTrace, rust_functions: Set[str], max_rows: int = 20):
    """
    Print formatted trace table.

    Args:
        trace: Execution trace
        rust_functions: Set of implemented Rust functions
        max_rows: Maximum rows to display
    """
    # Summary metrics
    python_calls = {f"{c.module}.{c.function}" for c in trace.calls}
    missing = python_calls - rust_functions
    implemented = python_calls & rust_functions

    print(f"\n🔍 Gateway Trace: {trace.endpoint}")
    print(f"   Total functions called: {len(python_calls)}")
    print(f"   Rust implementations: {len(implemented)}")
    print(f"   Missing in Rust: {len(missing)}")
    print(f"   Coverage: {len(implemented)/len(python_calls)*100:.1f}%\n")

    # Execution trace table
    print("📋 Execution Trace (Top {}):\n".format(min(max_rows, len(trace.calls))))
    table_data = []
    for call in trace.calls[:max_rows]:
        func_key = f"{call.module}.{call.function}"
        status = "✅" if func_key in rust_functions else "❌"
        table_data.append([
            status,
            call.function[:40],  # Truncate long names
            call.module[:50],
            trace.call_counts[func_key],
            call.depth
        ])

    print(tabulate(
        table_data,
        headers=["Status", "Function", "Module", "Calls", "Depth"],
        tablefmt="grid"
    ))

    # Missing functions table
    if missing:
        print(f"\n❌ Missing Functions (Top 10 by call count):\n")
        missing_sorted = sorted(missing, key=lambda f: trace.call_counts.get(f, 0), reverse=True)
        missing_data = []
        for func in missing_sorted[:10]:
            count = trace.call_counts[func]
            priority = "🔥 High" if count > 5 else "⚠️ Medium"
            category = _categorize_function(func)
            func_name = func.split('.')[-1][:40]
            missing_data.append([priority, func_name, count, category])

        print(tabulate(
            missing_data,
            headers=["Priority", "Function", "Calls", "Category"],
            tablefmt="grid"
        ))


def _categorize_function(func: str) -> str:
    """Categorize function by type."""
    if "proxy_server" in func:
        return "Gateway Handler"
    elif "ProxyBaseLLMRequestProcessing" in func:
        return "Request Processor"
    elif "process_llm_request" in func:
        return "Core Processing"
    elif "headers" in func:
        return "Response Headers"
    elif "request_body" in func or "parse" in func:
        return "Request Parsing"
    elif "logging" in func or "hook" in func:
        return "Logging/Hooks"
    else:
        return "Other"
