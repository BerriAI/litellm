from __future__ import annotations

from typing import Final

from tests.sdk_function_trace.proxy_runtime import run_python_ocr_proxy_trace
from tests.sdk_function_trace.proxy_steps import (
    python_ocr_proxy_issues,
    python_ocr_proxy_steps,
)
from tests.sdk_function_trace.table import format_trace_table


def render_ocr_proxy_trace(*, colorize: bool) -> tuple[str, bool]:
    events: Final = run_python_ocr_proxy_trace()
    python_steps: Final = python_ocr_proxy_steps(events)
    issues: Final = python_ocr_proxy_issues(python_steps)
    output: Final = "\n".join(
        (
            "route: /ocr    surface: proxy HTTP    mode: async",
            "",
            format_trace_table(python_steps, (), colorize=colorize),
            "",
            "rust proxy: MISSING — ai-gateway has no Axum /ocr route",
            f"python proxy pipeline: {'FAIL: ' + '; '.join(issues) if issues else 'PASS'}",
            "proxy parity: FAIL",
            "",
        )
    )
    return output, False
