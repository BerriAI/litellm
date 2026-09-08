from __future__ import annotations

from datetime import datetime


def initialize_logging(arguments: dict[str, object], asynchronous: bool, route: str) -> object:
    from litellm.rust_bridge.ocr import initialize_logging as initialize_ocr_logging

    return initialize_ocr_logging(arguments, asynchronous, route)


def invoke_terminal(
    action: str,
    roots: object,
    logger: object,
    value: object,
    start_time: datetime,
    end_time: datetime,
) -> object:
    from litellm.rust_bridge.ocr import invoke_terminal as invoke_ocr_terminal

    return invoke_ocr_terminal(action, roots, logger, value, start_time, end_time)
