"""How the line items of one finished batch are laid out as a single object in a storage bucket."""

import json
from datetime import datetime, timezone
from typing import Final

from litellm.types.utils import StandardBatchLineItemsPayload


def batch_line_items_ndjson(payload: StandardBatchLineItemsPayload) -> str:
    return "".join(f"{_ndjson_line(line)}\n" for line in payload["lines"])


def _ndjson_line(line: object) -> str:
    """One line as JSON, with non-ASCII text kept readable. A line holding a lone surrogate, which a provider's
    JSON escapes can carry, has no UTF-8 form, so that line alone is written with every non-ASCII character escaped."""
    text: Final = json.dumps(line, default=str, ensure_ascii=False)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return json.dumps(line, default=str)
    return text


def batch_line_items_object_name(payload: StandardBatchLineItemsPayload) -> str:
    """batch_line_items/<finished day, UTC>/<batch id>/part-<part>.jsonl, the same name on every retry so a rewrite
    replaces it."""
    day: Final = datetime.fromtimestamp(payload["finished_at"], tz=timezone.utc).strftime("%Y-%m-%d")
    return f"batch_line_items/{day}/{payload['batch_id'].replace('/', '_')}/part-{payload['part']:05d}.jsonl"
