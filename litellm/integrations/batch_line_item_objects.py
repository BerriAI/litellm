"""How the line items of one finished batch are laid out as a single object in a storage bucket."""

import json
from datetime import datetime, timezone
from typing import Final

from litellm.types.utils import StandardBatchLineItemsPayload


def batch_line_items_ndjson(payload: StandardBatchLineItemsPayload) -> str:
    return "\n".join(json.dumps(line, default=str, ensure_ascii=False) for line in payload["lines"])


def batch_line_items_object_name(payload: StandardBatchLineItemsPayload) -> str:
    """batch_line_items/<finished day, UTC>/<batch id>.jsonl, the same name on every retry so a rewrite replaces it."""
    day: Final = datetime.fromtimestamp(payload["finished_at"], tz=timezone.utc).strftime("%Y-%m-%d")
    return f"batch_line_items/{day}/{payload['batch_id'].replace('/', '_')}.jsonl"
