"""
OpenAI Batches API transformation utilities.

Owns helpers that encode the OpenAI Batches contract so that
``litellm.create_batch`` (and any Bedrock-side response materialization
that pretends to be OpenAI-shaped) can share the same logic.
"""

from typing import Any, Dict, Optional

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


def sanitize_openai_batch_metadata(
    metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    """
    OpenAI's Batches API (and OpenAI-compatible providers like Azure) require
    metadata values to be strings - ``Dict[str, str]``. The proxy pre-call
    pipeline can inject non-string values (e.g. ``applied_policies`` as a
    ``List[str]``), which causes upstream 400 errors:

        Invalid type for 'metadata.<field>': expected a string, but got an
        array instead. (code=invalid_type)

    This helper normalizes a metadata dict so every value is a string. Non-
    string scalars are JSON-stringified via ``safe_dumps``. Internal-only keys
    such as ``standard_logging_guardrail_information`` and ``None`` values are
    dropped (they are not part of the OpenAI contract).

    ``None`` and non-dict inputs return ``None`` so callers can omit the
    metadata field entirely.
    """
    if metadata is None or not isinstance(metadata, dict):
        return None

    sanitized: Dict[str, str] = {}
    for key, value in metadata.items():
        if key == "standard_logging_guardrail_information" or value is None:
            continue
        str_key = str(key)
        if isinstance(value, str):
            sanitized[str_key] = value
        else:
            sanitized[str_key] = safe_dumps(value)
    return sanitized
