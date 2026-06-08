"""End-user extraction.

Thin wrapper over the existing six-check chain in
``litellm.proxy.auth.auth_utils.get_end_user_id_from_request_body``.
Validation against the DB stays in ``resolve_and_validate_end_user_id``
and runs from the legacy auth path; this extractor returns the raw
identifier only.
"""

from typing import Optional


def extract_end_user_id(
    body: Optional[dict],
    headers: Optional[dict] = None,
) -> Optional[str]:
    if body is None:
        body = {}

    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    return get_end_user_id_from_request_body(
        request_body=body, request_headers=headers
    )
