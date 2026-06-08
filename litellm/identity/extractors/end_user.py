"""End-user id extraction.

Wraps ``auth_utils.get_end_user_id_from_request_body``; returns the raw
identifier only. DB validation stays in ``resolve_and_validate_end_user_id``.
"""

from typing import Optional


def extract_end_user_id(
    body: Optional[dict],
    headers: Optional[dict] = None,
) -> Optional[str]:
    if body is None:
        body = {}

    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    return get_end_user_id_from_request_body(request_body=body, request_headers=headers)
