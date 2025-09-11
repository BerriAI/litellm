# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from urllib.parse import ParseResult


def _parse_url(account_url: str) -> "ParseResult":
    try:
        if not account_url.lower().startswith("http"):
            account_url = "https://" + account_url
    except AttributeError as exc:
        raise ValueError("Account URL must be a string.") from exc
    parsed_url = urlparse(account_url.rstrip("/"))
    if not parsed_url.netloc:
        raise ValueError(f"Invalid URL: {account_url}")
    return parsed_url


def _format_url(scheme: str, hostname: str, query_str: str) -> str:
    return f"{scheme}://{hostname}/{query_str}"
