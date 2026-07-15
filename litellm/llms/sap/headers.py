from typing import Dict, Mapping, Optional

_SAP_RESERVED_HEADERS = frozenset(
    {
        "authorization",
        "ai-resource-group",
        "content-type",
        "ai-client-type",
    }
)


def merge_sap_request_headers(
    provider_headers: Mapping[str, str],
    caller_headers: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    if not caller_headers:
        return dict(provider_headers)

    safe_caller_headers = {
        key: value for key, value in caller_headers.items() if key.lower() not in _SAP_RESERVED_HEADERS
    }
    return {**safe_caller_headers, **provider_headers}
