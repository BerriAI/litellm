from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter


def embedding_metadata(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    metadata: Final = TypeAdapter[dict[str, object] | None](dict[str, object] | None).validate_python(
        kwargs.get("metadata")
    )
    if metadata is None:
        return None
    if kwargs.get("proxy_server_request") is not None:
        requester: Final = TypeAdapter[dict[str, object] | None](dict[str, object] | None).validate_python(
            metadata.get("requester_metadata")
        )
        if requester is None:
            return None
        return {key: value for key, value in requester.items() if key not in ("headers", "used_client_oauth_token")}
    return metadata
