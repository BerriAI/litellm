from collections.abc import Mapping


def iter_metadata_dicts(request_kwargs: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        metadata
        for metadata_key in ("litellm_metadata", "metadata")
        if isinstance(metadata := request_kwargs.get(metadata_key), dict)
    )


def get_session_id_from_request_kwargs(request_kwargs: Mapping[str, object]) -> str | None:
    for metadata in iter_metadata_dicts(request_kwargs):
        session_id = metadata.get("session_id")
        if session_id is not None:
            return str(session_id)
    return None


def get_user_api_key_hash_from_request_kwargs(request_kwargs: Mapping[str, object]) -> str | None:
    for metadata in iter_metadata_dicts(request_kwargs):
        user_key = metadata.get("user_api_key_hash")
        if user_key is not None:
            return str(user_key)
    return None
