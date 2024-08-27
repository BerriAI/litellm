from litellm.types.utils import GenericStreamingChunk as GChunk


def generic_chunk_has_all_required_fields(chunk: dict) -> bool:
    """
    Checks if the provided chunk dictionary contains all required fields for GenericStreamingChunk.

    :param chunk: The dictionary to check.
    :return: True if all required fields are present, False otherwise.
    """
    _all_fields = GChunk.__annotations__

    decision = all(key in _all_fields for key in chunk)
    return decision
