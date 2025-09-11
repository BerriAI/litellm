def _calculate_byte_size(data):
    if isinstance(data, str):
        # We encode here to handle non-ascii characters
        # If there are non-unicode characters, we replace
        # with a single character/byte
        return len(data.encode("utf-8", errors="replace"))

    if isinstance(data, bytes):
        return len(data)

    if isinstance(data, dict):
        total = 0
        for k, v in data.items():
            total += _calculate_byte_size(k)
            total += _calculate_byte_size(v)
        return total

    return 0  # Return 0 to avoid breaking calculations if its a type we don't know
