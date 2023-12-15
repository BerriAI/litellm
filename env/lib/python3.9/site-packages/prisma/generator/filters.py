def quote(string: str) -> str:
    """Surround the given string with single quotes. e.g.

    foo -> 'foo'

    This does not do any form of escaping, the input is expected to not contain any single quotes.
    """
    return "'" + string + "'"
