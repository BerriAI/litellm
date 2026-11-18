class TraceChanged(Exception):
    """The paging snapshot no longer matches the stored trace, so the client must start a new traversal.

    Raised by the Rust trace reader when a cursor's snapshot version differs from the graph it rebuilt.
    """
