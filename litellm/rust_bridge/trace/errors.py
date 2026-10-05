from typing import Final, Literal


class TraceChanged(Exception):
    """The paging snapshot no longer matches the stored trace, so the client must start a new traversal.

    Raised by the Rust trace reader when a cursor's snapshot version differs from the graph it rebuilt.
    """


class TraceQueryError(Exception):
    def __init__(
        self,
        kind: Literal["rejected", "limited", "unavailable"],
        database_code: int | None,
        message: str,
    ) -> None:
        self.kind: Final = kind
        self.database_code: Final = database_code
        self.message: Final = message
        super().__init__(message)
