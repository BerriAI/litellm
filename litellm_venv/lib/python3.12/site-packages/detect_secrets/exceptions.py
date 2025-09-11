class UnableToReadBaselineError(ValueError):
    """Think of this as a 404, if getting a baseline had a HTTPError code."""

    pass


class InvalidBaselineError(ValueError):
    """Think of this as a 400, if getting a baseline had a HTTPError code."""

    pass


class InvalidFile(ValueError):
    """Think of this as a 400, if FileNotFoundError was a 404 HTTPError code."""

    pass


class SecretNotFoundOnSpecifiedLineError(Exception):
    def __init__(self, line: int) -> None:
        super().__init__(
            "ERROR: Secret not found on line {}!\n".format(line)
            + "Try recreating your baseline to fix this issue.",
        )


class NoLineNumberError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "ERROR: No line numbers found in baseline! Line numbers are needed "
            "for auditing secret occurrences. Try recreating your baseline to fix "
            "this issue.",
        )
