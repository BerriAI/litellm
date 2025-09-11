class CIVisibilityError(Exception):
    pass


class CIVisibilityDataError(CIVisibilityError):
    """Raised when data is invalid or missing:

    Examples;
        - adding an item that already exists
        - trying to fetch an item that doesn't exist
        - etc
    """

    pass


class CIVisibilityProcessError(CIVisibilityError):
    """Raised when items are in an unexpected state

    Examples:
        - finishing an item that's already been finished
        - setting status for an item that's already had its status set
    """


class CIVisibilityAuthenticationException(Exception):
    pass
