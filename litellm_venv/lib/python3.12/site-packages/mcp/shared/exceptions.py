from mcp.types import ErrorData


class McpError(Exception):
    """
    Exception type raised when an error arrives over an MCP connection.
    """

    error: ErrorData

    def __init__(self, error: ErrorData):
        """Initialize McpError."""
        super().__init__(error.message)
        self.error = error
