"""Alice WonderFence guardrail exception types."""


class WonderFenceMissingSecrets(Exception):
    """Raised when Alice API key cannot be resolved from any source."""


class WonderFenceBlockedError(Exception):
    """Raised when WonderFence blocks a request/response."""

    def __init__(self, detail: dict):
        self.detail = detail
        super().__init__(detail.get("error", "Blocked by Alice WonderFence guardrail"))


class WonderFenceScanBudgetExceeded(Exception):
    """Raised when a request/response exceeds the configured total-work cap.

    A fail-closed configuration/abuse guard, never a transport failure: it is
    mapped to HTTP 400 before any WonderFence call and is not subject to
    ``fail_open`` (a caller must not be able to bypass scanning by overflowing
    the cap).
    """

    def __init__(self, detail: dict):
        self.detail = detail
        super().__init__(detail.get("error", "Alice WonderFence scan budget exceeded"))
