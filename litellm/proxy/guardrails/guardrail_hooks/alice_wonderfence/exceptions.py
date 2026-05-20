"""Alice WonderFence guardrail exception types."""


class WonderFenceMissingSecrets(Exception):
    """Raised when Alice API key cannot be resolved from any source."""


class WonderFenceBlockedError(Exception):
    """Raised when WonderFence blocks a request/response."""

    def __init__(self, detail: dict):
        self.detail = detail
        super().__init__(detail.get("error", "Blocked by Alice WonderFence guardrail"))
