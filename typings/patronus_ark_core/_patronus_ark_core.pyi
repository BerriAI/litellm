# SPDX-License-Identifier: MIT OR Apache-2.0
# Type-only SDK interface from patronus-ark-core 0.1.14; runtime SDK is installed separately.
from .types import ScannerConfig, ScanReport, WarmupOptions, WolfExecutionOptions

__version__: str

class CancelledError(RuntimeError): ...
class ScannerBusyError(RuntimeError): ...

class ScanControl:
    def __init__(self, timeout_ms: int | None = None) -> None: ...
    def cancel(self) -> None: ...

class Scanner:
    def __init__(
        self,
        config: dict[str, object] | ScannerConfig | None = None,
        *,
        warmup: dict[str, object] | WarmupOptions | None = None,
        wolf_options: dict[str, int] | WolfExecutionOptions | None = None,
    ) -> None: ...
    def scan(
        self,
        text: str,
        *,
        pii: bool = True,
        dlp: bool = True,
        injection: bool = True,
        control: ScanControl | None = None,
    ) -> ScanReport: ...
