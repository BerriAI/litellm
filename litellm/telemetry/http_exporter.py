import json
from typing import Final

import httpx

from litellm.telemetry.report import Report, report_to_json
from litellm.telemetry.sink import ExportOutcome

_RETRYABLE_STATUS: Final = frozenset({408, 425, 429})


def _outcome_for(status_code: int) -> ExportOutcome:
    if 200 <= status_code < 300:
        return ExportOutcome.SENT
    if status_code in _RETRYABLE_STATUS or status_code >= 500:
        return ExportOutcome.RETRY
    return ExportOutcome.REJECTED


class HttpExporter:
    """POSTs each report as JSON; failures come back as an ``ExportOutcome``, never as an exception"""

    def __init__(self, client: httpx.AsyncClient, endpoint: str) -> None:
        self._client: Final = client
        self._endpoint: Final = endpoint

    async def export(self, report: Report) -> ExportOutcome:
        body: Final = json.dumps(report_to_json(report), separators=(",", ":")).encode()
        try:
            response: Final = await self._client.post(
                self._endpoint, content=body, headers={"content-type": "application/json"}
            )
        except httpx.HTTPError:
            return ExportOutcome.RETRY
        return _outcome_for(response.status_code)
