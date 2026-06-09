"""Pure verdict combination. Worst-wins, no I/O, trivially testable."""

from typing import Sequence

from litellm.integrations.governor.model.decisions import Decision, Status, Verdict


def combine(
    verdicts: Sequence[Verdict], *, request_id: str, latency_ms: float
) -> Decision:
    status: Status = "admitted"
    if any(v.status == "rejected" for v in verdicts):
        status = "rejected"
    elif any(v.status == "admitted_degraded" for v in verdicts):
        status = "admitted_degraded"
    return Decision(
        status=status,
        verdicts=tuple(verdicts),
        request_id=request_id,
        latency_ms=latency_ms,
    )
