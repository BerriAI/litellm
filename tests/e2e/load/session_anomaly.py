from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import Result, Success
from models import CacheControl, RichMessage, TextBlock
from transport import Transport


class SessionMessagesRequest(BaseModel):
    model: str
    max_tokens: int = 128
    system: list[TextBlock]
    messages: list[RichMessage]


class SessionUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class SessionContentBlock(BaseModel):
    type: str | None = None
    text: str | None = None


class SessionMessagesResponse(BaseModel):
    content: list[SessionContentBlock] = []
    usage: SessionUsage = SessionUsage()

    @property
    def text(self) -> str:
        return "".join(block.text or "" for block in self.content)


@dataclass(frozen=True, slots=True)
class TurnMetric:
    turn_index: int
    ok: bool
    latency_seconds: float
    uncached_input_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    failure: str | None


@dataclass(frozen=True, slots=True)
class AnomalyReport:
    planned_turns: int
    attempted_turns: int
    failed_turns: int
    warm_turns: int
    warm_uncached_input_tokens: int
    warm_cache_read_tokens: int
    warm_cache_creation_tokens: int
    p95_turn_seconds: float

    @property
    def error_ratio(self) -> float:
        return self.failed_turns / self.planned_turns if self.planned_turns else 1.0

    @property
    def warm_cache_read_share(self) -> float:
        billed = (
            self.warm_uncached_input_tokens
            + self.warm_cache_read_tokens
            + self.warm_cache_creation_tokens
        )
        return self.warm_cache_read_tokens / billed if billed else 0.0


def _system_prefix_block(marker: str) -> TextBlock:
    text = " ".join(
        f"Project context paragraph {index} for session {marker}." for index in range(300)
    )
    return TextBlock(text=text, cache_control=CacheControl())


def _user_turn_text(marker: str, turn_index: int) -> str:
    notes = " ".join(
        f"Working note {index} of turn {turn_index} in session {marker}."
        for index in range(80)
    )
    return f"Reply with one short sentence.\n{notes}"


def _reminder_turn() -> RichMessage:
    return RichMessage(
        role="system",
        content=[
            TextBlock(
                text="<system-reminder>Keep the answer to one short sentence.</system-reminder>"
            )
        ],
    )


def _without_cache_control(message: RichMessage) -> RichMessage:
    return RichMessage(
        role=message.role,
        content=[TextBlock(text=block.text) for block in message.content],
    )


RETRY_BACKOFF_SECONDS = 2.0


def retried(
    call: Callable[[], Result[SessionMessagesResponse]],
    attempts: int,
    backoff_seconds: float = RETRY_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Result[SessionMessagesResponse]:
    result = call()
    if isinstance(result, Success) or attempts <= 1:
        return result
    sleep(backoff_seconds)
    return retried(call, attempts - 1, backoff_seconds, sleep)


def _metric(
    result: Result[SessionMessagesResponse], turn_index: int, latency_seconds: float
) -> TurnMetric:
    if isinstance(result, Success):
        usage = result.data.usage
        return TurnMetric(
            turn_index=turn_index,
            ok=True,
            latency_seconds=latency_seconds,
            uncached_input_tokens=usage.input_tokens,
            cache_read_tokens=usage.cache_read_input_tokens,
            cache_creation_tokens=usage.cache_creation_input_tokens,
            failure=None,
        )
    return TurnMetric(
        turn_index=turn_index,
        ok=False,
        latency_seconds=latency_seconds,
        uncached_input_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        failure=repr(result),
    )


def _drive_turns(
    transport: Transport,
    key: str,
    model: str,
    marker: str,
    system_block: TextBlock,
    history: tuple[RichMessage, ...],
    turn_index: int,
    remaining_turns: int,
    attempts_per_turn: int,
) -> tuple[TurnMetric, ...]:
    if remaining_turns == 0:
        return ()
    user_turn = RichMessage(
        role="user",
        content=[
            TextBlock(
                text=_user_turn_text(marker, turn_index), cache_control=CacheControl()
            )
        ],
    )
    started = time.monotonic()
    result = retried(
        lambda: transport.post(
            "/v1/messages",
            headers=transport.bearer(key),
            json=SessionMessagesRequest(
                model=model,
                system=[system_block],
                messages=[*history, user_turn],
            ),
            response_type=SessionMessagesResponse,
        ),
        attempts_per_turn,
    )
    turn = _metric(result, turn_index, time.monotonic() - started)
    if not isinstance(result, Success):
        return (turn,)
    assistant_turn = RichMessage(
        role="assistant", content=[TextBlock(text=result.data.text or "Understood.")]
    )
    return (
        turn,
        *_drive_turns(
            transport,
            key,
            model,
            marker,
            system_block,
            (
                *history,
                _without_cache_control(user_turn),
                _reminder_turn(),
                assistant_turn,
            ),
            turn_index + 1,
            remaining_turns - 1,
            attempts_per_turn,
        ),
    )


def run_session(
    transport: Transport, key: str, model: str, turns: int, attempts_per_turn: int
) -> tuple[TurnMetric, ...]:
    marker = unique_marker()
    return _drive_turns(
        transport,
        key,
        model,
        marker,
        _system_prefix_block(marker),
        (),
        1,
        turns,
        attempts_per_turn,
    )


def run_concurrent_sessions(
    transport: Transport,
    key: str,
    model: str,
    sessions: int,
    turns_per_session: int,
    attempts_per_turn: int,
) -> tuple[TurnMetric, ...]:
    with ThreadPoolExecutor(max_workers=sessions) as pool:
        futures = [
            pool.submit(
                run_session, transport, key, model, turns_per_session, attempts_per_turn
            )
            for _ in range(sessions)
        ]
        return tuple(turn for future in futures for turn in future.result())


def settled_spend(
    read_spend: Callable[[], float],
    poll_interval: float,
    settle_seconds: float,
    timeout_seconds: float,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    deadline = now() + timeout_seconds + settle_seconds

    def settle(previous: float, stable_since: float) -> float:
        current = read_spend()
        observed = now()
        since = stable_since if current == previous else observed
        if current > 0 and observed - since >= settle_seconds:
            return current
        if observed >= deadline:
            raise AssertionError(
                f"key spend never held a stable non-zero value for {settle_seconds}s "
                f"within {timeout_seconds + settle_seconds}s (last read {current}); "
                f"spend stopped being recorded, which is itself a spend anomaly"
            )
        sleep(poll_interval)
        return settle(current, since)

    return settle(-1.0, now())


def _p95(latencies: tuple[float, ...]) -> float:
    if not latencies:
        return 0.0
    ranked = sorted(latencies)
    return ranked[max(0, -(-len(ranked) * 95 // 100) - 1)]


def summarize(turns: tuple[TurnMetric, ...], planned_turns: int) -> AnomalyReport:
    warm = tuple(turn for turn in turns if turn.ok and turn.turn_index >= 2)
    return AnomalyReport(
        planned_turns=planned_turns,
        attempted_turns=len(turns),
        failed_turns=planned_turns - sum(1 for turn in turns if turn.ok),
        warm_turns=len(warm),
        warm_uncached_input_tokens=sum(turn.uncached_input_tokens for turn in warm),
        warm_cache_read_tokens=sum(turn.cache_read_tokens for turn in warm),
        warm_cache_creation_tokens=sum(turn.cache_creation_tokens for turn in warm),
        p95_turn_seconds=_p95(
            tuple(turn.latency_seconds for turn in turns if turn.ok)
        ),
    )
