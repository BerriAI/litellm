from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping
from itertools import accumulate, chain, pairwise
from typing import TYPE_CHECKING, Final, Literal

import httpx

from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # existing decorator has untyped provider arguments
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared factory accepts untyped provider params
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.patronus_ark import (
    ArkConfig,
    PatronusArkConfigModel,
    Source,
    WolfJob,
    WolfSubmission,
)
from litellm.types.utils import GenericGuardrailAPIInputs

from .content import rewrite_message

if TYPE_CHECKING:
    from patronus_ark_core import EvidenceSpan, NativeResult, ScanControl, Scanner

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


def _scanner(config: Mapping[str, object], wolf_options: Mapping[str, int]) -> Scanner:
    from patronus_ark_core import Scanner

    return Scanner(dict(config), wolf_options=dict(wolf_options))


def _control(timeout_ms: int) -> ScanControl:
    from patronus_ark_core import ScanControl

    return ScanControl(timeout_ms=timeout_ms)


def _redact(text: str, spans: tuple[EvidenceSpan, ...]) -> str:
    if not spans:
        return text
    if any(
        span["start_char"] < 0 or span["end_char"] > len(text) or span["end_char"] <= span["start_char"]
        for span in spans
    ):
        raise ValueError("Ark returned invalid evidence offsets")
    ordered: Final = tuple(sorted((span["start_char"], span["end_char"]) for span in spans))
    ends: Final = tuple(accumulate((end for _, end in ordered), max))
    breaks: Final = (
        0,
        *tuple(index for index in range(1, len(ordered)) if ordered[index][0] > ends[index - 1]),
        len(ordered),
    )
    merged: Final = tuple((ordered[left][0], ends[right - 1]) for left, right in pairwise(breaks))
    return (
        "".join(
            text[0 if index == 0 else merged[index - 1][1] : start] + "[REDACTED]"
            for index, (start, _) in enumerate(merged)
        )
        + text[merged[-1][1] :]
    )


class PatronusArkGuardrail(CustomGuardrail):
    def __init__(
        self,
        ark: ArkConfig | None = None,
        *,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks  # mutable-ok: public base configuration contract
        | list[GuardrailEventHooks]
        | Mode
        | None = None,
        default_on: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        scanner: Scanner | None = None,
        control_factory: Callable[[int], ScanControl] = _control,
        async_handler: AsyncHTTPHandler | None = None,
    ) -> None:
        self.ark = ark or ArkConfig()
        self.scan_timeout = timeout if timeout is not None else 10.0
        if not 0 < self.scan_timeout < float("inf"):
            raise ValueError("Ark timeout must be positive and finite")
        self._scanner = scanner or _scanner(
            {
                "categories": ["pii", "dlp", "injection"],
                "pii_mode": self.ark.pii_mode,
                "dlp_mode": self.ark.dlp_mode,
                "rules": dict(self.ark.rules),
                "models": {"native:injection_l1": self.ark.native_pi},
                "max_input_bytes": self.ark.max_input_bytes,
                "max_findings": self.ark.max_findings,
                **(
                    {
                        "wolf": {
                            "bundle_dir": self.ark.wolf.bundle_dir,
                            "threshold": self.ark.wolf.threshold,
                            "precision": self.ark.wolf.precision,
                            "backend": "cpu",
                        }
                    }
                    if self.ark.wolf.backend == "local"
                    else {}
                ),
            },
            {
                "window_tokens": self.ark.wolf.window_tokens,
                "overlap_tokens": self.ark.wolf.overlap_tokens,
                "batch_size": self.ark.wolf.batch_size,
            },
        )
        self._control_factory = control_factory
        self._scan_lock = asyncio.Lock()
        self._pending = 0
        self.api_key = api_key
        self.api_base = (api_base or "https://control.patronus.studio").rstrip("/")
        if self.ark.wolf.backend == "api" and not self.api_key:
            raise ValueError("Wolf API requires api_key")
        self.async_handler = async_handler or get_async_httpx_client(
            httpxSpecialProvider.GuardrailCallback,
            {"follow_redirects": False},
        )
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # base accepts provider-specific untyped kwargs
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            supported_event_hooks=self.get_supported_event_hooks(),
            timeout=timeout,
        )

    @staticmethod
    def get_config_model() -> type[PatronusArkConfigModel]:
        return PatronusArkConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: base hook metadata contract
        return [GuardrailEventHooks.pre_call]

    def supports_structured_messages(self) -> bool:
        return False

    def _blocked(self, category: str) -> GuardrailRaisedException:
        return GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=f"Blocked by Patronus Ark {category} policy",
            blocked_content=True,
        )

    async def _scan_wolf(self, text: str, deadline: float) -> bool:
        budget: Final = deadline - asyncio.get_running_loop().time()
        if budget <= 0:
            raise TimeoutError("Wolf deadline exceeded")
        response: Final[httpx.Response] = await self.async_handler.post(  # pyright: ignore[reportUnknownMemberType, reportAny]  # shared HTTP handler returns an httpx response
            url=f"{self.api_base}/api/v1/scan",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "text": text,
                "config": {
                    "categories": ["injection"],
                    "max_level": "L3",
                    "gates": {"l1": False, "l2": False, "l3": True},
                },
            },
            timeout=budget,
        )
        response.raise_for_status()
        if response.status_code not in (200, 202):
            raise ValueError("Unexpected Wolf submission status")
        submission: Final = WolfSubmission.model_validate_json(response.content)
        job = submission.jobs[0]  # rebind-ok: polling replaces immutable job snapshots
        job_id: Final = job.job_id
        while job.status == "running":
            if re.fullmatch(r"job_[0-9a-f]{32}", job_id) is None:
                raise ValueError("Invalid Wolf job ID")
            await asyncio.sleep(0.1)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Wolf deadline exceeded")
            poll: httpx.Response = await self.async_handler.get(  # pyright: ignore[reportUnknownMemberType, reportAny]  # shared handler returns an httpx response
                url=f"{self.api_base}/api/v1/scan/{job_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=remaining,
                follow_redirects=False,
            )
            poll.raise_for_status()
            job = WolfJob.model_validate_json(poll.content)
            if job.job_id != job_id:
                raise ValueError("Wolf job ID changed")
        if (
            job.status != "completed"
            or job.completion is None
            or job.completion.state != "complete"
            or job.completion.failures
        ):
            raise ValueError("Wolf scan is incomplete")
        result: Final = job.categories.get("injection")
        if result is None:
            raise ValueError("Wolf injection result is missing")
        return result.final_result.class_name == "attack"

    async def _scan(self, text: str, source: Source, pi_only: bool, deadline: float) -> str:
        if len(text.encode("utf-8")) > self.ark.max_input_bytes:
            raise ValueError("Ark input limit exceeded")
        pii: Final = not pi_only and self.ark.pii != "off" and source in self.ark.pii_sources
        dlp: Final = not pi_only and self.ark.dlp != "off" and source in self.ark.dlp_sources
        pi: Final = (pi_only or source != "tool_call") and source in self.ark.pi_sources
        if not text.strip() or not (pii or dlp or (pi and (self.ark.native_pi or self.ark.wolf.backend != "off"))):
            return text
        remaining: Final = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Ark deadline exceeded")
        control: Final = self._control_factory(max(1, int(remaining * 1000)))
        task: Final = asyncio.create_task(
            asyncio.to_thread(
                self._scanner.scan,
                text,
                pii=pii,
                dlp=dlp,
                injection=pi,
                control=control,
            )
        )
        try:
            report: Final = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except (asyncio.CancelledError, TimeoutError):
            control.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        if pi and self.ark.wolf.backend == "local" and report["wolf"] is None:
            raise ValueError("Wolf result is missing")
        if report["wolf"] is not None and report["wolf"]["matched"]:
            raise self._blocked("prompt-injection")
        groups: Final = tuple(self._result_spans(result, pii, dlp, pi) for result in report["native"])
        if pi and self.ark.wolf.backend == "api" and await self._scan_wolf(text, deadline):
            raise self._blocked("prompt-injection")
        return _redact(text, tuple(chain.from_iterable(groups)))

    def _result_spans(self, result: NativeResult, pii: bool, dlp: bool, pi: bool) -> tuple[EvidenceSpan, ...]:
        if result["class_name"] == "error":
            raise ValueError("Ark detector failed")
        if result["class_name"] in ("safe", "benign"):
            return ()
        category: Final = result["category"]
        if category == "injection":
            if pi and self.ark.native_pi:
                raise self._blocked("prompt-injection")
            return ()
        if not (pii if category == "pii" else dlp):
            return ()
        action: Final = self.ark.pii if category == "pii" else self.ark.dlp
        if action == "block" or not result["evidence_spans"]:
            raise self._blocked(category)
        return tuple(result["evidence_spans"])

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: shared hook contract carries the proxy payload
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "request" or not isinstance(request_data.get("messages"), list):
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name, message="Ark supports Chat Completions input"
            )
        if self._pending >= self.ark.max_queued + 1:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name, status_code=429, message="Ark scan queue is full"
            )
        self._pending += 1
        try:
            deadline: Final = asyncio.get_running_loop().time() + self.scan_timeout
            await asyncio.wait_for(self._scan_lock.acquire(), timeout=min(self.ark.queue_timeout, self.scan_timeout))
            try:

                async def visit(text: str, source: Source, pi_only: bool) -> str:
                    return await self._scan(text, source, pi_only, deadline)

                messages: Final = inputs.get("structured_messages")
                if messages is None:
                    raise ValueError("Ark requires structured Chat Completions messages")
                rewritten: Final = [await rewrite_message(message, visit) for message in messages]
                return {**inputs, "structured_messages": rewritten}
            finally:
                self._scan_lock.release()
        except GuardrailRaisedException:
            raise
        except Exception as error:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name, status_code=503, message="Ark analysis could not complete"
            ) from error
        finally:
            self._pending -= 1
