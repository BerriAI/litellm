import asyncio
import importlib
from collections.abc import Mapping
from itertools import accumulate, pairwise
from typing import ClassVar, Final, Protocol, runtime_checkable

import httpx
from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared client factory has untyped provider params
    httpxSpecialProvider,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.guardrails.guardrail_hooks.patronus_ark import (
    ArkConfig,
    ArkNativeResult,
    ArkReport,
    ArkSpan,
    Source,
)
from litellm.types.utils import CallTypesLiteral

from .content import rewrite_message
from .wolf_api import WolfAPI


class Control(Protocol):
    def cancel(self) -> None: ...


class Scanner(Protocol):
    def scan(self, text: str, *, pii: bool, dlp: bool, injection: bool, control: Control) -> object: ...


@runtime_checkable
class CoreModule(Protocol):
    def Scanner(self, config: Mapping[str, object], *, wolf_options: Mapping[str, int]) -> Scanner: ...
    def ScanControl(self, timeout_ms: int) -> Control: ...


def _redact(text: str, spans: tuple[ArkSpan, ...]) -> str:
    if not spans:
        return text
    if any(span.end_char > len(text) or span.end_char <= span.start_char for span in spans):
        raise ValueError("Ark returned invalid evidence offsets")
    ordered: Final = tuple(sorted((span.start_char, span.end_char) for span in spans))
    ends: Final = tuple(accumulate((end for _, end in ordered), max))
    breaks: Final = (
        0,
        *tuple(index for index in range(1, len(ordered)) if ordered[index][0] > ends[index - 1]),
        len(ordered),
    )
    merged: Final = tuple((ordered[left][0], ends[right - 1]) for left, right in pairwise(breaks))
    pieces: Final = tuple(
        text[0 if index == 0 else merged[index - 1][1] : start] + "[REDACTED]"
        for index, (start, _) in enumerate(merged)
    )
    return "".join(pieces) + text[merged[-1][1] :]


class PatronusArkGuardrail(CustomGuardrail):
    use_native_lifecycle_hooks: ClassVar[bool] = True

    @classmethod
    def get_supported_event_hooks(
        cls,
    ) -> list[GuardrailEventHooks]:  # mutable-ok: public CustomGuardrail contract returns a list
        return [GuardrailEventHooks.pre_call]

    def __init__(
        self,
        ark: ArkConfig | None = None,
        *,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks  # mutable-ok: public CustomGuardrail configuration contract
        | list[GuardrailEventHooks]
        | Mode
        | None = None,
        default_on: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        core_module: CoreModule | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.ark = ark or ArkConfig()
        self.scan_timeout = timeout if timeout is not None else 10.0
        if not 0 < self.scan_timeout < float("inf"):
            raise ValueError("Ark timeout must be positive and finite")
        module: Final = core_module or importlib.import_module("patronus_ark_core")
        if not isinstance(module, CoreModule):
            raise TypeError("Install patronus-ark-core with Scanner and ScanControl support")
        self._core = module
        self._scanner = self._core.Scanner(
            {
                "categories": ["pii", "dlp", "injection"],
                "pii_mode": self.ark.pii_mode,
                "dlp_mode": self.ark.dlp_mode,
                "rules": self.ark.rules,
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
            wolf_options={
                "window_tokens": self.ark.wolf.window_tokens,
                "overlap_tokens": self.ark.wolf.overlap_tokens,
                "batch_size": self.ark.wolf.batch_size,
            },
        )
        self._scan_lock = asyncio.Lock()
        self._pending = 0
        resolved_key: Final = api_key or get_secret_str("PATRONUS_API_KEY")
        if self.ark.wolf.backend == "api" and not resolved_key:
            raise ValueError("Wolf API requires PATRONUS_API_KEY or api_key")
        self._wolf_api = (
            WolfAPI(
                http_client
                or get_async_httpx_client(httpxSpecialProvider.GuardrailCallback, {"follow_redirects": False}).client,
                api_base or "https://control.patronus.studio",
                resolved_key or "",
            )
            if self.ark.wolf.backend == "api"
            else None
        )
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail accepts untyped provider kwargs
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            supported_event_hooks=[GuardrailEventHooks.pre_call],
            timeout=timeout,
        )

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
            raise TimeoutError("Ark scan deadline exceeded")
        control: Final = self._core.ScanControl(timeout_ms=max(1, int(remaining * 1000)))
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
            raw: Final = await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except (asyncio.CancelledError, TimeoutError):
            control.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        report: Final = ArkReport.model_validate(raw)
        if report.input_bytes != len(text.encode("utf-8")):
            raise ValueError("Ark report does not match input")
        expected: Final = frozenset(
            name for name, enabled in (("pii", pii), ("dlp", dlp), ("injection", pi and self.ark.native_pi)) if enabled
        )
        if not expected.issubset(result.category for result in report.native):
            raise ValueError("Ark report is incomplete")
        if pi and self.ark.wolf.backend == "local" and report.wolf is None:
            raise ValueError("Wolf result is missing")
        if report.wolf is not None and report.wolf.matched:
            raise HTTPException(status_code=400, detail="Blocked by Patronus Ark prompt-injection policy")
        spans: Final = tuple(self._result_spans(result, pii, dlp, pi) for result in report.native)
        if pi and self._wolf_api is not None:
            api_budget: Final = deadline - asyncio.get_running_loop().time()
            if api_budget <= 0:
                raise TimeoutError("Ark scan deadline exceeded")
            if await asyncio.wait_for(self._wolf_api.scan(text, api_budget), timeout=api_budget):
                raise HTTPException(status_code=400, detail="Blocked by Patronus Ark prompt-injection policy")
        selected: Final = tuple(
            span for group in spans for span in group
        )  # comprehension-ok: flatten validated evidence groups
        return _redact(text, selected)

    def _result_spans(self, result: ArkNativeResult, pii: bool, dlp: bool, pi: bool) -> tuple[ArkSpan, ...]:
        if result.class_name == "error":
            raise ValueError("Ark detector failed")
        if result.class_name in ("safe", "benign"):
            return ()
        if result.category == "injection":
            if pi and self.ark.native_pi:
                raise HTTPException(status_code=400, detail="Blocked by Patronus Ark prompt-injection policy")
            return ()
        enabled: Final = pii if result.category == "pii" else dlp
        if not enabled:
            return ()
        action: Final = self.ark.pii if result.category == "pii" else self.ark.dlp
        if action == "block" or not result.evidence_spans:
            raise HTTPException(status_code=400, detail=f"Blocked by Patronus Ark {result.category} policy")
        return result.evidence_spans

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: litellm.DualCache,
        data: Mapping[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:  # mutable-ok: proxy dispatch requires a mutable request dictionary
        if not self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call):  # pyright: ignore[reportUnknownMemberType]  # base hook accepts the untyped proxy payload
            return dict(data)
        if call_type not in ("acompletion", "completion"):
            raise HTTPException(status_code=400, detail="Patronus Ark currently supports Chat Completions only")
        if self._pending >= self.ark.max_queued + 1:
            raise HTTPException(status_code=429, detail="Patronus Ark scan queue is full")
        self._pending += 1
        try:
            return await self._guard(data)
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=503, detail="Patronus Ark analysis could not complete") from error
        finally:
            self._pending -= 1

    async def _guard(
        self, data: Mapping[str, object]
    ) -> dict[str, object]:  # mutable-ok: proxy dispatch requires a mutable request dictionary
        deadline: Final = asyncio.get_running_loop().time() + self.scan_timeout
        await asyncio.wait_for(self._scan_lock.acquire(), timeout=min(self.ark.queue_timeout, self.scan_timeout))
        try:
            messages: Final = TypeAdapter(list[JsonValue]).validate_python(data.get("messages"))

            async def visit(text: str, source: Source, pi_only: bool) -> str:
                return await self._scan(text, source, pi_only, deadline)

            rewritten: Final = [await rewrite_message(message, visit) for message in messages]
            return {**data, "messages": rewritten}
        finally:
            self._scan_lock.release()
