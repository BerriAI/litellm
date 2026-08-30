"""
Pipeline Executor - Executes guardrail pipelines with conditional step logic.

Runs guardrails sequentially per pipeline step definitions, handling
pass/fail actions (allow, block, next, modify_response) and data forwarding.
"""

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import independent_snapshot
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.proxy.policy_engine.pipeline_types import (
    PipelineExecutionResult,
    PipelineStep,
    PipelineStepResult,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.guardrail_translation.base_translation import (
        BaseTranslation,
    )

try:
    from fastapi.exceptions import HTTPException
except ImportError:
    HTTPException = None


class UndeliverableStreamRewrite(Exception):
    def __init__(self, guardrail_name: str) -> None:
        super().__init__(
            f"Guardrail '{guardrail_name}' rewrote the streamed response, which streaming pipelines cannot deliver"
        )
        self.guardrail_name: Final = guardrail_name


def _tool_call_shape(tool_call: object) -> tuple[object, object]:
    plain: Final = tool_call.model_dump() if isinstance(tool_call, BaseModel) else tool_call
    function: Final = plain.get("function") if isinstance(plain, Mapping) else None
    if not isinstance(function, Mapping):
        return (None, None)
    return (function.get("name"), function.get("arguments"))


def _texts_snapshot(texts: Sequence[str] | None) -> tuple[str, ...] | None:
    return None if texts is None else tuple(texts)


def _tool_call_shapes_snapshot(
    tool_calls: Sequence[object] | None,
) -> tuple[tuple[object, object], ...] | None:
    return None if tool_calls is None else tuple(_tool_call_shape(tc) for tc in tool_calls)


class _StreamRewriteObserver(CustomGuardrail):
    """Stand-in handed to the endpoint translation in place of a streaming pipeline step's
    guardrail. Translations cannot rewrite every buffered chunk consistently, so the gate
    withholds the stream whenever the guardrail returned different output than it was given,
    which for guardrails like Bedrock's ANONYMIZED action is only known at runtime."""

    def __init__(self, inner: CustomGuardrail) -> None:
        super().__init__(guardrail_name=inner.guardrail_name)
        self.inner: Final = inner
        self.rewrote = False

    def structured_messages_cover_full_request(self) -> bool:
        return self.inner.structured_messages_cover_full_request()

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,  # mutable-ok: matches CustomGuardrail.apply_guardrail
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        # Snapshot inputs *before* dispatching. Guardrails that rewrite in
        # place (``inputs["texts"] = masked; return inputs`` or mutating the
        # same list/tool-call objects) leave the pre and post views pointing
        # at the already-rewritten values, which would hide the rewrite from
        # a post-call diff and let the original buffered chunks reach the
        # client unredacted.
        sent_texts: Final = _texts_snapshot(inputs.get("texts"))
        sent_tool_call_shapes: Final = _tool_call_shapes_snapshot(inputs.get("tool_calls"))
        outputs: Final = await self.inner.apply_guardrail(
            inputs=inputs, request_data=request_data, input_type=input_type, logging_obj=logging_obj
        )
        returned_texts: Final = _texts_snapshot(outputs.get("texts"))
        returned_tool_call_shapes: Final = _tool_call_shapes_snapshot(outputs.get("tool_calls"))
        texts_rewrote: Final = (
            sent_texts is not None and returned_texts is not None and returned_texts != sent_texts
        )
        tool_calls_rewrote: Final = (
            sent_tool_call_shapes is not None
            and returned_tool_call_shapes is not None
            and returned_tool_call_shapes != sent_tool_call_shapes
        )
        self.rewrote = self.rewrote or texts_rewrote or tool_calls_rewrote
        return outputs


def _prepare_hook_input(
    step: PipelineStep,
    callback: CustomGuardrail,
    data: dict,  # mutable-ok: same request-payload shape the hooks mutate
    raw_request_snapshot: dict | None,  # mutable-ok: same request-payload shape as data
) -> tuple[dict, bool]:  # mutable-ok: returns that same request-payload dict
    """Inject the step's guardrail name into metadata so should_run_guardrail() allows it,
    and pick the payload the step scans: a scan_raw_request step evaluates the pristine
    pre-pipeline snapshot instead of `data` (which earlier pass_data steps in this same
    pipeline may have already rewritten), same reason the normal sequential/parallel
    guardrail loops do this."""
    if "metadata" not in data:
        data["metadata"] = {}  # mutable-ok: request metadata bucket, hooks mutate it
    data["metadata"]["guardrails"] = [
        step.guardrail
    ]  # mutable-ok: guardrails list is part of the request-payload shape

    scans_raw_request: Final = callback.scan_raw_request
    hook_input: Final[dict] = (  # mutable-ok: same request-payload shape as data
        independent_snapshot(raw_request_snapshot) if scans_raw_request and raw_request_snapshot is not None else data
    )
    if hook_input is not data:
        hook_input.setdefault("metadata", {})["guardrails"] = [step.guardrail]  # mutable-ok: request metadata shape
    return hook_input, scans_raw_request


class PipelineExecutor:
    """Executes guardrail pipelines with ordered, conditional step logic."""

    @staticmethod
    async def execute_steps(
        steps: list[PipelineStep],
        mode: str,
        data: dict,
        user_api_key_dict: Any,
        call_type: str,
        policy_name: str,
        raw_request_snapshot: dict | None = None,  # mutable-ok: same request-payload shape as data
        streaming_chunks: list[Any] | None = None,  # mutable-ok: shared buffered-stream chunks, read per step
        endpoint_translation: "BaseTranslation | None" = None,
    ) -> PipelineExecutionResult:
        """
        Execute pipeline steps sequentially with conditional actions.

        Args:
            steps: Ordered list of pipeline steps
            mode: Event hook mode (pre_call, post_call)
            data: Request data dict
            user_api_key_dict: User API key auth
            call_type: Type of call (completion, etc.)
            policy_name: Name of the owning policy (for logging)
            raw_request_snapshot: pristine pre-pipeline, pre-guardrail request
                (taken by the caller before any guardrail or pipeline ran), so a
                step whose guardrail opted into ``scan_raw_request`` evaluates
                the original request instead of whatever an earlier
                ``pass_data`` step in this same pipeline already rewrote.
            streaming_chunks: buffered chunks of a completed stream. When set
                (with ``endpoint_translation``), post_call steps scan the
                assembled streamed output through the endpoint translation
                instead of calling ``async_post_call_success_hook``.
            endpoint_translation: the guardrail translation for the streamed
                endpoint, resolved by the caller.

        Returns:
            PipelineExecutionResult with terminal action and step results
        """
        step_results: Final[list[PipelineStepResult]] = []
        working_data = data.copy()
        if "metadata" in working_data:
            working_data["metadata"] = working_data["metadata"].copy()

        for i, step in enumerate(steps):
            start_time = time.perf_counter()

            (
                outcome,
                modified_data,
                error_detail,
                original_exception,
            ) = await PipelineExecutor._run_step(
                step=step,
                mode=mode,
                data=working_data,
                user_api_key_dict=user_api_key_dict,
                call_type=call_type,
                raw_request_snapshot=raw_request_snapshot,
                streaming_chunks=streaming_chunks,
                endpoint_translation=endpoint_translation,
            )

            duration = time.perf_counter() - start_time

            action = _pipeline_action_for_outcome(step, outcome)

            step_result = PipelineStepResult(
                guardrail_name=step.guardrail,
                outcome=outcome,
                action_taken=action,
                modified_data=modified_data,
                error_detail=error_detail,
                duration_seconds=round(duration, 4),
            )
            step_results.append(step_result)

            verbose_proxy_logger.debug(
                "Pipeline '%s' step %s: guardrail=%s, outcome=%s, action=%s",
                policy_name,
                i,
                step.guardrail,
                outcome,
                action,
            )

            # Forward modified data to the next step if pass_data is True;
            # post_call response replacements always chain, matching the flat
            # callback loop where each hook sees the previous hook's response
            if modified_data is not None and (step.pass_data or mode == "post_call"):
                working_data = {**working_data, **modified_data}

            # Handle terminal actions
            if action == "allow":
                return PipelineExecutionResult(
                    terminal_action="allow",
                    step_results=step_results,
                    modified_data=working_data if working_data != data else None,
                )

            if action == "block":
                return PipelineExecutionResult(
                    terminal_action="block",
                    step_results=step_results,
                    error_message=error_detail,
                    original_exception=original_exception,
                )

            if action == "modify_response":
                return PipelineExecutionResult(
                    terminal_action="modify_response",
                    step_results=step_results,
                    modify_response_message=step.modify_response_message or error_detail,
                )

            # action == "next" → continue to next step

        # Ran out of steps without a terminal action → default allow
        return PipelineExecutionResult(
            terminal_action="allow",
            step_results=step_results,
            modified_data=working_data if working_data != data else None,
        )

    @staticmethod
    async def _run_step(
        step: PipelineStep,
        mode: str,
        data: dict,
        user_api_key_dict: Any,
        call_type: str,
        raw_request_snapshot: dict | None = None,  # mutable-ok: same request-payload shape as data
        streaming_chunks: list[Any] | None = None,  # mutable-ok: shared buffered-stream chunks, read per step
        endpoint_translation: "BaseTranslation | None" = None,
    ) -> tuple[
        Literal["pass", "fail", "error"],
        dict | None,
        str | None,
        Exception | None,
    ]:
        """
        Run a single pipeline step's guardrail.

        Returns:
            Tuple of (outcome, modified_data, error_detail, original_exception):
            - outcome: "pass", "fail", or "error"
            - modified_data: dict if guardrail returned modified data, else None
            - error_detail: error message string if fail/error, else None
            - original_exception: the exception the guardrail raised, so the
              pipeline can re-raise it verbatim and match the direct-attachment
              response/trace, else None
        """
        callback: Final = PipelineExecutor.find_guardrail_callback(step.guardrail)
        if callback is None:
            verbose_proxy_logger.warning("Pipeline: guardrail '%s' not found in callbacks", step.guardrail)
            return ("error", None, f"Guardrail '{step.guardrail}' not found", None)

        try:
            hook_input, scans_raw_request = _prepare_hook_input(step, callback, data, raw_request_snapshot)

            # Use unified_guardrail path if callback implements apply_guardrail
            target: CustomLogger = callback
            use_unified: Final = PipelineExecutor.supports_unified_execution(callback)
            if use_unified and streaming_chunks is None:
                hook_input["guardrail_to_apply"] = callback
                target = UnifiedLLMGuardrails()

            if mode == "pre_call":
                response = await target.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=None,
                    data=hook_input,
                    call_type=call_type,
                )
                if isinstance(callback, CustomGuardrail):
                    callback.mark_pre_call_hook_ran(data)
                    if isinstance(response, dict):
                        callback.mark_pre_call_hook_ran(response)
            elif mode == "post_call" and streaming_chunks is not None:
                if not use_unified or endpoint_translation is None:
                    return (
                        "error",
                        None,
                        f"Guardrail '{step.guardrail}' does not support streaming pipeline execution",
                        None,
                    )
                observer: Final = _StreamRewriteObserver(callback)
                await endpoint_translation.process_output_streaming_response(
                    responses_so_far=streaming_chunks,
                    guardrail_to_apply=observer,
                    litellm_logging_obj=data.get("litellm_logging_obj"),
                    user_api_key_dict=user_api_key_dict,
                    request_data=hook_input,
                )
                if observer.rewrote:
                    raise UndeliverableStreamRewrite(step.guardrail)
                response = None
            elif mode == "post_call":
                response = await target.async_post_call_success_hook(
                    user_api_key_dict=user_api_key_dict,
                    data=data,
                    response=data.get("response"),
                )
            else:
                return ("error", None, f"Unsupported pipeline mode: {mode}", None)

            # Normal return means pass. A scan_raw_request step is block-only,
            # same contract as run_in_parallel/scan_raw_request elsewhere: any
            # data it returned is discarded, since applying it on top of the
            # raw snapshot would silently undo whatever an earlier step in
            # this pipeline already did. A post_call hook's non-None return is
            # a replacement response (the flat callback-loop contract), carried
            # under the same "response" key the step input uses.
            if response is None or scans_raw_request:
                return ("pass", None, None, None)
            if mode == "post_call":
                return (
                    "pass",
                    {"response": response},
                    None,
                    None,
                )  # mutable-ok: modified-data contract is a plain dict
            return ("pass", response if isinstance(response, dict) else None, None, None)

        except UndeliverableStreamRewrite:
            raise
        except Exception as e:
            if CustomGuardrail._is_guardrail_intervention(e):
                error_msg: Final = _extract_error_message(e)
                return ("fail", None, error_msg, e)
            else:
                verbose_proxy_logger.error("Pipeline: unexpected error from guardrail '%s': %s", step.guardrail, e)
                return ("error", None, str(e), e)

    @staticmethod
    def supports_unified_execution(callback: CustomGuardrail) -> bool:
        """Whether this guardrail runs through the unified apply_guardrail path,
        the interface streaming pipeline execution requires."""
        return "apply_guardrail" in type(callback).__dict__ and not callback.use_native_lifecycle_hooks

    @staticmethod
    def find_guardrail_callback(guardrail_name: str) -> CustomGuardrail | None:
        """Look up an initialized guardrail callback by name from litellm.callbacks."""
        for callback in litellm.callbacks:
            if isinstance(callback, CustomGuardrail):
                if callback.guardrail_name == guardrail_name:
                    return callback
        return None


def _pipeline_action_for_outcome(step: PipelineStep, outcome: str) -> str:
    """
    Map pipeline step outcome to the configured action.

    - pass -> on_pass
    - fail -> on_fail (content/policy intervention)
    - error -> on_error if set, else on_fail (backward compatible)
    """
    if outcome == "pass":
        return step.on_pass
    if outcome == "fail":
        return step.on_fail
    if step.on_error is not None:
        return step.on_error
    return step.on_fail


def _extract_error_message(e: Exception) -> str:
    """Extract a human-readable error message from a guardrail exception."""
    if isinstance(e, ModifyResponseException):
        return str(e)
    if HTTPException is not None and isinstance(e, HTTPException):
        detail: Final = getattr(e, "detail", None)
        if detail:
            return str(detail)
    return str(e)
