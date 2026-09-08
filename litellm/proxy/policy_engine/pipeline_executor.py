"""
Pipeline Executor - Executes guardrail pipelines with conditional step logic.

Runs guardrails sequentially per pipeline step definitions, handling
pass/fail actions (allow, block, next, modify_response) and data forwarding.
"""

import time
from collections.abc import Sequence
from typing import Any, Final, Literal

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

try:
    from fastapi.exceptions import HTTPException
except ImportError:
    HTTPException = None


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

            # Forward modified data to next step if pass_data is True
            if step.pass_data and modified_data is not None:
                working_data = {**working_data, **modified_data}

            # Handle terminal actions
            if action == "allow":
                return _allow_result(step_results=step_results, working_data=working_data, request_data=data)

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
        return _allow_result(step_results=step_results, working_data=working_data, request_data=data)

    @staticmethod
    async def _run_step(
        step: PipelineStep,
        mode: str,
        data: dict,
        user_api_key_dict: Any,
        call_type: str,
        raw_request_snapshot: dict | None = None,  # mutable-ok: same request-payload shape as data
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
            # Inject guardrail name into metadata so should_run_guardrail() allows it
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["guardrails"] = [step.guardrail]

            # A scan_raw_request step evaluates the pristine pre-pipeline
            # snapshot instead of `data` (which earlier pass_data steps in
            # this same pipeline may have already rewritten), same reason
            # the normal sequential/parallel guardrail loops do this.
            scans_raw_request: Final = callback.scan_raw_request
            hook_input: Final[dict] = (  # mutable-ok: same request-payload shape as data
                independent_snapshot(raw_request_snapshot)
                if scans_raw_request and raw_request_snapshot is not None
                else data
            )
            if hook_input is not data:
                hook_input.setdefault("metadata", {})["guardrails"] = [step.guardrail]

            # Use unified_guardrail path if callback implements apply_guardrail
            target: CustomLogger = callback
            use_unified: Final = (
                "apply_guardrail" in type(callback).__dict__ and not callback.use_native_lifecycle_hooks
            )
            if use_unified:
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
            # this pipeline already did.
            modified_data = None
            if response is not None and isinstance(response, dict) and not scans_raw_request:
                modified_data = response
            return ("pass", modified_data, None, None)

        except Exception as e:
            if CustomGuardrail._is_guardrail_intervention(e):
                error_msg: Final = _extract_error_message(e)
                return ("fail", None, error_msg, e)
            else:
                verbose_proxy_logger.error("Pipeline: unexpected error from guardrail '%s': %s", step.guardrail, e)
                return ("error", None, str(e), e)

    @staticmethod
    def find_guardrail_callback(guardrail_name: str) -> CustomGuardrail | None:
        """Look up an initialized guardrail callback by name from litellm.callbacks."""
        for callback in litellm.callbacks:
            if isinstance(callback, CustomGuardrail):
                if callback.guardrail_name == guardrail_name:
                    return callback
        return None


def _allow_result(
    step_results: Sequence[PipelineStepResult],
    working_data: dict,  # mutable-ok: same request-payload shape as execute_steps' data
    request_data: dict,  # mutable-ok: same request-payload shape as execute_steps' data
) -> PipelineExecutionResult:
    """Build the terminal-allow result, propagating pipeline modifications without the per-step guardrail override."""
    restored: Final = _restore_request_guardrails(working_data, request_data)
    return PipelineExecutionResult(
        terminal_action="allow",
        step_results=list(step_results),  # mutable-ok: PipelineExecutionResult field is a list
        modified_data=restored if restored != request_data else None,
    )


def _restore_request_guardrails(
    working_data: dict,  # mutable-ok: same request-payload shape as execute_steps' data
    request_data: dict,  # mutable-ok: same request-payload shape as execute_steps' data
) -> dict:  # mutable-ok: merged back into the request dict, which downstream code mutates
    """
    Restore the request's own metadata["guardrails"] activation list.

    _run_step overrides it to [step.guardrail] so should_run_guardrail() allows each
    step; letting that override escape via modified_data permanently drops every
    independently activated guardrail from later lifecycle stages (post_call, etc.).
    """
    working_metadata: Final = working_data.get("metadata")
    if not isinstance(working_metadata, dict):
        return working_data
    request_metadata: Final = request_data.get("metadata")
    original_guardrails: Final = request_metadata.get("guardrails") if isinstance(request_metadata, dict) else None
    stripped: Final = {k: v for k, v in working_metadata.items() if k != "guardrails"}  # mutable-ok: request dict
    if original_guardrails is not None:
        restored: Final = {**stripped, "guardrails": original_guardrails}  # mutable-ok: request dict
        return {**working_data, "metadata": restored}  # mutable-ok: request dict
    if not stripped and not isinstance(request_metadata, dict):
        return {k: v for k, v in working_data.items() if k != "metadata"}  # mutable-ok: request dict
    return {**working_data, "metadata": stripped}  # mutable-ok: request dict


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
