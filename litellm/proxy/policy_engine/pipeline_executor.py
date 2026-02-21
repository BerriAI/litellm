"""
Pipeline Executor - Executes guardrail pipelines with conditional step logic.

Runs guardrails sequentially per pipeline step definitions, handling
pass/fail actions (allow, block, next, modify_response) and data forwarding.
"""

import time
from typing import Any, List, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.integrations.custom_logger import CustomLogger
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
    HTTPException = None  # type: ignore


class PipelineExecutor:
    """Executes guardrail pipelines with ordered, conditional step logic."""

    @staticmethod
    async def execute_steps(
        steps: List[PipelineStep],
        mode: str,
        data: dict,
        user_api_key_dict: Any,
        call_type: str,
        policy_name: str,
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

        Returns:
            PipelineExecutionResult with terminal action and step results
        """
        step_results: List[PipelineStepResult] = []
        working_data = data.copy()
        if "metadata" in working_data:
            working_data["metadata"] = working_data["metadata"].copy()

        for i, step in enumerate(steps):
            start_time = time.perf_counter()

            outcome, modified_data, error_detail = await PipelineExecutor._run_step(
                step=step,
                mode=mode,
                data=working_data,
                user_api_key_dict=user_api_key_dict,
                call_type=call_type,
            )

            duration = time.perf_counter() - start_time

            action = step.on_pass if outcome == "pass" else step.on_fail

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
                f"Pipeline '{policy_name}' step {i}: guardrail={step.guardrail}, "
                f"outcome={outcome}, action={action}"
            )

            # Forward modified data to next step if pass_data is True
            if step.pass_data and modified_data is not None:
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
    ) -> tuple:
        """
        Run a single pipeline step's guardrail.

        Returns:
            Tuple of (outcome, modified_data, error_detail) where:
            - outcome: "pass", "fail", or "error"
            - modified_data: dict if guardrail returned modified data, else None
            - error_detail: error message string if fail/error, else None
        """
        callback = PipelineExecutor._find_guardrail_callback(step.guardrail)
        if callback is None:
            verbose_proxy_logger.warning(
                f"Pipeline: guardrail '{step.guardrail}' not found in callbacks"
            )
            return ("error", None, f"Guardrail '{step.guardrail}' not found")

        try:
            # Inject guardrail name into metadata so should_run_guardrail() allows it
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["guardrails"] = [step.guardrail]

            # Use unified_guardrail path if callback implements apply_guardrail
            target: CustomLogger = callback
            use_unified = "apply_guardrail" in type(callback).__dict__
            if use_unified:
                data["guardrail_to_apply"] = callback
                target = UnifiedLLMGuardrails()

            if mode == "pre_call":
                response = await target.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=None,  # type: ignore
                    data=data,
                    call_type=call_type,  # type: ignore
                )
            elif mode == "post_call":
                response = await target.async_post_call_success_hook(
                    user_api_key_dict=user_api_key_dict,
                    data=data,
                    response=data.get("response"),  # type: ignore
                )
            else:
                return ("error", None, f"Unsupported pipeline mode: {mode}")

            # Normal return means pass
            modified_data = None
            if response is not None and isinstance(response, dict):
                modified_data = response
            return ("pass", modified_data, None)

        except Exception as e:
            if CustomGuardrail._is_guardrail_intervention(e):
                error_msg = _extract_error_message(e)
                return ("fail", None, error_msg)
            else:
                verbose_proxy_logger.error(
                    f"Pipeline: unexpected error from guardrail '{step.guardrail}': {e}"
                )
                return ("error", None, str(e))

    @staticmethod
    def _find_guardrail_callback(guardrail_name: str) -> Optional[CustomGuardrail]:
        """Look up an initialized guardrail callback by name from litellm.callbacks."""
        for callback in litellm.callbacks:
            if isinstance(callback, CustomGuardrail):
                if callback.guardrail_name == guardrail_name:
                    return callback
        return None


def _extract_error_message(e: Exception) -> str:
    """Extract a human-readable error message from a guardrail exception."""
    if isinstance(e, ModifyResponseException):
        return str(e)
    if HTTPException is not None and isinstance(e, HTTPException):
        detail = getattr(e, "detail", None)
        if detail:
            return str(detail)
    return str(e)
