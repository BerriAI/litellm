"""
Pipeline Executor - Executes guardrail pipelines with conditional step logic.

Runs guardrails sequentially per pipeline step definitions, handling
pass/fail actions (allow, block, next, modify_response) and data forwarding.
"""

import copy
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import LOGS_GUARDRAIL_INFORMATION_MARKER
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    get_metadata_variable_name_from_kwargs,
    get_or_create_metadata_bucket,
    independent_snapshot,
)
from litellm.proxy.common_utils.callback_utils import add_guardrail_to_applied_guardrails_header
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.proxy.policy_engine.pipeline_types import (
    PipelineExecutionResult,
    PipelineStep,
    PipelineStepResult,
)
from litellm.types.utils import GenericGuardrailAPIInputs, StandardLoggingGuardrailInformation

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.guardrail_translation.base_translation import (
        BaseTranslation,
    )
    from litellm.proxy._types import UserAPIKeyAuth

try:
    from fastapi.exceptions import HTTPException
except ImportError:
    HTTPException = None


class UndeliverableStreamRewrite(Exception):
    def __init__(self, guardrail_name: str, reason: str) -> None:
        super().__init__(
            f"Guardrail '{guardrail_name}' rewrote the streamed response but the rewrite cannot be written "
            f"back to the stream: {reason}"
        )
        self.guardrail_name: Final = guardrail_name
        self.reason: Final = reason


class UnappliableRequestRewrite(Exception):
    def __init__(self, guardrail_name: str) -> None:
        super().__init__(
            f"Guardrail '{guardrail_name}' rewrote the request in a way this endpoint cannot apply, "
            "so the request was rejected rather than sent unrewritten"
        )
        self.guardrail_name: Final = guardrail_name


def _tool_call_shape(tool_call: object) -> tuple[object, object]:
    plain: Final = tool_call.model_dump() if isinstance(tool_call, BaseModel) else tool_call
    function: Final = plain.get("function") if isinstance(plain, Mapping) else None
    if not isinstance(function, Mapping):
        return (None, None)
    return (function.get("name"), function.get("arguments"))


def _text_snapshot(texts: Sequence[str] | None) -> tuple[str, ...] | None:
    return None if texts is None else tuple(texts)


def _scanned_texts(texts: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(texts or ())


def _tool_call_shapes(tool_calls: Sequence[object] | None) -> tuple[tuple[object, object], ...] | None:
    return None if tool_calls is None else tuple(_tool_call_shape(tool_call) for tool_call in tool_calls)


def _rewrote(sent: tuple[object, ...] | None, returned: tuple[object, ...] | None) -> bool:
    return sent is not None and returned is not None and returned != sent


def _count_change(sent: tuple[object, ...] | None, returned: tuple[object, ...] | None) -> tuple[int, int] | None:
    if sent is None or returned is None or len(returned) == len(sent):
        return None
    return (len(sent), len(returned))


def _tool_call_mismatch_reason(
    sent: tuple[tuple[object, object], ...] | None, returned: tuple[tuple[object, object], ...] | None
) -> str | None:
    if sent == returned:
        return None
    sent_count: Final = len(sent or ())
    returned_count: Final = len(returned or ())
    if sent_count == returned_count:
        return "the legacy hook changed a tool call's name or arguments, which this path cannot write back"
    return f"the legacy hook returned {returned_count} tool calls for a stream that carried {sent_count}"


_GuardrailMethodT = TypeVar("_GuardrailMethodT", bound=Callable[..., object])


def _logged_by_inner_guardrail(method: _GuardrailMethodT) -> _GuardrailMethodT:
    vars(method)[LOGS_GUARDRAIL_INFORMATION_MARKER] = True  # rebind-ok: stamps the method the class body just defined
    return method


class _StreamRewriteObserver(CustomGuardrail):
    """Stand-in handed to the endpoint translation in place of a streaming pipeline step's
    guardrail. It records whether the guardrail returned different output than it was given,
    which for guardrails like Bedrock's ANONYMIZED action is only known at runtime. Text and
    tool-call rewrites are deliverable on translations that write them back across the
    buffered chunks (``delivers_ended_stream_rewrites``); rewrites on any other translation,
    and a rewrite that drops or adds a tool call on any translation, are discarded by the
    executor, which releases the original chunks.
    The inner guardrail's ``apply_guardrail`` already records the guardrail information
    and span, so the observer's stays out of ``log_guardrail_information``."""

    def __init__(self, inner: CustomGuardrail) -> None:
        super().__init__(guardrail_name=inner.guardrail_name)
        self.inner: Final = inner
        self.rewrote_texts = False
        self.rewrote_tool_calls = False
        self.tool_call_count_change: tuple[int, int] | None = None

    def structured_messages_cover_full_request(self) -> bool:
        return self.inner.structured_messages_cover_full_request()

    @_logged_by_inner_guardrail
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,  # mutable-ok: matches CustomGuardrail.apply_guardrail
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        sent_texts: Final = _text_snapshot(inputs.get("texts"))
        sent_tool_shapes: Final = _tool_call_shapes(inputs.get("tool_calls"))
        outputs: Final = await self.inner.apply_guardrail(
            inputs=inputs, request_data=request_data, input_type=input_type, logging_obj=logging_obj
        )
        returned_tool_shapes: Final = _tool_call_shapes(outputs.get("tool_calls"))
        self.rewrote_texts = self.rewrote_texts or _rewrote(sent_texts, _text_snapshot(outputs.get("texts")))
        self.rewrote_tool_calls = self.rewrote_tool_calls or _rewrote(sent_tool_shapes, returned_tool_shapes)
        self.tool_call_count_change = self.tool_call_count_change or _count_change(
            sent_tool_shapes, returned_tool_shapes
        )
        return outputs

    def discard_reason(self, deliver_rewrites: bool) -> str | None:
        if self.tool_call_count_change is not None:
            sent, returned = self.tool_call_count_change
            return (
                f"the guardrail returned {returned} tool calls for a stream that carried {sent}, and a rewrite "
                "that drops or adds a tool call cannot be written back"
            )
        if not deliver_rewrites and (self.rewrote_texts or self.rewrote_tool_calls):
            return "this endpoint's streaming pipeline does not write ended-stream rewrites back yet"
        return None


class _ScannedTextRecorder(CustomGuardrail):
    def __init__(self, guardrail_name: str) -> None:
        super().__init__(guardrail_name=guardrail_name)
        self.inputs: GenericGuardrailAPIInputs | None = None

    @_logged_by_inner_guardrail
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,  # mutable-ok: matches CustomGuardrail.apply_guardrail
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        self.inputs = inputs
        return inputs


class _LegacyHookStreamAdapter(CustomGuardrail):
    """Runs a guardrail that only implements the legacy post-call hook (no unified
    ``apply_guardrail``, or ``use_native_lifecycle_hooks``) as a streaming pipeline step. The
    endpoint translation hands it the texts it scanned plus the assembled response under
    ``request_data["response"]``; the hook gets that response in the shape its route gives
    non-streaming hooks, an exception it raises ends the stream through the executor's
    fail/error classification, and the response it hands back, or the one it changed in place
    and returned ``None`` for, is re-scanned by the same translation so its texts reach the
    client through the translation's ended-stream write-back. A
    replacement whose scanned texts do not line up with the originals, or whose tool calls
    differ from them, is undeliverable, so the executor releases the original chunks. A stream
    that carried no text to scan, such as a tool-only Anthropic message, stays deliverable as
    long as the hook left the tool calls alone."""

    def __init__(
        self,
        inner: CustomGuardrail,
        endpoint_translation: "BaseTranslation",
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> None:
        super().__init__(guardrail_name=inner.guardrail_name)
        self.inner: Final = inner
        self.endpoint_translation: Final = endpoint_translation
        self.user_api_key_dict: Final = user_api_key_dict

    def structured_messages_cover_full_request(self) -> bool:
        return self.inner.structured_messages_cover_full_request()

    @_logged_by_inner_guardrail
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,  # mutable-ok: matches CustomGuardrail.apply_guardrail
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        hooked: Final = self.endpoint_translation.post_call_hook_response(request_data.get("response"))
        replacement: Final = await self.inner.async_post_call_success_hook(
            data=request_data,
            user_api_key_dict=self.user_api_key_dict,
            response=hooked,
        )
        rewrite: Final = hooked if replacement is None else replacement
        if rewrite is None:
            return inputs
        rescanned: Final = await self._rescan(rewrite, logging_obj)
        guardrail_name: Final = self.guardrail_name or "unknown"
        if rescanned is None:
            raise UndeliverableStreamRewrite(
                guardrail_name, "the legacy hook's response could not be rescanned by this endpoint's translation"
            )
        rewritten: Final = rescanned.get("texts")
        returned_text_count: Final = len(_scanned_texts(rewritten))
        sent_text_count: Final = len(_scanned_texts(inputs.get("texts")))
        if returned_text_count != sent_text_count:
            raise UndeliverableStreamRewrite(
                guardrail_name,
                f"the legacy hook returned {returned_text_count} texts for a stream that carried {sent_text_count}",
            )
        tool_call_mismatch: Final = _tool_call_mismatch_reason(
            _tool_call_shapes(inputs.get("tool_calls")), _tool_call_shapes(rescanned.get("tool_calls"))
        )
        if tool_call_mismatch is not None:
            raise UndeliverableStreamRewrite(guardrail_name, tool_call_mismatch)
        if not rewritten:
            return inputs
        rewritten_inputs: Final[GenericGuardrailAPIInputs] = {**inputs, "texts": rewritten}
        return rewritten_inputs

    async def _rescan(
        self, response: object, logging_obj: "LiteLLMLoggingObj | None"
    ) -> GenericGuardrailAPIInputs | None:
        recorder: Final = _ScannedTextRecorder(self.guardrail_name or "unknown")
        await self.endpoint_translation.process_output_response(
            response=response,
            guardrail_to_apply=recorder,
            litellm_logging_obj=logging_obj,
            user_api_key_dict=self.user_api_key_dict,
        )
        return recorder.inputs


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


def _release_original_chunks(
    guardrail_name: str,
    reason: str,
    streaming_chunks: list[object],  # mutable-ok: shared buffered-stream chunks, restored in place
    originals: Sequence[object],
) -> None:
    streaming_chunks[:] = originals  # rebind-ok: the caller's buffer is the stream the client receives
    verbose_proxy_logger.warning(
        "Pipeline: guardrail '%s' rewrote the streamed response but the rewrite could not be written back to "
        "the stream: %s. The whole rewrite, text rewrites included, was discarded and the original stream released",
        guardrail_name,
        reason,
    )


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
                return _allow_result(step_results=step_results, working_data=working_data, request_data=data)

            if action == "block":
                _carry_working_guardrail_information(working_data=working_data, request_data=data)
                return PipelineExecutionResult(
                    terminal_action="block",
                    step_results=step_results,
                    error_message=error_detail,
                    original_exception=original_exception,
                    modified_data=working_data if working_data != data else None,
                )

            if action == "modify_response":
                _carry_working_guardrail_information(working_data=working_data, request_data=data)
                return PipelineExecutionResult(
                    terminal_action="modify_response",
                    step_results=step_results,
                    modify_response_message=step.modify_response_message or error_detail,
                    modified_data=working_data if working_data != data else None,
                )

            # action == "next" → continue to next step

        # Ran out of steps without a terminal action → default allow
        return _allow_result(step_results=step_results, working_data=working_data, request_data=data)

    @staticmethod
    async def _run_streaming_step(
        step: PipelineStep,
        callback: CustomGuardrail,
        endpoint_translation: "BaseTranslation",
        streaming_chunks: list[object],  # mutable-ok: shared buffered-stream chunks the translation rewrites in place
        hook_input: dict[str, object],  # mutable-ok: same request-payload shape as data
        user_api_key_dict: "UserAPIKeyAuth",
        litellm_logging_obj: "LiteLLMLoggingObj | None",
    ) -> None:
        """Run one streaming post_call step through the endpoint translation, delivering
        text and tool-call rewrites on translations that support ended-stream write-back. A
        guardrail without the unified interface runs its legacy post-call hook against the
        assembled response through ``_LegacyHookStreamAdapter``. A rewrite that cannot reach the
        client yet (one on a translation without write-back, one that drops or adds a tool call,
        or one the translation or adapter refused with ``UndeliverableStreamRewrite``) is
        discarded: the buffered chunks go back to the originals and the step passes, so the
        client gets the stream the merge base sent, and the guardrail stays out of the
        applied-guardrails header since its output never reached the client. The response an
        earlier step's translation stored under ``request_data["response"]`` is dropped first,
        so this step's hook sees the stream as the steps before it left it."""
        scanner: Final = (
            callback
            if PipelineExecutor.supports_unified_execution(callback)
            else _LegacyHookStreamAdapter(callback, endpoint_translation, user_api_key_dict)
        )
        observer: Final = _StreamRewriteObserver(scanner)
        deliver_rewrites: Final = type(endpoint_translation).delivers_ended_stream_rewrites
        originals: Final = copy.deepcopy(streaming_chunks)
        hook_input.pop("response", None)  # rebind-ok: an earlier step's stored response goes so this step's is stored
        try:
            if deliver_rewrites:
                await endpoint_translation.process_output_streaming_response(
                    responses_so_far=streaming_chunks,
                    guardrail_to_apply=observer,
                    litellm_logging_obj=litellm_logging_obj,
                    user_api_key_dict=user_api_key_dict,
                    request_data=hook_input,
                    deliver_ended_stream_rewrites=True,
                )
            else:
                await endpoint_translation.process_output_streaming_response(
                    responses_so_far=streaming_chunks,
                    guardrail_to_apply=observer,
                    litellm_logging_obj=litellm_logging_obj,
                    user_api_key_dict=user_api_key_dict,
                    request_data=hook_input,
                )
        except UndeliverableStreamRewrite as undeliverable:
            _release_original_chunks(step.guardrail, undeliverable.reason, streaming_chunks, originals)
            return
        discard_reason: Final = observer.discard_reason(deliver_rewrites)
        if discard_reason is not None:
            _release_original_chunks(step.guardrail, discard_reason, streaming_chunks, originals)
            return
        if not callback.records_own_guardrail_information:
            add_guardrail_to_applied_guardrails_header(request_data=hook_input, guardrail_name=step.guardrail)

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

        hook_input, scans_raw_request = _prepare_hook_input(step, callback, data, raw_request_snapshot)
        snapshot_entries_before: Final = len(_recorded_guardrail_information(hook_input))

        # Use unified_guardrail path if callback implements apply_guardrail
        target: CustomLogger = callback
        use_unified: Final = PipelineExecutor.supports_unified_execution(callback)
        if use_unified and streaming_chunks is None:
            hook_input["guardrail_to_apply"] = callback
            target = UnifiedLLMGuardrails()

        try:
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
                if endpoint_translation is None:
                    return (
                        "error",
                        None,
                        f"Guardrail '{step.guardrail}' cannot run on a stream without an endpoint translation",
                        None,
                    )
                await PipelineExecutor._run_streaming_step(
                    step=step,
                    callback=callback,
                    endpoint_translation=endpoint_translation,
                    streaming_chunks=streaming_chunks,
                    hook_input=hook_input,
                    user_api_key_dict=user_api_key_dict,
                    litellm_logging_obj=data.get("litellm_logging_obj"),
                )
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

        except Exception as e:
            if CustomGuardrail._is_guardrail_intervention(e):
                error_msg: Final = _extract_error_message(e)
                return ("fail", None, error_msg, e)
            else:
                verbose_proxy_logger.error("Pipeline: unexpected error from guardrail '%s': %s", step.guardrail, e)
                return ("error", None, str(e), e)
        finally:
            if hook_input is not data:
                _append_guardrail_information(
                    request_data=data,
                    entries=_recorded_guardrail_information(hook_input)[snapshot_entries_before:],
                )

    @staticmethod
    def supports_unified_execution(callback: CustomGuardrail) -> bool:
        """Whether this guardrail runs through the unified apply_guardrail path."""
        return "apply_guardrail" in type(callback).__dict__ and not callback.use_native_lifecycle_hooks

    @staticmethod
    def supports_streaming_execution(callback: CustomGuardrail) -> bool:
        """Whether a streaming pipeline step can run this guardrail against the buffered
        stream: through the unified path, or through its post-call hook on the assembled
        response when that hook is its only streaming path. A guardrail with its own
        streaming iterator hook, or with neither hook, keeps running on its own."""
        callback_type: Final = type(callback)
        return PipelineExecutor.supports_unified_execution(callback) or (
            callback_type.async_post_call_success_hook is not CustomLogger.async_post_call_success_hook
            and callback_type.async_post_call_streaming_iterator_hook
            is CustomLogger.async_post_call_streaming_iterator_hook
        )

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


_GUARDRAIL_INFORMATION_KEY: Final = "standard_logging_guardrail_information"


def _recorded_guardrail_information(source: Mapping[str, object]) -> list[StandardLoggingGuardrailInformation]:
    bucket: Final = source.get(get_metadata_variable_name_from_kwargs(source))
    recorded: Final = bucket.get(_GUARDRAIL_INFORMATION_KEY) if isinstance(bucket, dict) else None
    return recorded if isinstance(recorded, list) else []


def _append_guardrail_information(
    request_data: dict[str, object],  # mutable-ok: same request-payload shape as execute_steps' data
    entries: Sequence[StandardLoggingGuardrailInformation],
) -> None:
    if not entries:
        return
    _, request_bucket = get_or_create_metadata_bucket(request_data)
    existing: Final = request_bucket.get(_GUARDRAIL_INFORMATION_KEY)
    if isinstance(existing, list):
        existing.extend(entries)
        return
    request_bucket[_GUARDRAIL_INFORMATION_KEY] = list(entries)


def _carry_working_guardrail_information(
    working_data: Mapping[str, object],
    request_data: dict[str, object],  # mutable-ok: same request-payload shape as execute_steps' data
) -> None:
    recorded: Final = _recorded_guardrail_information(working_data)
    existing: Final = _recorded_guardrail_information(request_data)
    if recorded is existing:
        return
    _append_guardrail_information(request_data=request_data, entries=[e for e in recorded if e not in existing])


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
