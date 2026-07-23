"""Alice WonderFence guardrail integration for LiteLLM."""

import logging
import os
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Optional

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.alice_wonderfence import (
    WonderFenceGuardrailConfigModel,
)
from litellm.types.utils import GenericGuardrailAPIInputs

from .chunked_evaluation import (
    DEFAULT_MAX_CONCURRENCY,
    evaluate_segments,
)
from .client_cache import ClientBuildSpec, get_or_create_client, load_sdk
from .credentials import CredentialConfig, resolve_credentials
from .exceptions import (
    WonderFenceBlockedError,
    WonderFenceMissingSecrets,
    WonderFenceScanBudgetExceeded,
)
from .processing import (
    JOINER,
    apply_response_verdicts,
    block_detail,
    build_analysis_context,
    check_scan_budget,
    function_definition_segments,
    raise_if_blocked,
    reconstruct,
    tool_call_arg_segments,
    tool_definition_segments,
)

if TYPE_CHECKING:
    from wonderfence_sdk.client import (  # pyright: ignore[reportMissingTypeStubs]  # wonderfence_sdk ships no type stubs
        WonderFenceV2Client as _WonderFenceV2Client,
    )

    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )


logger = verbose_proxy_logger.getChild("alice_wonderfence")


class WonderFenceGuardrail(CustomGuardrail):
    """Alice WonderFence guardrail handler using the V2 SDK client.

    ``api_key`` and ``app_id`` are resolved per request from API-key metadata,
    team metadata, optionally request metadata, with ``api_key`` falling back
    to a configured default. ``app_id`` has no default. See ``credentials``
    module for the full precedence rationale.

    A V2 SDK client is cached per resolved ``api_key`` (LRU).
    """

    def __init__(
        self,
        guardrail_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        api_timeout: float = 10.0,
        platform: str | None = None,
        fail_open: bool = False,
        block_message: str = "Content violates our policies and has been blocked",
        debug: bool = False,
        max_cached_clients: int | None = None,
        connection_pool_limit: int | None = None,
        allow_request_metadata_override: bool = False,
        max_scan_chars: int | None = None,
        max_scan_segments: int | None = None,
        event_hook: (GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None) = None,
        default_on: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Alice WonderFence guardrail.

        Args:
            guardrail_name: Unique identifier for this guardrail instance.
            api_key: Default WonderFence API key. Falls back to ``ALICE_API_KEY``.
            api_base: Optional base URL override for the WonderFence API.
            api_timeout: Per-call timeout in seconds (rounded to int for SDK).
            platform: Cloud platform identifier (e.g., aws, azure, databricks).
            fail_open: When True, allow requests/responses through if WonderFence
                is unreachable. BLOCK actions are always enforced.
            block_message: User-facing error message returned on BLOCK action.
            debug: Set guardrail logger to DEBUG level.
            max_cached_clients: Max SDK clients cached per guardrail (LRU,
                keyed by api_key). Default 10. Env: ALICE_MAX_CACHED_CLIENTS.
            connection_pool_limit: Max connections per SDK client HTTP pool.
                Env: ALICE_CONNECTION_POOL_LIMIT.
            allow_request_metadata_override: When True, allow per-request
                ``metadata.alice_wonderfence_api_key`` /
                ``metadata.alice_wonderfence_app_id`` as a last-resort source
                (after API-key and team metadata). Defaults to False so
                caller-controlled fields cannot bypass admin-pinned credentials.
            max_scan_chars: Fail-closed total-work cap on combined scan
                characters per request/response. Default 1_000_000. Env:
                ALICE_MAX_SCAN_CHARS.
            max_scan_segments: Fail-closed total-work cap on scan segment count
                per request/response. Default 1_000. Env:
                ALICE_MAX_SCAN_SEGMENTS.
            event_hook: Event hook mode.
            default_on: Whether the guardrail is enabled by default.
        """
        WonderFenceV2Client, AnalysisContext = load_sdk()
        self._WonderFenceV2Client = WonderFenceV2Client
        self._AnalysisContext = AnalysisContext

        self.api_key = api_key or os.environ.get("ALICE_API_KEY")
        self.api_base = api_base
        self.api_timeout = api_timeout
        self.platform = platform
        self.fail_open = fail_open
        self.block_message = block_message
        self.allow_request_metadata_override = allow_request_metadata_override
        env_max_chars = os.environ.get("ALICE_MAX_SCAN_CHARS")
        self.max_scan_chars: int | None = (
            max_scan_chars if max_scan_chars is not None else (int(env_max_chars) if env_max_chars else 1_000_000)
        )
        env_max_segments = os.environ.get("ALICE_MAX_SCAN_SEGMENTS")
        self.max_scan_segments: int | None = (
            max_scan_segments
            if max_scan_segments is not None
            else (int(env_max_segments) if env_max_segments else 1_000)
        )

        if debug:
            logger.setLevel(logging.DEBUG)

        self._client_cache: OrderedDict[str, _WonderFenceV2Client] = OrderedDict()
        self._client_cache_maxsize = max_cached_clients or int(os.environ.get("ALICE_MAX_CACHED_CLIENTS", "10"))
        env_pool = os.environ.get("ALICE_CONNECTION_POOL_LIMIT")
        self._connection_pool_limit: int | None = (
            connection_pool_limit if connection_pool_limit is not None else (int(env_pool) if env_pool else None)
        )

        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

        super().__init__(
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )
        # Narrow attribute type: base class declares Optional[str], but our
        # __init__ requires a non-empty string and the factory rejects empty.
        self.guardrail_name: str = guardrail_name

        key_suffix = f"***{self.api_key[-4:]}" if self.api_key else "<unset>"
        logger.debug(
            "Alice WonderFence guardrail initialized: name=%s default_api_key=%s",
            guardrail_name,
            key_suffix,
        )

    async def _get_client(self, api_key: str) -> "_WonderFenceV2Client":
        """Return a cached WonderFenceV2Client for the given api_key (LRU)."""
        return get_or_create_client(
            api_key,
            self._client_cache,
            self._client_cache_maxsize,
            ClientBuildSpec(
                client_class=self._WonderFenceV2Client,
                api_timeout=self.api_timeout,
                api_base=self.api_base,
                platform=self.platform,
                connection_pool_limit=self._connection_pool_limit,
            ),
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """Apply WonderFence guardrail using V2 client + per-request app_id.

        Request side joins all scan pieces (message text plus detection-only
        tool-call args and tool/function descriptions) into one document and
        scans it in ~1 call (chunked only when it exceeds the size limit), so
        call volume scales with total size rather than message count. Response
        side stays per-segment (independent choices / model tool-call args)
        because the handler's response write-back is purely positional and has
        no ``structured_messages`` path.
        """
        texts = inputs.get("texts") or []
        tool_indices, tool_arg_segments = tool_call_arg_segments(inputs)
        tool_def_texts = tool_definition_segments(inputs)
        # Legacy top-level functions[] only exist on the request body; the
        # translation layer does not surface them in inputs, so read request_data.
        function_def_texts = function_definition_segments(request_data) if input_type == "request" else []

        if input_type == "request":
            scan_pieces = [*texts, *tool_arg_segments, *tool_def_texts, *function_def_texts]
        else:
            scan_pieces = [*texts, *tool_arg_segments]

        if not scan_pieces:
            logger.debug(
                "Alice WonderFence (apply_guardrail): nothing to scan for %s",
                input_type,
            )
            return inputs

        try:
            check_scan_budget(scan_pieces, self.max_scan_chars, self.max_scan_segments)
            api_key, app_id = resolve_credentials(
                request_data,
                input_type,
                logging_obj,
                CredentialConfig(
                    guardrail_name=self.guardrail_name,
                    default_api_key=self.api_key,
                    allow_request_metadata_override=self.allow_request_metadata_override,
                ),
            )
            client = await self._get_client(api_key)
            context = build_analysis_context(request_data, self.platform, self._AnalysisContext)
            max_concurrency = self._connection_pool_limit or DEFAULT_MAX_CONCURRENCY

            if input_type == "request":

                async def evaluate(text: str) -> object:
                    return await client.evaluate_prompt(app_id=app_id, prompt=text, context=context, custom_fields=None)

                await self._scan_request(inputs, scan_pieces, len(texts), evaluate, max_concurrency, app_id)
            else:

                async def evaluate(text: str) -> object:
                    return await client.evaluate_response(
                        app_id=app_id,
                        response=text,
                        context=context,
                        custom_fields=None,
                    )

                await self._scan_response(inputs, texts, tool_indices, tool_arg_segments, evaluate, max_concurrency)

        except WonderFenceScanBudgetExceeded as e:
            # Fail-closed config/abuse guard: reject before any provider call and
            # never fall through to the fail_open path below.
            raise HTTPException(status_code=400, detail=e.detail)
        except WonderFenceBlockedError as e:
            raise HTTPException(status_code=400, detail=e.detail)
        except WonderFenceMissingSecrets as e:
            # Configuration errors (no api_key / app_id resolvable) are never
            # fail-open: a misconfigured tenant must not silently bypass the
            # guardrail.
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Error in Alice WonderFence Guardrail",
                    "guardrail_name": self.guardrail_name,
                    "exception": str(e),
                },
            ) from e
        except Exception as e:
            if self.fail_open:
                # Log only — do not add to the applied-guardrails header. The
                # header lists configured guardrail_names verbatim; consumers
                # rely on its membership to decide whether scanning ran. A
                # synthetic suffix (e.g. ":unscanned") would silently pass the
                # membership check and mask audit gaps.
                logger.error(
                    "Alice WonderFence unreachable; fail-open enabled, proceeding "
                    "without guardrail. guardrail_name=%s input_type=%s "
                    "guardrail_status=unscanned_fail_open error=%s",
                    self.guardrail_name,
                    input_type,
                    str(e),
                    exc_info=e,
                )
                return inputs
            logger.error(
                "Alice WonderFence unreachable; fail-open disabled, blocking "
                "request. guardrail_name=%s input_type=%s error=%s",
                self.guardrail_name,
                input_type,
                str(e),
                exc_info=e,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Error in Alice WonderFence Guardrail",
                    "guardrail_name": self.guardrail_name,
                    "exception": str(e),
                },
            ) from e

        add_guardrail_to_applied_guardrails_header(request_data=request_data, guardrail_name=self.guardrail_name)
        return inputs

    async def _scan_request(
        self,
        inputs: GenericGuardrailAPIInputs,
        pieces: list[str],
        n_text: int,
        evaluate: "Callable[[str], Awaitable[object]]",
        max_concurrency: int,
        app_id: str,
    ) -> None:
        """Scan the joined request document and write MASK back to ``texts``.

        The first ``n_text`` pieces are maskable message-text parts; the rest are
        detection-only tool-call args and tool/function descriptions. On MASK we
        reconstruct per-part masked text by aligning the join against the masked
        document and write the recovered message-text parts back to
        ``inputs["texts"]`` (positional write-back / "Path B"): the handler
        already maps that list onto the right message parts, so there is no need
        to rebuild ``structured_messages``. Reconstruction failure fails closed
        (block) rather than misassigning a redaction.
        """
        document = JOINER.join(pieces)
        logger.debug(
            "Alice WonderFence (apply_guardrail request): scanning joined document of %d piece(s) "
            "(%d text + %d detection-only), %d chars, guardrail=%s app_id=%s",
            len(pieces),
            n_text,
            len(pieces) - n_text,
            len(document),
            self.guardrail_name,
            app_id,
        )
        verdict = (await evaluate_segments([document], evaluate, max_concurrency=max_concurrency))[0]
        raise_if_blocked([verdict], self.guardrail_name, self.block_message)

        correlation_id = verdict.correlation_ids[0] if verdict.correlation_ids else None
        if verdict.action == "MASK":
            recovered = reconstruct(pieces, verdict.masked_text or "")
            if recovered is None:
                logger.warning(
                    "Alice WonderFence (apply_guardrail request): MASK reconstruction failed "
                    "(a joiner or part boundary landed inside a masked span); failing closed. guardrail=%s correlation_id=%s",
                    self.guardrail_name,
                    correlation_id,
                )
                raise WonderFenceBlockedError(block_detail([verdict], self.guardrail_name, self.block_message))
            inputs["texts"] = recovered[:n_text]
            logger.info(
                "Alice WonderFence (apply_guardrail request): MASK applied to request text guardrail=%s correlation_id=%s",
                self.guardrail_name,
                correlation_id,
            )
        elif verdict.action == "DETECT":
            logger.warning(
                "Alice WonderFence (apply_guardrail request): DETECT on joined document guardrail=%s correlation_id=%s",
                self.guardrail_name,
                correlation_id,
            )

    async def _scan_response(
        self,
        inputs: GenericGuardrailAPIInputs,
        texts: list[str],
        tool_indices: list[int],
        tool_arg_segments: list[str],
        evaluate: "Callable[[str], Awaitable[object]]",
        max_concurrency: int,
    ) -> None:
        """Scan response segments per-index and write masks back in place."""
        segments = [*texts, *tool_arg_segments]
        logger.debug(
            "Alice WonderFence (apply_guardrail response): evaluating %d text + %d tool-call segment(s) guardrail=%s",
            len(texts),
            len(tool_arg_segments),
            self.guardrail_name,
        )
        verdicts = await evaluate_segments(segments, evaluate, max_concurrency=max_concurrency)
        n_text = len(texts)
        apply_response_verdicts(
            inputs,
            verdicts[:n_text],
            tool_indices,
            verdicts[n_text:],
            self.guardrail_name,
            self.block_message,
        )

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        """Return the config model for UI rendering."""
        return WonderFenceGuardrailConfigModel
