"""Alice WonderFence guardrail integration for LiteLLM."""

import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Literal, Optional, Type, Union

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

from .client_cache import get_or_create_client, load_sdk
from .credentials import resolve_credentials
from .exceptions import WonderFenceBlockedError, WonderFenceMissingSecrets
from .processing import build_analysis_context, extract_relevant_text, handle_action

if TYPE_CHECKING:
    from wonderfence_sdk.client import (  # type: ignore[import-untyped]
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
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_timeout: float = 10.0,
        platform: Optional[str] = None,
        fail_open: bool = False,
        block_message: str = "Content violates our policies and has been blocked",
        debug: bool = False,
        max_cached_clients: Optional[int] = None,
        connection_pool_limit: Optional[int] = None,
        allow_request_metadata_override: bool = False,
        event_hook: Optional[
            Union[GuardrailEventHooks, List[GuardrailEventHooks], Mode]
        ] = None,
        default_on: bool = True,
        **kwargs,
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

        if debug:
            logger.setLevel(logging.DEBUG)

        self._client_cache: "OrderedDict[str, _WonderFenceV2Client]" = OrderedDict()
        self._client_cache_maxsize = max_cached_clients or int(
            os.environ.get("ALICE_MAX_CACHED_CLIENTS", "10")
        )
        env_pool = os.environ.get("ALICE_CONNECTION_POOL_LIMIT")
        self._connection_pool_limit: Optional[int] = (
            connection_pool_limit
            if connection_pool_limit is not None
            else (int(env_pool) if env_pool else None)
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
            self._WonderFenceV2Client,
            self.api_timeout,
            self.api_base,
            self.platform,
            self._connection_pool_limit,
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """Apply WonderFence guardrail using V2 client + per-request app_id."""
        text, text_source = extract_relevant_text(inputs, input_type)
        if not text:
            logger.debug(
                "Alice WonderFence (apply_guardrail): no relevant text for %s",
                input_type,
            )
            return inputs

        try:
            api_key, app_id = resolve_credentials(
                request_data,
                input_type,
                logging_obj,
                self.guardrail_name,
                self.api_key,
                self.allow_request_metadata_override,
            )
            client = await self._get_client(api_key)
            context = build_analysis_context(
                request_data, self.platform, self._AnalysisContext
            )

            if input_type == "request":
                logger.debug(
                    "Alice WonderFence (apply_guardrail): evaluating prompt app_id=%s guardrail=%s",
                    app_id,
                    self.guardrail_name,
                )
                result = await client.evaluate_prompt(
                    app_id=app_id,
                    prompt=text,
                    context=context,
                    custom_fields=None,
                )
            else:
                logger.debug(
                    "Alice WonderFence (apply_guardrail): evaluating response app_id=%s guardrail=%s",
                    app_id,
                    self.guardrail_name,
                )
                result = await client.evaluate_response(
                    app_id=app_id,
                    response=text,
                    context=context,
                    custom_fields=None,
                )

            handle_action(
                result, inputs, text_source, self.guardrail_name, self.block_message
            )

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

        add_guardrail_to_applied_guardrails_header(
            request_data=request_data, guardrail_name=self.guardrail_name
        )
        return inputs

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        """Return the config model for UI rendering."""
        return WonderFenceGuardrailConfigModel
