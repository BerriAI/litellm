"""Alice WonderFence guardrail integration for LiteLLM."""

import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Type, Union

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_last_user_message,
)
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.alice_wonderfence import (
    WonderFenceGuardrailConfigModel,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from wonderfence_sdk.client import (  # type: ignore[import-untyped]
        WonderFenceV2Client as _WonderFenceV2Client,
    )

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


logger = verbose_proxy_logger.getChild("alice_wonderfence")


# Key used to stash per-request resolved (api_key, app_id) on
# logging_obj.model_call_details so post_call can recover it. See
# _stash_resolved for the full rationale.
_LOGGING_OBJ_STASH_KEY = "alice_wonderfence_resolved"


class WonderFenceMissingSecrets(Exception):
    """Raised when Alice API key cannot be resolved from any source."""


class WonderFenceBlockedError(Exception):
    """Raised when WonderFence blocks a request/response."""

    def __init__(self, detail: dict):
        self.detail = detail
        super().__init__(detail.get("error", "Blocked by Alice WonderFence guardrail"))


class WonderFenceGuardrail(CustomGuardrail):
    """Alice WonderFence guardrail handler using the V2 SDK client.

    ``api_key`` and ``app_id`` are resolved per request from request metadata,
    API-key metadata, or team metadata. ``api_key`` falls back to a configured
    default; ``app_id`` has no default and must be supplied per request.

    Resolution order for ``api_key``:
    1. Request metadata: ``metadata.alice_wonderfence_api_key``
    2. API key metadata: ``user_api_key_metadata.alice_wonderfence_api_key``
    3. Team metadata: ``user_api_key_team_metadata.alice_wonderfence_api_key``
    4. Default: configured ``api_key`` or ``ALICE_API_KEY`` env var

    Resolution order for ``app_id`` (no default — error if missing):
    1. Request metadata: ``metadata.alice_wonderfence_app_id``
    2. API key metadata: ``user_api_key_metadata.alice_wonderfence_app_id``
    3. Team metadata: ``user_api_key_team_metadata.alice_wonderfence_app_id``

    A V2 SDK client is cached per resolved ``api_key`` (LRU).
    """

    def __init__(
        self,
        guardrail_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_timeout: float = 20.0,
        platform: Optional[str] = None,
        fail_open: bool = False,
        block_message: str = "Content violates our policies and has been blocked",
        debug: bool = False,
        max_cached_clients: Optional[int] = None,
        connection_pool_limit: Optional[int] = None,
        event_hook: Optional[
            Union[GuardrailEventHooks, List[GuardrailEventHooks], Mode]
        ] = None,
        default_on: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the Alice WonderFence guardrail.

        Args:
            guardrail_name: Unique identifier for this guardrail instance.
            api_key: Default WonderFence API key. Overridable per request via
                ``metadata.alice_wonderfence_api_key``. Falls back to
                ``ALICE_API_KEY`` env var.
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
            event_hook: Event hook mode.
            default_on: Whether the guardrail is enabled by default.
        """
        try:
            import wonderfence_sdk  # type: ignore[import-untyped]  # noqa: F401  # pyright: ignore[reportUnusedImport]

            del wonderfence_sdk
        except ImportError as e:
            raise ImportError(
                "Alice WonderFence SDK not installed. Install with: pip install wonderfence-sdk"
            ) from e

        self.api_key = api_key or os.environ.get("ALICE_API_KEY")
        self.api_base = api_base
        self.api_timeout = api_timeout
        self.platform = platform
        self.fail_open = fail_open
        self.block_message = block_message

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
        from wonderfence_sdk.client import (  # type: ignore[import-untyped]
            WonderFenceV2Client,
        )

        if api_key in self._client_cache:
            self._client_cache.move_to_end(api_key)
            return self._client_cache[api_key]

        client_kwargs: dict = {
            "api_key": api_key,
            "api_timeout": round(self.api_timeout),
        }
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        if self.platform:
            client_kwargs["platform"] = self.platform
        if self._connection_pool_limit is not None:
            client_kwargs["connection_pool_limit"] = self._connection_pool_limit

        client = WonderFenceV2Client(**client_kwargs)
        self._client_cache[api_key] = client

        if len(self._client_cache) > self._client_cache_maxsize:
            _, evicted = self._client_cache.popitem(last=False)
            try:
                await evicted.close()
            except Exception:
                logger.warning(
                    "Failed to close evicted WonderFence client", exc_info=True
                )

        return client

    @staticmethod
    def _get_metadata(request_data: dict) -> dict:
        return (
            request_data.get("metadata") or request_data.get("litellm_metadata") or {}
        )

    def _resolve_api_key(self, request_data: dict) -> str:
        """Resolve api_key from request → key → team metadata, falling back to default.

        The LiteLLM framework copies key/team metadata from ``UserAPIKeyAuth``
        into ``data['metadata']`` under ``user_api_key_metadata`` and
        ``user_api_key_team_metadata``, so all sources are read from
        ``request_data``.
        """
        metadata = self._get_metadata(request_data)

        req_api_key = metadata.get("alice_wonderfence_api_key")
        if req_api_key:
            return req_api_key

        key_metadata = metadata.get("user_api_key_metadata") or {}
        if isinstance(key_metadata, dict) and key_metadata.get(
            "alice_wonderfence_api_key"
        ):
            return key_metadata["alice_wonderfence_api_key"]

        team_metadata = metadata.get("user_api_key_team_metadata") or {}
        if isinstance(team_metadata, dict) and team_metadata.get(
            "alice_wonderfence_api_key"
        ):
            return team_metadata["alice_wonderfence_api_key"]

        if self.api_key:
            return self.api_key

        raise WonderFenceMissingSecrets(
            "No alice_wonderfence_api_key found in request metadata, API-key "
            "metadata, team metadata, or default config (ALICE_API_KEY)."
        )

    def _resolve_app_id(self, request_data: dict) -> str:
        """Resolve app_id from request → key → team metadata. No default — raise if missing."""
        metadata = self._get_metadata(request_data)

        req_app_id = metadata.get("alice_wonderfence_app_id")
        if req_app_id:
            return req_app_id

        key_metadata = metadata.get("user_api_key_metadata") or {}
        if isinstance(key_metadata, dict) and key_metadata.get(
            "alice_wonderfence_app_id"
        ):
            return key_metadata["alice_wonderfence_app_id"]

        team_metadata = metadata.get("user_api_key_team_metadata") or {}
        if isinstance(team_metadata, dict) and team_metadata.get(
            "alice_wonderfence_app_id"
        ):
            return team_metadata["alice_wonderfence_app_id"]

        raise ValueError(
            "No alice_wonderfence_app_id found in request metadata, API-key "
            "metadata, or team metadata. app_id must be provided per request."
        )

    def _build_analysis_context(self, request_data: dict) -> Any:
        """Build WonderFence AnalysisContext from request data."""
        from wonderfence_sdk.models import (  # type: ignore[import-untyped]
            AnalysisContext,
        )

        metadata = self._get_metadata(request_data)
        model_str = request_data.get("model", "")

        provider = None
        model_name = model_str
        if model_str:
            try:
                model_name, provider, _, _ = litellm.get_llm_provider(model=model_str)
            except Exception:
                if "/" in model_str:
                    provider, model_name = model_str.split("/", 1)

        user_id = (
            metadata.get("user_api_key_end_user_id")
            or metadata.get("end_user_id")
            or metadata.get("user_id")
        )

        session_id = (
            request_data.get("litellm_session_id")
            or metadata.get("litellm_session_id")
            or metadata.get("session_id")
        )

        return AnalysisContext(
            session_id=session_id,
            user_id=user_id,
            model_name=model_name,
            provider=provider,
            platform=self.platform,
        )

    def _stash_resolved(
        self,
        logging_obj: Optional["LiteLLMLoggingObj"],
        api_key: str,
        app_id: str,
    ) -> None:
        """Persist resolved (api_key, app_id) on the request-scoped logging_obj
        so post_call can recover it.

        Why we need this:
            LiteLLM's per-provider chat translation handler synthesizes a
            fresh `request_data` for post_call (`process_output_response`,
            e.g. `litellm/llms/openai/chat/guardrail_translation/handler.py:312`).
            That dict only carries `litellm_metadata.user_api_key_metadata`
            and `user_api_key_team_metadata` — the original request body's
            `metadata` field (where per-request `alice_wonderfence_app_id`
            lives) is dropped. Without a bridge, post_call resolution fails
            even though the request explicitly supplied the value.

        Why logging_obj.model_call_details (and not a ContextVar):
            during_call hooks run via `asyncio.gather` in
            `litellm/proxy/utils.py:1500`, which wraps each coroutine in
            its own asyncio Task with a *copied* context. ContextVar
            writes in a child Task are not visible to the parent Task that
            runs post_call, so a ContextVar bridge silently fails.
            `logging_obj` is passed through every hook by reference (same
            object across pre_call, during_call, and post_call), so
            mutations to its `model_call_details` dict are visible
            regardless of task boundary.

        Why this isn't a layering hack:
            Despite the name, `model_call_details` is used throughout
            LiteLLM as a generic request-scoped state bag (see
            `main.py:6444`, `proxy/utils.py:1885-1895`, every passthrough
            handler under `proxy/pass_through_endpoints/`). It stores
            things like `model`, `custom_llm_provider`, `response_cost`,
            `messages`, `client`, `litellm_call_id` — well beyond log
            payload material.

        Keyed by guardrail_name so multiple alice_wonderfence instances
        configured on the same proxy don't collide.
        """
        if logging_obj is None:
            return
        container: Dict[str, Tuple[str, str]] = (
            logging_obj.model_call_details.setdefault(_LOGGING_OBJ_STASH_KEY, {})
        )
        container[self.guardrail_name] = (api_key, app_id)

    def _recover_resolved(
        self, logging_obj: Optional["LiteLLMLoggingObj"]
    ) -> Optional[Tuple[str, str]]:
        """Look up (api_key, app_id) stashed earlier in this request.

        Prefer this instance's own stash. If absent, fall back to any
        sibling alice_wonderfence instance's stash on the same request.

        Why the sibling fallback exists:
            LiteLLM serializes parallel during_call hooks through a single
            shared slot `data["guardrail_to_apply"]` (proxy/utils.py:1483).
            That slot is overwritten in a loop *before* any gather() task
            runs, so only the last-registered guardrail callback actually
            executes its during_call — the others see `None` and bail.
            Post_call, by contrast, iterates sequentially and *all*
            registered guardrails run.
            Net effect when a single request lists multiple
            alice_wonderfence guardrails (e.g. `guardrails: ["wonderfence",
            "alice-wonderfence"]` against a config that defines both):
            only one writes a stash, but every one tries to read one in
            post_call.
            Since every alice_wonderfence instance resolves api_key /
            app_id from the same request-body / key / team metadata
            fields, sibling stashes carry equivalent values.
        """
        if logging_obj is None:
            return None
        container = logging_obj.model_call_details.get(_LOGGING_OBJ_STASH_KEY)
        if not container:
            return None
        own = container.get(self.guardrail_name)
        if own is not None:
            return own
        sibling_name, sibling_value = next(iter(container.items()))
        logger.warning(
            "Alice WonderFence: post_call recovering stash from sibling "
            "guardrail '%s' (own name '%s' not in stash). See "
            "_recover_resolved docstring for why.",
            sibling_name,
            self.guardrail_name,
        )
        return sibling_value

    def _extract_relevant_text(
        self,
        inputs: GenericGuardrailAPIInputs,
        input_type: Literal["request", "response"],
    ) -> Optional[str]:
        """Extract latest user message (request) or latest assistant message (response)."""
        if input_type == "request":
            structured_messages = inputs.get("structured_messages", [])
            if structured_messages:
                return get_last_user_message(structured_messages)
            texts = inputs.get("texts", [])
            return texts[-1] if texts else None
        texts = inputs.get("texts", [])
        return texts[-1] if texts else None

    def _resolve_credentials(
        self,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> Tuple[str, str]:
        """Resolve (api_key, app_id) for this call.

        For ``request``: read from request_data (canonical pre_call path) and
        stash on logging_obj so post_call can recover.

        For ``response`` (post_call): try synthesized request_data first
        (works when supplied via virtual key or team metadata, which the
        framework preserves as ``litellm_metadata.user_api_key_metadata`` /
        ``user_api_key_team_metadata``); fall back to the per-request
        logging_obj stash for values supplied in the original request body's
        metadata, which the framework drops before post_call.
        """
        if input_type == "request":
            api_key = self._resolve_api_key(request_data)
            app_id = self._resolve_app_id(request_data)
            self._stash_resolved(logging_obj, api_key, app_id)
            return api_key, app_id
        try:
            return self._resolve_api_key(request_data), self._resolve_app_id(
                request_data
            )
        except (WonderFenceMissingSecrets, ValueError):
            recovered = self._recover_resolved(logging_obj)
            if recovered is None:
                raise
            return recovered

    def _handle_action(
        self,
        result: Any,
        inputs: GenericGuardrailAPIInputs,
    ) -> None:
        """Dispatch BLOCK/MASK/DETECT/NO_ACTION. Raises WonderFenceBlockedError on BLOCK."""
        action = (
            result.action.value if hasattr(result.action, "value") else result.action
        )
        correlation_id = getattr(result, "correlation_id", None)

        if action == "BLOCK":
            detail: dict = {
                "error": self.block_message,
                "type": "alice_wonderfence_content_policy_violation",
                "guardrail_name": self.guardrail_name,
                "action": "BLOCK",
                "wonderfence_correlation_id": correlation_id,
            }
            if hasattr(result, "detections") and result.detections:
                detail["detections"] = [
                    d.model_dump() if hasattr(d, "model_dump") else str(d)
                    for d in result.detections
                ]
            raise WonderFenceBlockedError(detail)
        if action == "MASK":
            masked_text = result.action_text or "[MASKED]"
            texts = inputs.get("texts", [])
            if texts:
                texts[-1] = masked_text
                inputs["texts"] = texts
            logger.info(
                "Alice WonderFence (apply_guardrail): MASK applied guardrail=%s correlation_id=%s",
                self.guardrail_name,
                correlation_id,
            )
        elif action == "DETECT":
            logger.warning(
                "Alice WonderFence (apply_guardrail): DETECT guardrail=%s correlation_id=%s",
                self.guardrail_name,
                correlation_id,
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
        text = self._extract_relevant_text(inputs, input_type)
        if not text:
            logger.debug(
                "Alice WonderFence (apply_guardrail): no relevant text for %s",
                input_type,
            )
            return inputs

        try:
            api_key, app_id = self._resolve_credentials(
                request_data, input_type, logging_obj
            )
            client = await self._get_client(api_key)
            context = self._build_analysis_context(request_data)

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

            self._handle_action(result, inputs)

        except WonderFenceBlockedError as e:
            raise HTTPException(status_code=400, detail=e.detail)
        except Exception as e:
            if self.fail_open:
                logger.critical(
                    "Alice WonderFence unreachable (fail-open). Proceeding without guardrail. "
                    "guardrail_name=%s input_type=%s error=%s",
                    self.guardrail_name,
                    input_type,
                    str(e),
                    exc_info=e,
                )
                add_guardrail_to_applied_guardrails_header(
                    request_data=request_data, guardrail_name=self.guardrail_name
                )
                return inputs
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
