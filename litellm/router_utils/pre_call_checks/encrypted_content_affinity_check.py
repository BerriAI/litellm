"""
Encrypted-content-aware deployment affinity for the Router.

When Codex or other models use `store: false` with `include: ["reasoning.encrypted_content"]`,
the response output items contain encrypted reasoning tokens tied to the originating
organization's API key. If a follow-up request containing those items is routed to a
different deployment (different org), OpenAI rejects it with an `invalid_encrypted_content`
error because the organization_id doesn't match.

This callback solves the problem by encoding the originating deployment's ``model_id``
directly into the item IDs of output items that carry ``encrypted_content`` (the same
approach used by the responses-API affinity for ``previous_response_id``).  The encoded
ID is decoded on the next request so the router can pin to the correct deployment without
any cache lookup.

Response post-processing (encoding) is handled by
``ResponsesAPIRequestUtils._update_encrypted_content_item_ids_in_response`` which is
called inside ``_update_responses_api_response_id_with_model_id`` in ``responses/utils.py``.

Request pre-processing (ID restoration before forwarding to upstream) is handled by
``ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input`` which is called
in ``get_optional_params_responses_api``.

This pre-call check is responsible only for the routing decision: it reads the encoded
``model_id`` out of the item IDs and pins the request to the matching deployment.

Safe to enable globally:
- Only activates when encoded item IDs appear in the request ``input``.
- No effect on embedding models, chat completions, or first-time requests.
- No quota reduction -- first requests are fully load balanced.
- No cache required.
"""

from typing import Any, List, Optional, cast

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import AllMessageValues


class EncryptedContentAffinityCheck(CustomLogger):
    """
    Routes follow-up Responses API requests to the deployment that produced
    the encrypted output items they reference.

    The ``model_id`` is decoded directly from the litellm-encoded item IDs –
    no caching or TTL management needed.

    Wired via ``Router(optional_pre_call_checks=["encrypted_content_affinity"])``.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_model_id_from_input(request_input: Any) -> Optional[str]:
        """
        Scan ``input`` items for litellm-encoded encrypted-content item IDs and
        return the ``model_id`` embedded in the first one found.

        ``input`` can be:
        - a plain string  -> no encoded IDs
        - a list of items -> check each item's ``id`` field
        """
        if not isinstance(request_input, list):
            return None

        for item in request_input:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if not item_id or not isinstance(item_id, str):
                continue
            decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(item_id)
            if decoded:
                return decoded.get("model_id")

        return None

    @staticmethod
    def _find_deployment_by_model_id(
        healthy_deployments: List[dict], model_id: str
    ) -> Optional[dict]:
        for deployment in healthy_deployments:
            model_info = deployment.get("model_info")
            if not isinstance(model_info, dict):
                continue
            deployment_model_id = model_info.get("id")
            if deployment_model_id is not None and str(deployment_model_id) == str(
                model_id
            ):
                return deployment
        return None

    # ------------------------------------------------------------------
    # Request routing  (pre-call filter)
    # ------------------------------------------------------------------

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> List[dict]:
        """
        If the request ``input`` contains litellm-encoded item IDs, decode the
        embedded ``model_id`` and pin the request to that deployment.
        """
        request_kwargs = request_kwargs or {}
        typed_healthy_deployments = cast(List[dict], healthy_deployments)

        # Signal to the response post-processor that encrypted item IDs should be
        # encoded in the output of this request.
        litellm_metadata = request_kwargs.setdefault("litellm_metadata", {})
        litellm_metadata["encrypted_content_affinity_enabled"] = True

        request_input = request_kwargs.get("input")
        model_id = self._extract_model_id_from_input(request_input)
        if not model_id:
            return typed_healthy_deployments

        verbose_router_logger.debug(
            "EncryptedContentAffinityCheck: decoded model_id=%s from input item IDs",
            model_id,
        )

        deployment = self._find_deployment_by_model_id(
            healthy_deployments=typed_healthy_deployments,
            model_id=model_id,
        )
        if deployment is not None:
            verbose_router_logger.debug(
                "EncryptedContentAffinityCheck: pinning -> deployment=%s",
                model_id,
            )
            request_kwargs["_encrypted_content_affinity_pinned"] = True
            return [deployment]

        verbose_router_logger.error(
            "EncryptedContentAffinityCheck: decoded deployment=%s not found in healthy_deployments",
            model_id,
        )
        return typed_healthy_deployments
