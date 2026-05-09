"""
Prompt-prefix-aware deterministic deployment affinity for the Router.

This is a stateless optimization for upstream implicit prompt caching. It
canonicalizes the prompt-bearing parts of a request, hashes the first N tokens,
then uses rendezvous hashing to choose a stable deployment from the current
healthy deployment set.

Unlike deployment/session affinity, this does not store a prompt -> deployment
mapping in Redis. All Router instances with the same config and deployment IDs
will make the same routing decision.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, cast

from litellm._logging import verbose_router_logger
from litellm.constants import MINIMUM_PROMPT_CACHE_TOKEN_COUNT
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import encode


class PromptPrefixAffinityCheck(CustomLogger):
    """
    Routes requests with the same canonical prompt prefix to the same deployment.

    This is intended to improve prompt-cache hit rate for providers where prompt
    caching is scoped to the account/key behind a deployment.
    """

    CACHE_KEY_EXCLUDED_FIELDS = frozenset({"encrypted_content"})

    def __init__(
        self,
        prefix_tokens: int = 2048,
        min_tokens: int = MINIMUM_PROMPT_CACHE_TOKEN_COUNT,
    ) -> None:
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.min_tokens = min_tokens

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if hasattr(value, "model_dump"):
            return cls._json_safe(value.model_dump())

        if hasattr(value, "dict"):
            return cls._json_safe(value.dict())

        if isinstance(value, dict):
            return {
                str(k): cls._json_safe(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
                if str(k) not in cls.CACHE_KEY_EXCLUDED_FIELDS
            }

        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]

        return str(value)

    @classmethod
    def _build_canonical_prompt(
        cls,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Dict[str, Any],
    ) -> Optional[str]:
        prompt_parts: List[Tuple[str, Any]] = []

        for key in ("instructions", "tools"):
            value = request_kwargs.get(key)
            if value is not None:
                prompt_parts.append((key, cls._json_safe(value)))

        if messages is not None:
            prompt_parts.append(("messages", cls._json_safe(messages)))

        for key in ("input",):
            value = request_kwargs.get(key)
            if value is not None:
                prompt_parts.append((key, cls._json_safe(value)))

        if not prompt_parts:
            return None

        return json.dumps(
            prompt_parts,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def _get_prefix_hash(
        self,
        model: str,
        canonical_prompt: str,
    ) -> Optional[str]:
        if self.prefix_tokens <= 0:
            return None

        try:
            token_ids = encode(model=model, text=canonical_prompt)
        except Exception as e:
            verbose_router_logger.debug(
                "PromptPrefixAffinityCheck: failed to tokenize prompt for model=%s; error=%s",
                model,
                e,
            )
            return None

        if len(token_ids) < self.min_tokens:
            return None

        prefix_token_ids = token_ids[: self.prefix_tokens]
        prefix_payload = json.dumps(prefix_token_ids, separators=(",", ":"))
        return hashlib.sha256(prefix_payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _get_deployment_model_id(deployment: dict) -> Optional[str]:
        model_info = deployment.get("model_info")
        if not isinstance(model_info, dict):
            return None

        model_id = model_info.get("id")
        if model_id is None:
            return None

        return str(model_id)

    def _score_deployment(
        self,
        prefix_hash: str,
        deployment_model_id: str,
    ) -> int:
        payload = f"{prefix_hash}:{deployment_model_id}"
        return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)

    def _select_deployment(
        self,
        prefix_hash: str,
        healthy_deployments: List[dict],
    ) -> Optional[dict]:
        best: Optional[Tuple[int, dict]] = None

        for deployment in healthy_deployments:
            deployment_model_id = self._get_deployment_model_id(deployment)
            if deployment_model_id is None:
                continue

            score = self._score_deployment(
                prefix_hash=prefix_hash,
                deployment_model_id=deployment_model_id,
            )
            if best is None or score > best[0]:
                best = (score, deployment)

        return best[1] if best is not None else None

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> List[dict]:
        typed_healthy_deployments = cast(List[dict], healthy_deployments)

        if len(typed_healthy_deployments) <= 1:
            return typed_healthy_deployments

        request_kwargs = request_kwargs or {}
        canonical_prompt = self._build_canonical_prompt(
            messages=messages,
            request_kwargs=request_kwargs,
        )
        if canonical_prompt is None:
            return typed_healthy_deployments

        prefix_hash = self._get_prefix_hash(
            model=model,
            canonical_prompt=canonical_prompt,
        )
        if prefix_hash is None:
            return typed_healthy_deployments

        deployment = self._select_deployment(
            prefix_hash=prefix_hash,
            healthy_deployments=typed_healthy_deployments,
        )
        if deployment is None:
            return typed_healthy_deployments

        request_kwargs["_prompt_prefix_affinity_pinned"] = True
        verbose_router_logger.debug(
            "PromptPrefixAffinityCheck: pinning model=%s prefix_hash=%s deployment=%s",
            model,
            prefix_hash[:8],
            self._get_deployment_model_id(deployment),
        )
        return [deployment]
