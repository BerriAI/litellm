import asyncio
import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final
from uuid import uuid4

import httpx

from litellm._logging import verbose_router_logger
from litellm.router_strategy.adept_router.config import DEFAULT_CONVERSATIONS_THRESHOLD
from litellm.router_strategy.adept_router.store.store_template import AdeptTemplateStore
from litellm.router_strategy.adept_router.template.router_template import (
    AdeptTemplateMatch,
    BaseTemplateRouter,
)

if TYPE_CHECKING:
    from litellm.router import Router
else:
    Router: Final = object


class AdeptTemplateRouter(BaseTemplateRouter):
    """
    Routes single-turn prompts to task-specific SLMs by matching their structural template.

    Intended use case: an agent/tool sends a fixed system prompt (the task definition) and
    XML-tagged variable user content (the runtime input). ADEPT strips the tag values, leaving
    a stable structural skeleton, and uses SHA-256(system_prompt | skeleton) as a routing key.
    Each unique tool gets its own template family, its own training dataset, and — after enough
    conversations — its own trained SLM.

    Flow:
      1. User message is normalized (whitespace) and XML tag values are stripped.
      2. Remaining variable spans (IDs, emails, URLs, numbers, UUIDs) are masked to placeholders.
      3. The masked skeleton is hashed together with the system prompt for per-tool isolation.
      4. Hash is looked up in Postgres — a hit routes to the template's target_model.
      5. On a miss the skeleton is stored; the default model handles the request.
      6. Every response is stored as a training conversation linked to the template.
      7. At every multiple of conversations_threshold, the external trainer is notified.
    """

    # Compiled at class level — shared across all instances, never recompiled per call.
    ID_RE = re.compile(r"\b[A-Z]{2,}-\d{3,}\b")
    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    URL_RE = re.compile(r"https?://\S+|www\.\S+")
    # UUID must be masked before NUM — UUID hex digits would otherwise partially match NUM_RE.
    UUID_RE = re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
    )
    NUM_RE = re.compile(r"\b\d{1,4}([/-]\d{1,2}([/-]\d{1,4})?)?\b")
    NORMALIZE_RE = re.compile(r"\s+")

    def __init__(
        self,
        model_name: str,
        litellm_router_instance: "Router",
        pg_url: str,
        tag_prefix: str = "",
        conversations_threshold: int = DEFAULT_CONVERSATIONS_THRESHOLD,
        trainer_url: str | None = None,
    ) -> None:
        from litellm.router_strategy.adept_router.store.implementation.prisma import (
            AdeptPrismaRepo,
        )

        self.model_name = model_name
        self.litellm_router_instance = litellm_router_instance
        self.tag_prefix = tag_prefix
        self.conversations_threshold = conversations_threshold
        self.trainer_url = trainer_url
        self._router_id_cache: str | None = None

        # Both the match pattern and its replacement string depend on tag_prefix, so they are
        # built once here rather than on every _extract_template call.
        escaped_prefix: Final = re.escape(self.tag_prefix)
        self.TAG_CONTENT_RE = re.compile(
            r"<" + escaped_prefix + r"([a-zA-Z0-9_ ]+)>([^<]*)</" + escaped_prefix + r"\1>"
        )
        self.TAG_REPLACEMENT = r"<" + escaped_prefix + r"\1></" + escaped_prefix + r"\1>"

        self.template_store: AdeptTemplateStore = AdeptPrismaRepo(pg_url)

    def get_router_id(self) -> str:
        if self._router_id_cache is None:
            self._router_id_cache = self.litellm_router_instance.get_model_ids(model_name=self.model_name)[0]
        return self._router_id_cache

    def _normalize_text(self, text: str) -> str:
        return self.NORMALIZE_RE.sub(" ", text.strip())

    def _mask_text(self, text: str) -> str:
        ids: Final = self.ID_RE.sub("{ID}", text)
        emails: Final = self.EMAIL_RE.sub("{EMAIL}", ids)
        urls: Final = self.URL_RE.sub("{URL}", emails)
        uuids: Final = self.UUID_RE.sub("{UUID}", urls)  # before NUM — see UUID_RE comment above
        return self.NUM_RE.sub("{NUM}", uuids)

    def _extract_tag_content(self, text: str) -> Sequence[tuple[str, str]]:
        """Return (tag_name, value) pairs for all XML-tagged spans in text."""
        return tuple((match.group(1), match.group(2)) for match in self.TAG_CONTENT_RE.finditer(text))

    def _extract_template(self, prompt: str) -> tuple[str, Sequence[tuple[str, str]]]:
        normalized: Final = self._normalize_text(prompt)
        extractions: Final = self._extract_tag_content(normalized)
        skeleton: Final = self.TAG_CONTENT_RE.sub(self.TAG_REPLACEMENT, normalized)
        masked_template: Final = self._mask_text(skeleton)
        verbose_router_logger.debug("Extracted template: %s... (%s tags)", masked_template[:100], len(extractions))
        return masked_template, extractions

    @staticmethod
    def _hash_template(masked_template: str, system_prompt: str | None = None) -> str:
        """
        Produce a routing key from the masked template skeleton.

        When a system prompt is provided it is prepended so that two tools with identical
        user-message structure but different task definitions hash to different templates.
        This is the per-tool isolation guarantee: same tool → same hash, different tool → different hash.
        """
        if system_prompt:
            normalized_sys: Final = re.sub(r"\s+", " ", system_prompt.strip())
            payload: Final = normalized_sys + " | " + masked_template
            return hashlib.sha256(payload.encode()).hexdigest()
        return hashlib.sha256(masked_template.encode()).hexdigest()

    async def seed_template(self, description: str, target_model: str) -> bool:
        """Pre-populate one template from a seed description. Returns True if a new template was stored."""
        masked: Final = self._mask_text(self._normalize_text(description))
        # Use the shared hash function so seeding stays consistent with live routing.
        template_hash: Final = self._hash_template(masked)
        router_id: Final = self.get_router_id()
        if await self.template_store.match_by_hash(template_hash, router_id) is not None:
            return False
        await self.template_store.store_template(
            template_id=str(uuid4()),
            template=masked,
            template_hash=template_hash,
            target_model=target_model,
            router_id=router_id,
        )
        return True

    async def route(self, prompt: str, system_prompt: str | None = None) -> AdeptTemplateMatch | None:
        try:
            masked_template, _ = self._extract_template(prompt)
            template_hash: Final = self._hash_template(masked_template, system_prompt)
            router_id: Final = self.get_router_id()
            template_id: Final = await self.template_store.match_by_hash(template_hash, router_id)
            if template_id is None:
                verbose_router_logger.debug("No matching template found")
                return None

            stored: Final = await self.template_store.get_template(template_id)
            if stored is None:
                # Hash pointed to a deleted row — safe to ignore; next store_conversation will recreate it.
                verbose_router_logger.debug("Template ID found but no metadata — stale reference")
                return None

            verbose_router_logger.info("Matched template %s", template_id)
            return AdeptTemplateMatch(
                template_id=stored.id,
                template=stored.template,
                target_model=stored.target_model,
                metadata=stored.additional_information,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
            verbose_router_logger.exception("Error matching template: %s", e)
            return None

    async def _resolve_template_id(
        self, masked_template: str, template_hash: str, router_id: str, system_prompt: str | None
    ) -> str:
        """Return the id of the template for this hash, storing a new one on a miss."""
        matched_id: Final = await self.template_store.match_by_hash(template_hash, router_id)
        if matched_id is not None:
            return matched_id

        verbose_router_logger.info("No existing template found, storing new template.")
        sys_prompt_payload: Final = {"system_prompt": system_prompt}  # mutable-ok: JSON column payload
        template_additional_info: Final[Mapping[str, object] | None] = sys_prompt_payload if system_prompt else None
        # store_template returns the surviving id (handles concurrent inserts safely).
        stored_id: Final = await self.template_store.store_template(
            template_id=str(uuid4()),
            template=masked_template,
            template_hash=template_hash,
            target_model="",
            router_id=router_id,
            additional_information=template_additional_info,
        )
        return stored_id or str(uuid4())

    async def store_conversation(
        self,
        prompt: str,
        response: str,
        model: str | None = None,
        token_usage: Mapping[str, object] | None = None,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        system_prompt: str | None = None,
        routed_to_slm: bool | None = None,
    ) -> None:
        try:
            masked_template, extractions = self._extract_template(prompt)
            template_hash: Final = self._hash_template(masked_template, system_prompt)
            # Resolve router_id once — used for both the hash lookup and, on a miss, template insert.
            router_id: Final = self.get_router_id()
            template_id: Final = await self._resolve_template_id(
                masked_template, template_hash, router_id, system_prompt
            )

            additional_info: Final[dict[str, object]] = {"extractions": extractions}  # mutable-ok: JSON column payload
            if model is not None:
                additional_info["model"] = model
            if token_usage is not None:
                additional_info["token_usage"] = token_usage
            if cost_usd is not None:
                additional_info["cost_usd"] = cost_usd
            if latency_ms is not None:
                additional_info["latency_ms"] = round(latency_ms, 2)
            if routed_to_slm is not None:
                additional_info["routed_to_slm"] = routed_to_slm

            await self.template_store.store_conversation(
                prompt=prompt,
                response=response,
                template_id=template_id,
                additional_information=additional_info,
            )

            conversation_count: Final = await self.template_store.count_conversation_by_template_id(template_id)
            # Modulo check re-triggers at N, 2N, 3N... so training improves as traffic grows.
            if (
                conversation_count is not None
                and conversation_count >= self.conversations_threshold
                and conversation_count % self.conversations_threshold == 0
            ):
                await self._trigger_trainer(template_id)

            verbose_router_logger.info("Stored interaction for template %s", template_id)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
            verbose_router_logger.exception("Error storing interaction: %s", e)

    async def _trigger_trainer(self, template_id: str) -> None:
        """Notify the external trainer that a template has reached a new training threshold."""
        if not self.trainer_url:
            # Visible by default — operators need to know they hit the threshold
            # but no trainer is wired (common during initial setup / when
            # `adept_router_trainer_url` was edited but the proxy not restarted).
            verbose_router_logger.info(
                "AdeptRouter: threshold reached for template %s but no trainer_url configured — skipping notification.",
                template_id,
            )
            return
        try:
            # httpx.post is sync; offload to a thread so the notification never blocks the loop.
            await asyncio.to_thread(httpx.post, url=f"{self.trainer_url}/run-workflow/{template_id}", timeout=10)
            verbose_router_logger.info("Triggered trainer for template %s", template_id)
        except httpx.HTTPError as e:
            verbose_router_logger.warning("Failed to trigger trainer: %s", e)
