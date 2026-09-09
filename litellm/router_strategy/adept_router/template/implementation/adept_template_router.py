import asyncio
import hashlib
import re
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, TypeAlias
from uuid import uuid4

import httpx

from litellm._logging import verbose_router_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.router_strategy.adept_router.config import DEFAULT_CONVERSATIONS_THRESHOLD
from litellm.router_strategy.adept_router.store.store_template import (
    AdeptTemplateStore,
    StoredTemplate,
)
from litellm.router_strategy.adept_router.template.router_template import (
    AdeptTemplateMatch,
    BaseTemplateRouter,
)
from litellm.types.llms.custom_http import httpxSpecialProvider

if TYPE_CHECKING:
    from litellm.router import Router
else:
    Router: Final = object

_TEMPLATE_CACHE_MAX_SIZE: Final = 1024
_TEMPLATE_CACHE_TTL_SECONDS: Final = 60.0
_REFRESH_INTERVAL_SECONDS: Final = 60.0
_TRAINER_HTTP_TIMEOUT_SECONDS: Final = 10.0

_TemplateCache: TypeAlias = OrderedDict[tuple[str, str], tuple[float, StoredTemplate]]


class AdeptTemplateRouter(BaseTemplateRouter):
    ID_RE = re.compile(r"\b[A-Z]{2,}-\d{3,}\b")
    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    URL_RE = re.compile(r"https?://\S+|www\.\S+")
    # UUID must be masked before NUM — UUID hex digits partially match NUM_RE.
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
        self._template_cache: _TemplateCache = OrderedDict()  # mutable-ok: LRU cache, bounded and TTL-gated
        self._refresh_task: asyncio.Task[None] | None = None  # mutable-ok: task handle rotates on restart

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
        uuids: Final = self.UUID_RE.sub("{UUID}", urls)
        return self.NUM_RE.sub("{NUM}", uuids)

    def _extract_tag_content(self, text: str) -> Sequence[tuple[str, str]]:
        return tuple((match.group(1), match.group(2)) for match in self.TAG_CONTENT_RE.finditer(text))

    def _extract_template(self, prompt: str) -> tuple[str, Sequence[tuple[str, str]]]:
        normalized: Final = self._normalize_text(prompt)
        extractions: Final = self._extract_tag_content(normalized)
        skeleton: Final = self.TAG_CONTENT_RE.sub(self.TAG_REPLACEMENT, normalized)
        masked_template: Final = self._mask_text(skeleton)
        verbose_router_logger.debug("Extracted template with %s tag(s)", len(extractions))
        return masked_template, extractions

    @staticmethod
    def _hash_template(masked_template: str, system_prompt: str | None = None) -> str:
        # Prepending the system prompt isolates two tools with identical user-message
        # structure but different task definitions.
        if system_prompt:
            normalized_sys: Final = re.sub(r"\s+", " ", system_prompt.strip())
            payload: Final = normalized_sys + " | " + masked_template
            return hashlib.sha256(payload.encode()).hexdigest()
        return hashlib.sha256(masked_template.encode()).hexdigest()

    async def seed_template(self, description: str, target_model: str) -> bool:
        masked: Final = self._mask_text(self._normalize_text(description))
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
            self._ensure_refresh_running()
            masked_template, _ = self._extract_template(prompt)
            template_hash: Final = self._hash_template(masked_template, system_prompt)
            router_id: Final = self.get_router_id()
            cached: Final = self._cache_get((router_id, template_hash))
            if cached is None:
                verbose_router_logger.debug("No matching template in cache")
                return None
            verbose_router_logger.debug("Template cache hit for hash %s", template_hash[:8])
            return AdeptTemplateMatch(
                template_id=cached.id,
                template=cached.template,
                target_model=cached.target_model,
                metadata=cached.additional_information,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
            verbose_router_logger.exception("Error matching template: %s", e)
            return None

    def _ensure_refresh_running(self) -> None:
        existing: Final = self._refresh_task
        if existing is not None and not existing.done():
            return
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._refresh_task = loop.create_task(self._refresh_loop())

    def stop_refresh(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None

    async def _refresh_loop(self) -> None:
        while True:
            try:
                await self.refresh_cache()
            except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
                verbose_router_logger.warning("AdeptRouter: cache refresh failed: %s", e)
            await asyncio.sleep(_REFRESH_INTERVAL_SECONDS)

    async def refresh_cache(self) -> None:
        router_id: Final = self.get_router_id()
        templates: Final = await self.template_store.load_all_for_router(router_id, limit=_TEMPLATE_CACHE_MAX_SIZE)
        now: Final = time.monotonic()
        rebuilt: Final[_TemplateCache] = OrderedDict(  # mutable-ok: cache rebuild populated in one pass then swapped in
            ((router_id, t.template_hash), (now, t)) for t in templates if t.template_hash is not None
        )
        self._template_cache = rebuilt  # rebind-ok: full cache swap after background refresh

    def _cache_get(self, key: tuple[str, str]) -> StoredTemplate | None:
        entry: Final = self._template_cache.get(key)
        if entry is None:
            return None
        inserted_at, stored = entry
        if time.monotonic() - inserted_at > _TEMPLATE_CACHE_TTL_SECONDS:
            self._template_cache.pop(key, None)
            return None
        self._template_cache.move_to_end(key)
        return stored

    def _cache_put(self, key: tuple[str, str], stored: StoredTemplate) -> None:
        self._template_cache[key] = (time.monotonic(), stored)
        self._template_cache.move_to_end(key)
        while len(self._template_cache) > _TEMPLATE_CACHE_MAX_SIZE:
            self._template_cache.popitem(last=False)

    def _cache_invalidate(self, key: tuple[str, str]) -> None:
        self._template_cache.pop(key, None)

    async def _resolve_template_id(
        self, masked_template: str, template_hash: str, router_id: str, system_prompt: str | None
    ) -> str:
        matched_id: Final = await self.template_store.match_by_hash(template_hash, router_id)
        if matched_id is not None:
            return matched_id

        verbose_router_logger.info("No existing template found, storing new template.")
        sys_prompt_payload: Final = {"system_prompt": system_prompt}  # mutable-ok: JSON column payload
        template_additional_info: Final[Mapping[str, object] | None] = sys_prompt_payload if system_prompt else None
        stored_id: Final = await self.template_store.store_template(
            template_id=str(uuid4()),
            template=masked_template,
            template_hash=template_hash,
            target_model="",
            router_id=router_id,
            additional_information=template_additional_info,
        )
        self._cache_invalidate((router_id, template_hash))
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
            if (
                conversation_count is not None
                and conversation_count >= self.conversations_threshold
                and conversation_count % self.conversations_threshold == 0
            ):
                # Trainer runs may flip target_model, so drop the cached row before firing.
                self._cache_invalidate((router_id, template_hash))
                self._trigger_trainer(template_id)

            verbose_router_logger.info("Stored interaction for template %s", template_id)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
            verbose_router_logger.exception("Error storing interaction: %s", e)

    def _trigger_trainer(self, template_id: str) -> None:
        if not self.trainer_url:
            verbose_router_logger.info(
                "AdeptRouter: threshold reached for template %s but no trainer_url configured — skipping notification.",
                template_id,
            )
            return
        asyncio.create_task(self._trainer_post(f"{self.trainer_url}/run-workflow/{template_id}", template_id))

    @staticmethod
    async def _trainer_post(url: str, template_id: str) -> None:
        from litellm.litellm_core_utils.url_utils import SSRFError, validate_url

        try:
            pinned_url, host_header = validate_url(url)
        except SSRFError as ssrf_err:
            verbose_router_logger.warning(
                "AdeptRouter: refusing trainer POST for template %s — %s", template_id, ssrf_err
            )
            return
        client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
        try:
            await client.post(
                url=pinned_url,
                headers={"Host": host_header},  # mutable-ok: one-shot per-request headers dict handed to httpx
                timeout=_TRAINER_HTTP_TIMEOUT_SECONDS,
            )
            verbose_router_logger.info("Triggered trainer for template %s", template_id)
        except httpx.HTTPError as e:
            verbose_router_logger.warning("Failed to trigger trainer: %s", e)
