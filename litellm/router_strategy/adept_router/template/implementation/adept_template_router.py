import hashlib
import re
import httpx
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from uuid import uuid4

from litellm._logging import verbose_router_logger
from litellm.router_strategy.adept_router.config import DEFAULT_CONVERSATIONS_THRESHOLD
from litellm.router_strategy.adept_router.store.store_template import AdeptTemplateStore
from litellm.router_strategy.adept_router.template.router_template import (
    BaseTemplateRouter,
)

if TYPE_CHECKING:
    from litellm.router import Router
else:
    Router = Any


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
        trainer_url: Optional[str] = None,
    ) -> None:
        from litellm.router_strategy.adept_router.store.implementation.postgresql import (
            PostgresTemplateRepo,
        )

        self.model_name = model_name
        self.litellm_router_instance = litellm_router_instance
        self.tag_prefix = tag_prefix
        self.conversations_threshold = conversations_threshold
        self.trainer_url = trainer_url
        self._router_id_cache: Optional[str] = None

        # Both the match pattern and its replacement string depend on tag_prefix, so they are
        # built once here rather than on every _extract_template call.
        escaped_prefix = re.escape(self.tag_prefix)
        self.TAG_CONTENT_RE = re.compile(
            r"<"
            + escaped_prefix
            + r"([a-zA-Z0-9_ ]+)>([^<]*)</"
            + escaped_prefix
            + r"\1>"
        )
        self.TAG_REPLACEMENT = (
            r"<" + escaped_prefix + r"\1></" + escaped_prefix + r"\1>"
        )

        self.template_store: AdeptTemplateStore = PostgresTemplateRepo(pg_url)

    def get_router_id(self) -> str:
        if self._router_id_cache is None:
            self._router_id_cache = self.litellm_router_instance.get_model_ids(
                model_name=self.model_name
            )[0]
        return self._router_id_cache

    def _normalize_text(self, text: str) -> str:
        return self.NORMALIZE_RE.sub(" ", text.strip())

    def _mask_text(self, text: str) -> str:
        text = self.ID_RE.sub("{ID}", text)
        text = self.EMAIL_RE.sub("{EMAIL}", text)
        text = self.URL_RE.sub("{URL}", text)
        text = self.UUID_RE.sub(
            "{UUID}", text
        )  # before NUM — see UUID_RE comment above
        text = self.NUM_RE.sub("{NUM}", text)
        return text

    def _extract_tag_content(self, text: str) -> List[Tuple[str, str]]:
        """Return (tag_name, value) pairs for all XML-tagged spans in text."""
        return [
            (match.group(1), match.group(2))
            for match in self.TAG_CONTENT_RE.finditer(text)
        ]

    def _extract_template(self, prompt: str) -> Tuple[str, List[Tuple[str, str]]]:
        normalized = self._normalize_text(prompt)
        extractions = self._extract_tag_content(normalized)
        skeleton = self.TAG_CONTENT_RE.sub(self.TAG_REPLACEMENT, normalized)
        masked_template = self._mask_text(skeleton)
        verbose_router_logger.debug(
            f"Extracted template: {masked_template[:100]}... ({len(extractions)} tags)"
        )
        return masked_template, extractions

    @staticmethod
    def _hash_template(
        masked_template: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Produce a routing key from the masked template skeleton.

        When a system prompt is provided it is prepended so that two tools with identical
        user-message structure but different task definitions hash to different templates.
        This is the per-tool isolation guarantee: same tool → same hash, different tool → different hash.
        """
        if system_prompt:
            normalized_sys = re.sub(r"\s+", " ", system_prompt.strip())
            payload = normalized_sys + " | " + masked_template
        else:
            payload = masked_template
        return hashlib.sha256(payload.encode()).hexdigest()

    def route(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            masked_template, _ = self._extract_template(prompt)
            template_hash = self._hash_template(masked_template, system_prompt)
            router_id = self.get_router_id()
            template_id = self.template_store.match_by_hash(template_hash, router_id)
            if template_id is None:
                verbose_router_logger.debug("No matching template found")
                return None

            stored = self.template_store.get_template(template_id)
            if stored is None:
                # Hash pointed to a deleted row — safe to ignore; next store_conversation will recreate it.
                verbose_router_logger.debug(
                    "Template ID found but no metadata — stale reference"
                )
                return None

            verbose_router_logger.info(f"Matched template {template_id}")
            return {
                "template_id": stored["id"],
                "template": stored["template"],
                "target_model": stored.get("target_model"),
                "metadata": stored.get("additional_information"),
            }
        except Exception as e:
            verbose_router_logger.exception(f"Error matching template: {e}")
            return None

    def store_conversation(
        self,
        prompt: str,
        response: str,
        model: Optional[str] = None,
        token_usage: Optional[Dict[str, int]] = None,
        cost_usd: Optional[float] = None,
        latency_ms: Optional[float] = None,
        system_prompt: Optional[str] = None,
        routed_to_slm: Optional[bool] = None,
    ) -> None:
        try:
            masked_template, extractions = self._extract_template(prompt)
            template_hash = self._hash_template(masked_template, system_prompt)
            # Resolve router_id once — used for both the hash lookup and, on a miss, template insert.
            router_id = self.get_router_id()
            template_id = self.template_store.match_by_hash(template_hash, router_id)

            if template_id is None:
                verbose_router_logger.info(
                    "No existing template found, storing new template."
                )
                template_additional_info = (
                    {"system_prompt": system_prompt} if system_prompt else None
                )
                # store_template returns the surviving id (handles concurrent inserts safely).
                stored_id = self.template_store.store_template(
                    template_id=str(uuid4()),
                    template=masked_template,
                    template_hash=template_hash,
                    target_model="",
                    router_id=router_id,
                    additional_information=template_additional_info,
                )
                template_id = stored_id or str(uuid4())

            additional_info: Dict[str, Any] = {"extractions": extractions}
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

            self.template_store.store_conversation(
                prompt=prompt,
                response=response,
                template_id=template_id,
                additional_information=additional_info,
            )

            conversation_count = self.template_store.count_conversation_by_template_id(
                template_id
            )
            # Modulo check re-triggers at N, 2N, 3N... so training improves as traffic grows.
            if (
                conversation_count is not None
                and conversation_count >= self.conversations_threshold
                and conversation_count % self.conversations_threshold == 0
            ):
                self._trigger_trainer(template_id)

            verbose_router_logger.info(f"Stored interaction for template {template_id}")
        except Exception as e:
            verbose_router_logger.exception(f"Error storing interaction: {e}")

    def _trigger_trainer(self, template_id: str) -> None:
        """Notify the external trainer that a template has reached a new training threshold."""
        if not self.trainer_url:
            # Visible by default — operators need to know they hit the threshold
            # but no trainer is wired (common during initial setup / when
            # `adept_router_trainer_url` was edited but the proxy not restarted).
            verbose_router_logger.info(
                f"AdeptRouter: threshold reached for template {template_id} but no "
                "trainer_url configured — skipping notification."
            )
            return
        try:
            httpx.post(
                url=f"{self.trainer_url}/run-workflow/{template_id}",
                timeout=10,
            )
            verbose_router_logger.info(f"Triggered trainer for template {template_id}")
        except Exception as e:
            verbose_router_logger.warning(f"Failed to trigger trainer: {e}")
