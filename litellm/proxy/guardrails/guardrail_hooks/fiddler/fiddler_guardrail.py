# +-------------------------------------------------------------+
#
#           Use Fiddler AI Guardrails for your LLM calls
#                   https://www.fiddler.ai/
#
# +-------------------------------------------------------------+
import os
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import (
    CallTypesLiteral,
    Choices,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
)

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


SAFETY_DIMENSIONS = [
    "fdl_harmful",
    "fdl_violent",
    "fdl_unethical",
    "fdl_illegal",
    "fdl_sexual",
    "fdl_racist",
    "fdl_jailbreaking",
    "fdl_harassing",
    "fdl_hateful",
    "fdl_sexist",
    "fdl_roleplaying",
]


class FiddlerGuardrailMissingSecrets(Exception):
    pass


class FiddlerGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        safety_threshold: float = 0.5,
        pii_threshold: float = 0.5,
        faithfulness_threshold: float = 0.5,
        enable_safety: bool = True,
        enable_pii: bool = True,
        enable_faithfulness: bool = True,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        self.api_key = api_key or os.environ.get("FIDDLER_API_KEY")
        if not self.api_key:
            raise FiddlerGuardrailMissingSecrets(
                "Couldn't get Fiddler api key, either set the `FIDDLER_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )
        self.api_base = (
            api_base or os.environ.get("FIDDLER_API_BASE") or "https://app.fiddler.ai"
        ).rstrip("/")
        self.safety_threshold = float(safety_threshold)
        self.pii_threshold = float(pii_threshold)
        self.faithfulness_threshold = float(faithfulness_threshold)
        self.enable_safety = bool(enable_safety)
        self.enable_pii = bool(enable_pii)
        self.enable_faithfulness = bool(enable_faithfulness)
        super().__init__(**kwargs)

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _extract_last_user_message(self, messages: List[dict]) -> str:
        """Return the text of the most recent user message."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
        return ""

    def _check_safety_scores(self, result: dict) -> Optional[str]:
        """Return the first safety dimension that exceeds the threshold, or None."""
        for dim in SAFETY_DIMENSIONS:
            if result.get(dim, 0.0) >= self.safety_threshold:
                return dim
        return None

    async def _call_safety_api(self, text: str) -> dict:
        response = await self.async_handler.post(
            f"{self.api_base}/v3/guardrails/ftl-safety",
            headers=self._build_headers(),
            json={"data": {"input": text}},
        )
        response.raise_for_status()
        return response.json()

    async def _call_pii_api(self, text: str) -> dict:
        response = await self.async_handler.post(
            f"{self.api_base}/v3/guardrails/sensitive-information",
            headers=self._build_headers(),
            json={"data": {"input": text}},
        )
        response.raise_for_status()
        return response.json()

    async def _call_faithfulness_api(self, response_text: str, context: str) -> dict:
        response = await self.async_handler.post(
            f"{self.api_base}/v3/guardrails/ftl-response-faithfulness",
            headers=self._build_headers(),
            json={"data": {"response": response_text, "context": context}},
        )
        response.raise_for_status()
        return response.json()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside Fiddler Pre-Call Hook")

        messages = data.get("messages", [])
        input_text = self._extract_last_user_message(messages)
        if not input_text:
            return data

        if self.enable_safety:
            result = await self._call_safety_api(input_text)
            triggered_dim = self._check_safety_scores(result)
            if triggered_dim:
                verbose_proxy_logger.info(
                    f"Fiddler safety guardrail triggered on input: dimension={triggered_dim}"
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Fiddler safety guardrail triggered",
                        "dimension": triggered_dim,
                        "scores": result,
                    },
                )

        if self.enable_pii:
            result = await self._call_pii_api(input_text)
            detections = result.get("fdl_sensitive_information_scores", [])
            flagged = [d for d in detections if d.get("score", 0.0) >= self.pii_threshold]
            if flagged:
                verbose_proxy_logger.info(
                    f"Fiddler PII guardrail triggered on input: detections={flagged}"
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Fiddler PII guardrail triggered",
                        "detections": flagged,
                    },
                )

        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
    ) -> Any:
        verbose_proxy_logger.debug("Inside Fiddler Post-Call Success Hook")

        if not (
            isinstance(response, ModelResponse)
            and response.choices
            and isinstance(response.choices[0], Choices)
        ):
            return response

        response_text = response.choices[0].message.content or ""
        if not response_text:
            return response

        if self.enable_safety:
            result = await self._call_safety_api(response_text)
            triggered_dim = self._check_safety_scores(result)
            if triggered_dim:
                verbose_proxy_logger.info(
                    f"Fiddler safety guardrail triggered on output: dimension={triggered_dim}"
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Fiddler safety guardrail triggered on response",
                        "dimension": triggered_dim,
                        "scores": result,
                    },
                )

        if self.enable_faithfulness:
            context = data.get("metadata", {}).get("fiddler_context", "")
            if context:
                result = await self._call_faithfulness_api(response_text, context)
                score = result.get("fdl_faithful_score", 1.0)
                if score < self.faithfulness_threshold:
                    verbose_proxy_logger.info(
                        f"Fiddler faithfulness guardrail triggered: score={score} < threshold={self.faithfulness_threshold}"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Fiddler faithfulness guardrail triggered",
                            "faithfulness_score": score,
                            "threshold": self.faithfulness_threshold,
                        },
                    )
            else:
                verbose_proxy_logger.debug(
                    "Fiddler faithfulness check skipped: no `fiddler_context` in request metadata"
                )

        return response

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.fiddler import (
            FiddlerGuardrailConfigModel,
        )

        return FiddlerGuardrailConfigModel
