import asyncio
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from itertools import chain
from typing import ClassVar, Final, Literal

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.constants import (
    DEFAULT_PROMPT_INJECTION_SIMILARITY_THRESHOLD,
    PROMPT_INJECTION_HEURISTICS_MAX_THREADS,
)
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.prompt_templates.factory import (
    prompt_injection_detection_default_pt,
)
from litellm.llms import load_guardrail_translation_mappings
from litellm.llms.base_llm.guardrail_translation.attachments import as_mapping, request_attachments
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.proxy._types import LiteLLMPromptInjectionParams, UserAPIKeyAuth
from litellm.router import Router
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes, CallTypesLiteral, GenericGuardrailAPIInputs, ModelResponse

GUARDRAIL_NAME: Final = "detect_prompt_injection"
REJECTION_MESSAGE: Final = "Rejected message. This is a prompt injection attack."
SCANNED_REQUEST: Final = TypeAdapter(dict[str, object])
REQUEST_ITEMS: Final = TypeAdapter(tuple[object, ...])
PLAIN_TEXT_REQUEST_FIELDS: Final = ("input", "prompt")
HEURISTICS_EXECUTOR: Final = ThreadPoolExecutor(
    max_workers=PROMPT_INJECTION_HEURISTICS_MAX_THREADS, thread_name_prefix="prompt-injection-heuristics"
)
INJECTION_VERBS: Final = (
    "Ignore",
    "Disregard",
    "Skip",
    "Forget",
    "Neglect",
    "Overlook",
    "Omit",
    "Bypass",
    "Pay no attention to",
    "Do not follow",
    "Do not obey",
)
INJECTION_ADJECTIVES: Final = ("", "prior", "previous", "preceding", "above", "foregoing", "earlier", "initial")
INJECTION_PREPOSITIONS: Final = ("", "and start over", "and start anew", "and begin afresh", "and start from scratch")


def _injection_phrases() -> Iterator[str]:
    for verb in INJECTION_VERBS:
        for adjective in INJECTION_ADJECTIVES:
            for preposition in INJECTION_PREPOSITIONS:
                yield " ".join(part for part in (verb, adjective, preposition) if part).lower()


INJECTION_KEYWORDS: Final = tuple(phrase for phrase in _injection_phrases() if len(phrase.split()) > 2)


def _rejection() -> HTTPException:
    return HTTPException(status_code=400, detail={"error": REJECTION_MESSAGE})


def _unscannable_rejection(part_types: Sequence[str]) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "error": (
                f"Prompt injection detection cannot scan {', '.join(part_types)} content and blocked the request; "
                "set prompt_injection_params.skip_unscannable_attachments to let such parts through unscanned"
            )
        },
    )


def _translation_handler(call_type: str) -> BaseTranslation | None:
    try:
        handler_class: Final = load_guardrail_translation_mappings().get(CallTypes(call_type))
    except ValueError:
        return None
    return None if handler_class is None else handler_class()


def _logging_obj(data: Mapping[str, object]) -> LiteLLMLoggingObj | None:
    logging_obj: Final = data.get("litellm_logging_obj")
    return logging_obj if isinstance(logging_obj, LiteLLMLoggingObj) else None


def _attachment_texts(request_data: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(attachment.text for attachment in request_attachments(request_data).texts)


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    try:
        items: Final = REQUEST_ITEMS.validate_python(value)
    except ValidationError:
        return ()
    return tuple(item for item in items if isinstance(item, str))


def _plain_request_texts(request_data: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(chain.from_iterable(_strings(request_data.get(field)) for field in PLAIN_TEXT_REQUEST_FIELDS))


class _PromptInjectionLLMJudge(CustomGuardrail):
    def __init__(self, params: LiteLLMPromptInjectionParams, llm_api_name: str, router: Router) -> None:
        super().__init__(
            guardrail_name=GUARDRAIL_NAME,
            supported_event_hooks=[GuardrailEventHooks.during_call],
            event_hook=[GuardrailEventHooks.during_call],
            default_on=True,
        )
        self.params = params
        self.llm_api_name = llm_api_name
        self.router = router

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type == "request":
            await self.reject_injection(inputs.get("texts", ()))
        return inputs

    async def reject_injection(self, texts: Iterable[str]) -> None:
        prompt: Final = "\n".join(texts)
        if not prompt.strip():
            return
        response: Final[ModelResponse] = await self.router.acompletion(
            model=self.llm_api_name,
            messages=[
                {
                    "role": "system",
                    "content": self.params.llm_api_system_prompt or prompt_injection_detection_default_pt(),
                },
                {"role": "user", "content": prompt},
            ],
        )
        if self._verdict_is_attack(response):
            raise _rejection()

    def _verdict_is_attack(self, response: ModelResponse) -> bool:
        fail_call_string: Final = self.params.llm_api_fail_call_string
        if fail_call_string is None or not response.choices:
            return False
        content: Final = response.choices[0].message.content
        return isinstance(content, str) and fail_call_string in content


class _OPTIONAL_PromptInjectionDetection(CustomGuardrail):
    use_native_lifecycle_hooks: ClassVar[bool] = True
    enforces_request_content: bool = True

    def __init__(
        self,
        prompt_injection_params: LiteLLMPromptInjectionParams | None = None,
    ):
        super().__init__(
            guardrail_name=GUARDRAIL_NAME,
            supported_event_hooks=[GuardrailEventHooks.pre_call, GuardrailEventHooks.during_call],
            event_hook=[GuardrailEventHooks.pre_call, GuardrailEventHooks.during_call],
            default_on=True,
        )
        self.prompt_injection_params = prompt_injection_params
        self.llm_router: Router | None = None
        self.llm_judge: _PromptInjectionLLMJudge | None = None
        if prompt_injection_params is not None and prompt_injection_params.vector_db_check:
            verbose_proxy_logger.warning(
                "prompt_injection_params.vector_db_check is not implemented; no vector similarity check runs"
            )

    def update_environment(self, router: Router | None = None) -> None:
        self.llm_router = router
        params: Final = self.prompt_injection_params
        if params is None or params.llm_api_check is not True:
            return
        if router is None:
            raise Exception("PromptInjectionDetection: Model List not set. Required for Prompt Injection detection.")
        if params.llm_api_name is None or params.llm_api_name not in router.model_names:
            raise Exception(
                "PromptInjectionDetection: Invalid LLM API Name. LLM API Name must be a 'model_name' in 'model_list'."
            )
        self.llm_judge = _PromptInjectionLLMJudge(params=params, llm_api_name=params.llm_api_name, router=router)

    def generate_injection_keywords(self) -> list[str]:
        return list(INJECTION_KEYWORDS)

    async def check_user_input_similarity_off_loop(self, user_input: str) -> bool:
        return await asyncio.get_running_loop().run_in_executor(
            HEURISTICS_EXECUTOR, self.check_user_input_similarity, user_input
        )

    def check_user_input_similarity(
        self,
        user_input: str,
        similarity_threshold: float = DEFAULT_PROMPT_INJECTION_SIMILARITY_THRESHOLD,
    ) -> bool:
        user_input_lower: Final = user_input.lower()
        for keyword in INJECTION_KEYWORDS:
            for start in range(len(user_input_lower) - len(keyword) + 1):
                match_ratio = SequenceMatcher(None, user_input_lower[start : start + len(keyword)], keyword).ratio()
                if match_ratio > similarity_threshold:
                    verbose_proxy_logger.debug(
                        "Rejected user input - %s. %s similar to %s", user_input, match_ratio, keyword
                    )
                    return True
        return False

    def _heuristics_enabled(self) -> bool:
        return self.prompt_injection_params is None or self.prompt_injection_params.heuristics_check is True

    def _fails_closed(self) -> bool:
        return self.prompt_injection_params is None or self.prompt_injection_params.fail_on_error

    def _skips_unscannable_attachments(self) -> bool:
        return self.prompt_injection_params is not None and self.prompt_injection_params.skip_unscannable_attachments

    def _response_for_rejection(self, exc: HTTPException) -> str | None:
        params: Final = self.prompt_injection_params
        if params is None or not params.reject_as_response or exc.status_code != 400:
            return None
        detail: Final = as_mapping(exc.detail)
        error: Final = None if detail is None else detail.get("error")
        return error if isinstance(error, str) else None

    async def _reject_injected_texts(self, texts: Iterable[str]) -> None:
        if not self._heuristics_enabled():
            return
        for text in texts:
            if await self.check_user_input_similarity_off_loop(text):
                raise _rejection()

    async def _scan_request(self, data: dict[str, object], call_type: str) -> dict[str, object]:
        attachments: Final = request_attachments(data)
        if attachments.unscannable and not self._skips_unscannable_attachments():
            raise _unscannable_rejection(attachments.unscannable)
        await self._reject_injected_texts(attachment.text for attachment in attachments.texts)
        handler: Final = _translation_handler(call_type)
        if handler is None:
            await self._reject_injected_texts(_plain_request_texts(data))
            return data
        return SCANNED_REQUEST.validate_python(
            await handler.process_input_messages(
                data=data, guardrail_to_apply=self, litellm_logging_obj=_logging_obj(data)
            )
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> str | dict[str, object]:
        try:
            return await self._scan_request(data=data, call_type=call_type)
        except HTTPException as exc:
            response: Final = self._response_for_rejection(exc)
            if response is None:
                raise
            return response
        except Exception as exc:
            if self._fails_closed():
                verbose_proxy_logger.error("Prompt injection detection failed and rejected the request: %s", exc)
                raise
            verbose_proxy_logger.exception("Prompt injection detection failed and let the request through: %s", exc)
            return data

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type == "request":
            await self._reject_injected_texts(inputs.get("texts", ()))
        return inputs

    async def _judge_request(self, judge: _PromptInjectionLLMJudge, data: dict[str, object], call_type: str) -> None:
        await judge.reject_injection(_attachment_texts(data))
        handler: Final = _translation_handler(call_type)
        if handler is None:
            await judge.reject_injection(_plain_request_texts(data))
            return
        await handler.process_input_messages(
            data=data, guardrail_to_apply=judge, litellm_logging_obj=_logging_obj(data)
        )

    async def async_moderation_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> None:
        judge: Final = self.llm_judge
        if judge is None:
            return
        try:
            await self._judge_request(judge, data, call_type)
        except HTTPException:
            raise
        except Exception as exc:
            if self._fails_closed():
                verbose_proxy_logger.error("Prompt injection LLM check failed and rejected the request: %s", exc)
                raise
            verbose_proxy_logger.exception("Prompt injection LLM check failed and let the request through: %s", exc)
