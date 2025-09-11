from functools import wraps

from sentry_sdk import consts
from sentry_sdk.ai.monitoring import record_token_usage
from sentry_sdk.ai.utils import set_data_normalized
from sentry_sdk.consts import SPANDATA

from typing import Any, Iterable, Callable

import sentry_sdk
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.utils import (
    capture_internal_exceptions,
    event_from_exception,
)

try:
    import huggingface_hub.inference._client

    from huggingface_hub import ChatCompletionStreamOutput, TextGenerationOutput
except ImportError:
    raise DidNotEnable("Huggingface not installed")


class HuggingfaceHubIntegration(Integration):
    identifier = "huggingface_hub"
    origin = f"auto.ai.{identifier}"

    def __init__(self, include_prompts=True):
        # type: (HuggingfaceHubIntegration, bool) -> None
        self.include_prompts = include_prompts

    @staticmethod
    def setup_once():
        # type: () -> None
        huggingface_hub.inference._client.InferenceClient.text_generation = (
            _wrap_text_generation(
                huggingface_hub.inference._client.InferenceClient.text_generation
            )
        )


def _capture_exception(exc):
    # type: (Any) -> None
    event, hint = event_from_exception(
        exc,
        client_options=sentry_sdk.get_client().options,
        mechanism={"type": "huggingface_hub", "handled": False},
    )
    sentry_sdk.capture_event(event, hint=hint)


def _wrap_text_generation(f):
    # type: (Callable[..., Any]) -> Callable[..., Any]
    @wraps(f)
    def new_text_generation(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        integration = sentry_sdk.get_client().get_integration(HuggingfaceHubIntegration)
        if integration is None:
            return f(*args, **kwargs)

        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
        elif len(args) >= 2:
            kwargs["prompt"] = args[1]
            prompt = kwargs["prompt"]
            args = (args[0],) + args[2:]
        else:
            # invalid call, let it return error
            return f(*args, **kwargs)

        model = kwargs.get("model")
        streaming = kwargs.get("stream")

        span = sentry_sdk.start_span(
            op=consts.OP.HUGGINGFACE_HUB_CHAT_COMPLETIONS_CREATE,
            name="Text Generation",
            origin=HuggingfaceHubIntegration.origin,
        )
        span.__enter__()
        try:
            res = f(*args, **kwargs)
        except Exception as e:
            _capture_exception(e)
            span.__exit__(None, None, None)
            raise e from None

        with capture_internal_exceptions():
            if should_send_default_pii() and integration.include_prompts:
                set_data_normalized(span, SPANDATA.AI_INPUT_MESSAGES, prompt)

            set_data_normalized(span, SPANDATA.AI_MODEL_ID, model)
            set_data_normalized(span, SPANDATA.AI_STREAMING, streaming)

            if isinstance(res, str):
                if should_send_default_pii() and integration.include_prompts:
                    set_data_normalized(
                        span,
                        "ai.responses",
                        [res],
                    )
                span.__exit__(None, None, None)
                return res

            if isinstance(res, TextGenerationOutput):
                if should_send_default_pii() and integration.include_prompts:
                    set_data_normalized(
                        span,
                        "ai.responses",
                        [res.generated_text],
                    )
                if res.details is not None and res.details.generated_tokens > 0:
                    record_token_usage(span, total_tokens=res.details.generated_tokens)
                span.__exit__(None, None, None)
                return res

            if not isinstance(res, Iterable):
                # we only know how to deal with strings and iterables, ignore
                set_data_normalized(span, "unknown_response", True)
                span.__exit__(None, None, None)
                return res

            if kwargs.get("details", False):
                # res is Iterable[TextGenerationStreamOutput]
                def new_details_iterator():
                    # type: () -> Iterable[ChatCompletionStreamOutput]
                    with capture_internal_exceptions():
                        tokens_used = 0
                        data_buf: list[str] = []
                        for x in res:
                            if hasattr(x, "token") and hasattr(x.token, "text"):
                                data_buf.append(x.token.text)
                            if hasattr(x, "details") and hasattr(
                                x.details, "generated_tokens"
                            ):
                                tokens_used = x.details.generated_tokens
                            yield x
                        if (
                            len(data_buf) > 0
                            and should_send_default_pii()
                            and integration.include_prompts
                        ):
                            set_data_normalized(
                                span, SPANDATA.AI_RESPONSES, "".join(data_buf)
                            )
                        if tokens_used > 0:
                            record_token_usage(span, total_tokens=tokens_used)
                    span.__exit__(None, None, None)

                return new_details_iterator()
            else:
                # res is Iterable[str]

                def new_iterator():
                    # type: () -> Iterable[str]
                    data_buf: list[str] = []
                    with capture_internal_exceptions():
                        for s in res:
                            if isinstance(s, str):
                                data_buf.append(s)
                            yield s
                        if (
                            len(data_buf) > 0
                            and should_send_default_pii()
                            and integration.include_prompts
                        ):
                            set_data_normalized(
                                span, SPANDATA.AI_RESPONSES, "".join(data_buf)
                            )
                        span.__exit__(None, None, None)

                return new_iterator()

    return new_text_generation
