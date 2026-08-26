"""
Helper functions for health check calls.
"""

import base64
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Final, Literal

from litellm.types.utils import LIST_BATCHES_SUPPORTED_PROVIDERS

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import ImageResponse

# Minimal PDF for health checks - base64 encoded 1-page PDF with just "test"
TEST_PDF_URL = "data:application/pdf;base64,JVBERi0xLjQKJeLjz9MKMyAwIG9iago8PC9UeXBlIC9QYWdlCi9QYXJlbnQgMSAwIFIKL01lZGlhQm94IFswIDAgNjEyIDc5Ml0KL0NvbnRlbnRzIDQgMCBSCi9SZXNvdXJjZXMgPDwvRm9udCA8PC9GMSAyIDAgUj4+Pj4+PgplbmRvYmoKNCAwIG9iago8PC9MZW5ndGggNDQ+PgpzdHJlYW0KQlQKL0YxIDI0IFRmCjEwMCA3MDAgVGQKKHRlc3QpIFRqCkVUCmVuZHN0cmVhbQplbmRvYmoKMiAwIG9iago8PC9UeXBlIC9Gb250Ci9TdWJ0eXBlIC9UeXBlMQovQmFzZUZvbnQgL0hlbHZldGljYT4+CmVuZG9iagoxIDAgb2JqCjw8L1R5cGUgL1BhZ2VzCi9LaWRzIFszIDAgUl0KL0NvdW50IDE+PgplbmRvYmoKNSAwIG9iago8PC9UeXBlIC9DYXRhbG9nCi9QYWdlcyAxIDAgUj4+CmVuZG9iagp0cmFpbGVyCjw8L1NpemUgNgovUm9vdCA1IDAgUj4+CnN0YXJ0eHJlZgozMjQKJSVFT0Y="

# Minimal image for health checks - base64 encoded 512x512 blue circle on a white background PNG
TEST_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAAJk0lEQVR42u3VQREAIRADwVWCOmTjBVzwSLorCri6nbkAVBpPACAAAAgAAAIAgAAAIAAACAAAAgCAAAAgAAAIAAACAIAAACAAAAgAAAIAgAAAIAAACAAAAgCAAAAgAAAIAAACACAAAAgAAAIAgAAAIAAACAAAAgCAAAAgAAAIAAACAIAAACAAAAgAAAIAgAAAIAAACAAAAgCAAAAgAAAIAAACACAAAAgAAAIAgAAAIAAACAAAAgCAAAAgAAAIAAACAIAAACAAAAgAAAIAgAAAIAAACAAAAgCAAAAgAAAIAAACAIAAAAgAAAIAgAAAIAAACAAAAgCAAAAgAAAIAAACAIAAACAAAAgAAAIAgAAAIAAACAAAAgCAAAAgAAAIAAACAIAAAAgAAAIAZdY+HgEBgIRr/meeGgGA8EMvDAgAuPh6gACAi68HCAA4+mKAAICjLwYIALj7SoAAgLuvBAgA7r4pAQKAu29KgADg7psSIAA4/SYDCADuvikBAoDTbzKAAOD0mwwgADj9JgMIAE6/yQACgNNvMoAA4PSbDCAAOP0mAwgATr/JAAKA668BIAA4/TKAAOD0mwwgALj+pgEIAE6/yQACgOtvGoAA4PSbDCAAuP6mAQgATr/JAAKA628agADg9JsMIAC4/qYBCACuv2kAAoDrbxqAAOD0mwwgALj+pgEIAK6/aQACgOtvGoAA4PqbBiAArr+ZBiAATr+ZDCAArr+ZBiAArr+ZBiAArr+ZBiAArr+ZBggArr+ZBggArr+ZBggArr+ZBggArr+ZBggArr+ZBggArr+ZBggAAmAmAAKA62+mAQKA62+mAQKA62/m1xYAXH/TAAQA1980AAHA9TcNQAAEwEwAEADX30wDEADX30wDEADX30wDEADX30wDEAABMBMABMD1N9MABMD1N9MABEAAzAQAAXD9zTQAAXD9zTRAABAAMwEQAFx/Mw0QAFx/Mw0QAATATAAEANffTAMEANffTAMEAAEwEwABwPU30wABQADMBEAAXH8z0wABcP3NTAMEQADMTAAEwPU3Mw0QAAEwMwEQANffTAMQAAEwEwAEwPU30wAEQADMBAABcP3NNAABEAAzAUAAXH8zDUAABMBMABAA199MAwQAATATAAHA9TfTAAFAAMwEQABw/c00QAAQADMBEAAEwEwABMD1NzMNEAABMDMBEADX38w0QAAEwMwEQAAEwMwEQABcfzPTAAEQADMTAAEQADMTAAFw/c1MAwRAAMxMAARAAMxMAATA9TczDRAAATATAARAAMwEAAFw/c00AAEQADMBQAAEwEwAEADX30wDBAABMBMAAUAAzARAABAAMwEQAFx/Mw0QAAEwMwEQAAEwMwEQAAEwMwEQANffzDRAAATAzARAAATAzARAAATAzARAAATAzARAAFx/M9MAARAAMxMAARAAMxMAARAAMxMAARAAMxMAAXD9zUwDBEAAzEwABEAAzAQAARAAMwFAAATATAAQAAEwEwABQADMBEAAEAAzARAAXH8zDRAAATAzARAAATAzARAAATAzARAAATAzARAAATAzARAAATAzARAAATAzARAAATAzARAAATAzARAAATAzARAAATAzARAAATAzARAAATAzARAAATAzARAAATAzARAAATAzARAA19/MNEAANMDM9UcABMBMABAAATATAAHwBAJgJgACgACYCYAAIABmAiAA+IvMBEAABMDMBEAABMDMBEAABMDMBEAABMDMBEAABMDMBEAABMDMBEAABMDMBEAABMDMBEAABMDMBEAABMDMBEAANMDMXH8BEAAzEwABEAAzEwABEAAzEwABEAAzEwABEAAzEwABEAAzAUAABMBMABAADTBz/REAATATAAFAAMwEQAAQADMBEAAEwEwABAANMHP9BUAAzEwABEAAzEwABEAAzEwABEAAzEwABEADzMz1FwABMDMBEAABMDMBEAABMDMBEAANMDPXXwAEwMwEQAAEwMwEQAAEwMwEQAA0wMxcfwEQADMTAAEQADMBQAA0wMz1RwAEwEwAEAABMBMABEADzFx/AUAAzARAABAAMwEQADTAzPUXAATATAAEAAEwEwABQAPMXH8BEAAzEwABEAAzEwAB0AAzc/0FQADMTAAEQAPMzPUXAAEwMwEQAAEwMwEQAA0wM9dfAATAzARAADTAzFx/ARAAMwFAADTAzPVHAATATAAQAA0wc/0RAAEwEwAEQAPMXH8EQADMBAAB0AAz118AEAAzARAANMDM9RcABMBMAAQADTBz/QUAATATAAFAA8xcfwFAA8xcfwFAAMwEQADQADPXXwAEwMwEQAA0wMxcfwHQADNz/QVAAMxMAARAA8xcfwRAA8xcfwRAAMwEAAHQADPXHwHQADPXHwEQADMBQAA0wMz1RwA0wMz1RwAEwEwAEAANMHP9BQANMHP9BQANMHP9BQANMHP9BQABMBMAAUADzFx/AUADzFx/AUADzFx/AUADzFx/AUADzFx/AUADzPVHABAAEwAEAA0w1x8BQAPM9UcA0ABz/REANMBcfwQADTDXHwHQADPXHwHQADPXHwHQADPXHwHQADPXHwHQADPXHwGQATOnHwHQADPXHwHQADPXXwDQADPXXwDQADPXXwDQADPXXwCQAXP6EQA0wFx/BAANMNcfAUADzPVHAJABc/oRADTAXH8EABkwpx8BQAPM9UcAkAFz+hEANMBcfwQAGTCnHwFAA8z1RwCQAXP6EQBkwJx+BAANMNcfAUAGzOlHAJABc/oRAGTA6QcBQAacfhAAZMDpRwBABpx+BABkwOlHAEAGnH4EAJTA3UcAQAacfgQAlMDdRwBACdx9BACUwN1HAEAJ3H0EAJTA3UcAQAwcfQQAmmLgsyIA0NIDHw4BgJYe+DQIAISHwVMjAJDQDI+AAAAgAAAIAAACAIAAACAAAAgAAAIAgAAAIAAACAAAAgCAAAAgAAAIAAACAIAAACAAAAgAAAIAgAAAIAAACACAAAAgAAAIAAACAIAAACAAAAgAAAIAgAAAIAAACAAAAgCAAAAgAAAIAAACAIAAACAAAAgAAAIAgAAAIAAACAAAAgAgAAAIAAACAIAAACAAAAgAAAIAgAAAIAAACAAAAgCAAAAgAAAIAAACAIAAACAAAAgAAAIAgAAAIAAACAAAAgAgAAAIAAACAIAAACAAAAgAAAIAgAAAIAAACAAAAgCAAAAgAAAIAAACAIAAACAAAAgAAAIAgAAAIAAACAAAAgCAAAAIAABNHpialFcmLajuAAAAAElFTkSuQmCC"


IMAGE_EDIT_HEALTH_CHECK_PROMPT: Final = (
    "Add a small yellow star in the top right corner of this simple drawing of a blue circle on a white background"
)


def get_image_file_for_health_check() -> bytes:
    """Return the image used for health checks."""
    return base64.b64decode(TEST_IMAGE_BASE64)


class HealthCheckHelpers:
    @staticmethod
    async def ahealth_check_wildcard_models(
        model: str,
        custom_llm_provider: str,
        model_params: dict,
        litellm_logging_obj: "Logging",
    ) -> dict:
        from litellm import acompletion
        from litellm.litellm_core_utils.llm_request_utils import (
            pick_cheapest_chat_models_from_llm_provider,
        )

        # this is a wildcard model, we need to pick a random model from the provider
        cheapest_models = pick_cheapest_chat_models_from_llm_provider(custom_llm_provider=custom_llm_provider, n=3)
        if len(cheapest_models) == 0:
            raise Exception(
                f"Unable to health check wildcard model for provider {custom_llm_provider}. Add a model on your config.yaml or contribute here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
            )
        if len(cheapest_models) > 1:
            fallback_models = cheapest_models[1:]  # Pick the last 2 models from the shuffled list
        else:
            fallback_models = None
        model_params["model"] = cheapest_models[0]
        model_params["litellm_logging_obj"] = litellm_logging_obj
        model_params["fallbacks"] = fallback_models
        model_params["max_tokens"] = model_params.get("max_tokens", 16)  # GPT-5 models require max_output_tokens >= 16
        await acompletion(**model_params)
        return {}

    @staticmethod
    def _update_model_params_with_health_check_tracking_information(
        model_params: dict,
    ) -> dict:
        """
        Updates the health check model params with tracking information.

        The following is added at this stage:
            1. `tags`: This helps identify health check calls in the DB.
            2. `user_api_key_auth`: This helps identify health check calls in the DB.
                We need this since the DB requires an API Key to track a log in the SpendLogs Table
        """
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

        _metadata_variable_name: Final = "litellm_metadata"
        litellm_metadata: Final = HealthCheckHelpers._get_metadata_for_health_check_call()
        model_params[_metadata_variable_name] = litellm_metadata
        model_params = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
            data=model_params,
            user_api_key_dict=UserAPIKeyAuth.get_litellm_internal_health_check_user_api_key_auth(),
            _metadata_variable_name=_metadata_variable_name,
        )
        return model_params

    @staticmethod
    def _get_metadata_for_health_check_call():
        """
        Returns the metadata for the health check call.
        """
        from litellm.constants import LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME

        return {
            "tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME],
        }

    @staticmethod
    async def _batch_health_check(
        custom_llm_provider: str,
        model_params: dict,
        filtered_model_params: dict,
    ) -> dict:
        """
        Health check for batch mode.

        Calls list_batches for providers that support it (openai, hosted_vllm, azure,
        vertex_ai). For all other providers (e.g. bedrock) the batch API surface doesn't
        include list_batches, so we fall back to acompletion to verify connectivity and
        credential validity instead.
        """
        import litellm

        logging_obj: Final = filtered_model_params.get("litellm_logging_obj")
        if logging_obj is not None:
            api_base: Final = filtered_model_params.get("api_base")
            logging_obj.update_from_kwargs(
                kwargs=filtered_model_params,
                model=filtered_model_params.get("model"),
                user=None,
                optional_params={},
                litellm_params={"api_base": api_base} if api_base else None,
            )

        if custom_llm_provider in LIST_BATCHES_SUPPORTED_PROVIDERS:
            return await litellm.alist_batches(**filtered_model_params)
        else:
            return await litellm.acompletion(**model_params)

    @staticmethod
    async def _image_edit_health_check(edit_request: Callable[[], Awaitable["ImageResponse"]]) -> "ImageResponse":
        import litellm

        try:
            return await edit_request()
        except litellm.BadRequestError as e:
            if isinstance(e, litellm.ContentPolicyViolationError) or "moderation_blocked" in str(e):
                return litellm.ImageResponse()
            raise

    @staticmethod
    def get_mode_handlers(
        model: str,
        custom_llm_provider: str,
        model_params: dict,
        prompt: str | None = None,
        input: list | None = None,
    ) -> dict[
        Literal[
            "chat",
            "completion",
            "embedding",
            "audio_speech",
            "audio_transcription",
            "image_generation",
            "image_edit",
            "video_generation",
            "rerank",
            "realtime",
            "batch",
            "responses",
            "ocr",
        ],
        Callable,
    ]:
        """
        Returns a dictionary of mode handlers for health check calls.

        Mode Handlers are Callables that need to be run for execution of the health check call.

        Args:
            model: The model name
            custom_llm_provider: The LLM provider
            model_params: The model parameters
            prompt: Optional prompt for health check
            input: Optional input for health check

        Returns:
            Dictionary mapping mode names to their handler functions
        """
        import litellm
        from litellm.litellm_core_utils.audio_utils.utils import (
            get_audio_file_for_health_check,
        )
        from litellm.litellm_core_utils.health_check_utils import _filter_model_params
        from litellm.realtime_api.main import _realtime_health_check

        return {
            "chat": lambda: litellm.acompletion(
                **model_params,
            ),
            "completion": lambda: litellm.atext_completion(
                **_filter_model_params(model_params=model_params),
                prompt=prompt or "test",
            ),
            "embedding": lambda: litellm.aembedding(
                **_filter_model_params(model_params=model_params),
                input=input or ["test"],
            ),
            "audio_speech": lambda: litellm.aspeech(
                **{
                    **_filter_model_params(model_params=model_params),
                    **({"voice": "alloy"} if "voice" not in _filter_model_params(model_params=model_params) else {}),
                },
                input=prompt or "test",
            ),
            "audio_transcription": lambda: litellm.atranscription(
                **_filter_model_params(model_params=model_params),
                file=get_audio_file_for_health_check(),
            ),
            "image_generation": lambda: litellm.aimage_generation(
                **_filter_model_params(model_params=model_params),
                prompt=prompt,
            ),
            "image_edit": lambda: HealthCheckHelpers._image_edit_health_check(
                edit_request=lambda: litellm.aimage_edit(
                    **_filter_model_params(model_params=model_params),
                    image=get_image_file_for_health_check(),
                    prompt=IMAGE_EDIT_HEALTH_CHECK_PROMPT,
                ),
            ),
            "video_generation": lambda: litellm.avideo_generation(
                **_filter_model_params(model_params=model_params),
                prompt=prompt or "test video generation",
            ),
            "rerank": lambda: litellm.arerank(
                **_filter_model_params(model_params=model_params),
                query=prompt or "",
                documents=["my sample text"],
            ),
            "realtime": lambda: _realtime_health_check(
                model=model,
                custom_llm_provider=custom_llm_provider,
                api_base=model_params.get("api_base", None),
                api_key=model_params.get("api_key", None),
                api_version=model_params.get("api_version", None),
                model_params=model_params,
            ),
            "batch": lambda: HealthCheckHelpers._batch_health_check(
                custom_llm_provider=custom_llm_provider,
                model_params=model_params,
                filtered_model_params=_filter_model_params(model_params=model_params),
            ),
            "responses": lambda: litellm.aresponses(
                **_filter_model_params(model_params=model_params),
                input=prompt or "test",
            ),
            "ocr": lambda: litellm.aocr(
                **_filter_model_params(model_params=model_params),
                document={
                    "type": "document_url",
                    "document_url": TEST_PDF_URL,
                },
            ),
        }
