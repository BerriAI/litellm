import pytest
from litellm import Router


def test_router_ocr_capability_gate_skips_chat_fallbacks() -> None:
    router = Router(
        model_list=[
            {
                "model_name": "document-ocr",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {"id": "ocr", "mode": "ocr"},
            },
            {
                "model_name": "document-ocr",
                "litellm_params": {"model": "openai/gpt-4o-mini"},
                "model_info": {"id": "chat", "mode": "chat"},
            },
        ],
        enable_pre_call_checks=True,
    )

    async def aocr(*args, **kwargs):
        return None

    async def chat(*args, **kwargs):
        return None

    deployments = router.get_model_list(model_name="document-ocr")
    chat_deployments = router._pre_call_checks(
        model="document-ocr",
        healthy_deployments=deployments,
        messages=None,
        input=None,
        request_kwargs={"original_function": chat},
    )
    assert [deployment["model_info"]["id"] for deployment in chat_deployments] == ["ocr", "chat"]

    ocr_deployments = router._pre_call_checks(
        model="document-ocr",
        healthy_deployments=deployments,
        messages=None,
        input=None,
        request_kwargs={"original_function": aocr},
    )
    assert [deployment["model_info"]["id"] for deployment in ocr_deployments] == ["ocr"]
