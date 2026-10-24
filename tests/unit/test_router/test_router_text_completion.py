from typing import Any
from unittest.mock import MagicMock, patch

import litellm
from litellm import Router


def test_text_completion_uses_deployment_model_name():
    router = Router(
        model_list=[
            {
                "model_name": "my-instruct",
                "litellm_params": {
                    "model": "text-completion-openai/gpt-3.5-turbo-instruct",
                    "api_key": "fake-key",
                },
            }
        ],
        num_retries=0,
    )
    expected_response = MagicMock()

    with patch("litellm.text_completion", return_value=expected_response) as mock_text_completion:
        response = router._text_completion(model="my-instruct", prompt="Hello")

    assert response is expected_response
    assert mock_text_completion.call_args.kwargs["model"] == "text-completion-openai/gpt-3.5-turbo-instruct"


def test_text_completion_falls_back_to_secondary_model():
    router = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "text-completion-openai/primary-model",
                    "api_key": "fake-key",
                },
            },
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": "text-completion-openai/backup-model",
                    "api_key": "fake-key",
                },
            },
        ],
        fallbacks=[{"primary": ["backup"]}],
        num_retries=0,
    )
    expected_response = MagicMock()
    called_models = []

    def fake_text_completion(**kwargs: Any) -> Any:
        called_models.append(kwargs["model"])
        if kwargs["model"] == "text-completion-openai/primary-model":
            raise litellm.InternalServerError(
                message="primary deployment is unavailable",
                llm_provider="text-completion-openai",
                model="primary-model",
            )
        return expected_response

    with patch("litellm.text_completion", side_effect=fake_text_completion):
        response = router.text_completion(model="primary", prompt="Hello")

    assert response is expected_response
    assert called_models == [
        "text-completion-openai/primary-model",
        "text-completion-openai/backup-model",
    ]


def test_text_completion_retries_after_deployment_failure():
    router = Router(
        model_list=[
            {
                "model_name": "instruct",
                "litellm_params": {
                    "model": "text-completion-openai/primary-model",
                    "api_key": "fake-key",
                },
            },
            {
                "model_name": "instruct",
                "litellm_params": {
                    "model": "text-completion-openai/backup-model",
                    "api_key": "fake-key",
                },
            },
        ],
        num_retries=1,
    )
    expected_response = MagicMock()
    called_models = []

    def fake_text_completion(**kwargs: Any) -> Any:
        called_models.append(kwargs["model"])
        if len(called_models) == 1:
            raise litellm.InternalServerError(
                message="first deployment is unavailable",
                llm_provider="text-completion-openai",
                model="primary-model",
            )
        return expected_response

    with patch("litellm.text_completion", side_effect=fake_text_completion):
        response = router.text_completion(model="instruct", prompt="Hello")

    assert response is expected_response
    assert len(called_models) == 2
    assert set(called_models).issubset(
        {
            "text-completion-openai/primary-model",
            "text-completion-openai/backup-model",
        }
    )
