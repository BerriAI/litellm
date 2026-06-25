import pytest

from litellm import Router


@pytest.fixture
def model_list():
    return [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {
                "model": "gpt-5-mini",
                "api_key": "test-key",
            },
        },
        {
            "model_name": "gpt-image-1",
            "litellm_params": {
                "model": "gpt-image-1",
                "api_key": "test-key",
            },
        },
    ]


def test_sync_embedding_forwards_request_kwargs_to_deployment_selection(model_list):
    router = Router(model_list=model_list)
    captured_kwargs = {}
    metadata = {
        "user_api_key": "hashed-key",
        "user_api_key_team_id": "team-1",
        "user_api_key_auth": object(),
    }

    def fake_get_available_deployment(**kwargs):
        captured_kwargs.update(kwargs)
        raise RuntimeError("stop before provider call")

    router.get_available_deployment = fake_get_available_deployment  # type: ignore

    with pytest.raises(RuntimeError, match="stop before provider call"):
        router._embedding(
            input="hello",
            model="gpt-5-mini",
            metadata=metadata,
        )

    assert captured_kwargs["request_kwargs"]["metadata"] is metadata


def test_sync_image_generation_forwards_request_kwargs_to_deployment_selection(
    model_list,
):
    router = Router(model_list=model_list)
    captured_kwargs = {}
    metadata = {
        "user_api_key": "hashed-key",
        "user_api_key_team_id": "team-1",
        "user_api_key_auth": object(),
    }

    def fake_get_available_deployment(**kwargs):
        captured_kwargs.update(kwargs)
        raise RuntimeError("stop before provider call")

    router.get_available_deployment = fake_get_available_deployment  # type: ignore

    with pytest.raises(RuntimeError, match="stop before provider call"):
        router._image_generation(
            prompt="draw a square",
            model="gpt-image-1",
            metadata=metadata,
        )

    assert captured_kwargs["request_kwargs"]["metadata"] is metadata
