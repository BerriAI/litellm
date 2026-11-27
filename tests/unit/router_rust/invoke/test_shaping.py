from types import MappingProxyType
from typing import cast

import litellm
from litellm.router_rust.invoke.shaping import Shaping, update_kwargs_with_deployment


def _shaping() -> Shaping:
    return Shaping(
        default_litellm_params=MappingProxyType({}),
        timeout=None,
        explicit_timeout=None,
        request_timeout=None,
        stream_timeout=None,
        cache_responses=False,
    )


def test_credential_tag_added_by_default() -> None:
    shaping = _shaping()
    deployment = {
        "model_name": "gpt-4o",
        "litellm_params": {
            "model": "gpt-4o",
            "api_key": "sk-test",
            "litellm_credential_name": "corp-openai-key",
        },
        "model_info": {"id": "gpt-4o-1"},
    }
    kwargs: dict[str, object] = {"model": "gpt-4o"}

    litellm.disable_credential_tagging = False
    update_kwargs_with_deployment(shaping, deployment, kwargs, usage=None)

    metadata = cast(dict[str, object], kwargs.get("metadata", {}))
    tags = metadata.get("tags")
    assert tags == ["Credential: corp-openai-key"]


def test_credential_tag_not_added_when_disabled() -> None:
    shaping = _shaping()
    deployment = {
        "model_name": "gpt-4o",
        "litellm_params": {
            "model": "gpt-4o",
            "api_key": "sk-test",
            "litellm_credential_name": "corp-openai-key",
        },
        "model_info": {"id": "gpt-4o-1"},
    }
    kwargs: dict[str, object] = {"model": "gpt-4o"}

    litellm.disable_credential_tagging = True
    try:
        update_kwargs_with_deployment(shaping, deployment, kwargs, usage=None)
    finally:
        litellm.disable_credential_tagging = False

    metadata = cast(dict[str, object], kwargs.get("metadata", {}))
    tags = metadata.get("tags")
    assert tags is None


def test_existing_tags_preserved_when_disabled() -> None:
    shaping = _shaping()
    deployment = {
        "model_name": "gpt-4o",
        "litellm_params": {
            "model": "gpt-4o",
            "api_key": "sk-test",
            "litellm_credential_name": "corp-openai-key",
        },
        "model_info": {"id": "gpt-4o-1"},
    }
    kwargs: dict[str, object] = {
        "model": "gpt-4o",
        "metadata": {"tags": ["production", "team-billing"]},
    }

    litellm.disable_credential_tagging = True
    try:
        update_kwargs_with_deployment(shaping, deployment, kwargs, usage=None)
    finally:
        litellm.disable_credential_tagging = False

    metadata = cast(dict[str, object], kwargs.get("metadata", {}))
    tags = metadata.get("tags")
    assert tags == ["production", "team-billing"]
