import asyncio
import json
import os
from typing import Final
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import litellm
import pytest
import respx
from litellm import Router
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


@pytest.fixture
def model_list():
    return [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {
                "model": "gpt-5-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "gpt-5.5",
            "litellm_params": {
                "model": "gpt-5.5",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "gpt-image-1",
            "litellm_params": {
                "model": "gpt-image-1",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "cohere-rerank",
            "litellm_params": {
                "model": "cohere/rerank-english-v3.0",
                "api_key": os.getenv("COHERE_API_KEY"),
            },
        },
        {
            "model_name": "claude-sonnet-4-5-20250929",
            "litellm_params": {
                "model": "gpt-5-mini",
                "mock_response": "hi this is macintosh.",
            },
        },
    ]


@pytest.mark.asyncio
async def test_aspeech_fallbacks_on_deployment_failure():
    router = Router(
        model_list=[
            {
                "model_name": "tts-main",
                "litellm_params": {"model": "openai/tts-1", "api_key": "fake-key"},
            },
            {
                "model_name": "tts-backup",
                "litellm_params": {"model": "openai/tts-1-hd", "api_key": "fake-key"},
            },
        ],
        fallbacks=[{"tts-main": ["tts-backup"]}],
        num_retries=0,
    )

    called_models = []

    async def mock_aspeech(*args, **kwargs):
        called_models.append(kwargs["model"])
        if kwargs["model"] == "openai/tts-1":
            raise litellm.InternalServerError(
                message="deployment down",
                llm_provider="openai",
                model="tts-1",
            )
        return MagicMock()

    with patch("litellm.aspeech", side_effect=mock_aspeech):
        response = await router.aspeech(
            model="tts-main",
            input="the quick brown fox jumped over the lazy dogs",
            voice="alloy",
        )

    assert response is not None
    assert called_models == ["openai/tts-1", "openai/tts-1-hd"]


@pytest.mark.asyncio
async def test_aspeech_success_returns_response():
    router = Router(
        model_list=[
            {
                "model_name": "tts",
                "litellm_params": {"model": "openai/tts-1", "api_key": "fake-key"},
            },
        ]
    )

    mock_response = MagicMock()
    with patch("litellm.aspeech", return_value=mock_response) as mock_aspeech:
        response = await router.aspeech(
            model="tts",
            input="the quick brown fox jumped over the lazy dogs",
            voice="alloy",
        )

    assert response is mock_response
    mock_aspeech.assert_called_once()
    assert mock_aspeech.call_args.kwargs["model"] == "openai/tts-1"


@pytest.mark.asyncio
async def test_aspeech_sets_deployment_metadata():
    router = Router(
        model_list=[
            {
                "model_name": "tts",
                "litellm_params": {"model": "openai/tts-1", "api_key": "fake-key"},
            },
        ]
    )

    mock_response = MagicMock()
    with patch("litellm.aspeech", return_value=mock_response) as mock_aspeech:
        response = await router.aspeech(
            model="tts",
            input="the quick brown fox jumped over the lazy dogs",
            voice="alloy",
        )

    assert response is mock_response
    metadata = mock_aspeech.call_args.kwargs["metadata"]
    assert metadata["deployment"] == "openai/tts-1"
    assert metadata["deployment_model_name"] == "tts"
    assert metadata["model_info"]["id"] is not None


@pytest.mark.asyncio()
async def test_moderation_endpoint_with_api_base():
    """
    Test that the moderation endpoint respects api_base configuration
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    custom_api_base = "https://us.api.openai.com/v1"

    router = Router(
        model_list=[
            {
                "model_name": "openai/omni-moderation-latest",
                "litellm_params": {
                    "model": "openai/omni-moderation-latest",
                    "api_base": custom_api_base,
                    "api_key": "test-key",
                },
            },
        ]
    )

    # Mock the OpenAI client to verify api_base is passed
    with patch("litellm.main.openai_chat_completions.get_openai_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "modr-123",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": False,
                    "categories": {},
                    "category_scores": {},
                    "category_applied_input_types": {},
                }
            ],
        }
        mock_client.moderations.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        response = await router.amoderation(model="openai/omni-moderation-latest", input="hello this is a test")

        # Verify that get_openai_client was called with the custom api_base
        mock_get_client.assert_called()
        call_kwargs = mock_get_client.call_args.kwargs
        assert call_kwargs.get("api_base") == custom_api_base, (
            f"Expected api_base to be {custom_api_base}, but got {call_kwargs.get('api_base')}"
        )

        print(f"✓ Moderation endpoint correctly uses api_base: {custom_api_base}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_aaaaatext_completion_endpoint(model_list, sync_mode):
    router = Router(model_list=model_list)

    if sync_mode:
        response = router.text_completion(
            model="gpt-5-mini",
            prompt="Hello, how are you?",
            mock_response="I'm fine, thank you!",
        )
    else:
        response = await router.atext_completion(
            model="gpt-5-mini",
            prompt="Hello, how are you?",
            mock_response="I'm fine, thank you!",
        )

    assert response.choices[0].text == "I'm fine, thank you!"


@pytest.mark.asyncio
async def test_router_with_empty_choices(model_list):
    """
    https://github.com/BerriAI/litellm/issues/8306
    """
    router = Router(model_list=model_list)
    mock_response = litellm.ModelResponse(
        choices=[],
        usage=litellm.Usage(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        ),
        model="gpt-5-mini",
        object="chat.completion",
        created=1723081200,
    ).model_dump()
    response = await router.acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response=mock_response,
    )
    assert response is not None


@pytest.mark.asyncio
async def test_aadapter_completion_calls_the_adapter_with_the_deployments_params():
    mock_response = {"id": "adapter_resp_123", "object": "adapter.completion"}

    with patch("litellm.aadapter_completion", new_callable=AsyncMock) as mock_adapter_completion:
        mock_adapter_completion.return_value = mock_response

        router = Router(
            model_list=[
                {
                    "model_name": "test-adapter-model",
                    "litellm_params": {
                        "model": "anthropic/test-model",
                        "api_key": "fake-api-key",
                    },
                }
            ]
        )

        response = await router.aadapter_completion(
            adapter_id="test-adapter-id",
            model="test-adapter-model",
            prompt="This is a test prompt",
            max_tokens=100,
        )

    assert response["id"] == "adapter_resp_123"
    mock_adapter_completion.assert_called_once()
    call_kwargs = mock_adapter_completion.call_args.kwargs
    assert call_kwargs["adapter_id"] == "test-adapter-id"
    assert call_kwargs["model"] == "anthropic/test-model"
    assert call_kwargs["prompt"] == "This is a test prompt"
    assert call_kwargs["max_tokens"] == 100
    assert call_kwargs["api_key"] == "fake-api-key"
    assert call_kwargs["metadata"]["model_group"] == "test-adapter-model"


def test_initialize_router_endpoints():
    """
    Test that initialize_router_endpoints correctly sets up all router endpoints
    """

    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "anthropic/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    router.initialize_router_endpoints()

    assert hasattr(router, "amoderation")
    assert hasattr(router, "aanthropic_messages")
    assert hasattr(router, "aresponses")
    assert hasattr(router, "responses")
    assert hasattr(router, "aget_responses")
    assert hasattr(router, "adelete_responses")

    assert callable(router.amoderation)
    assert callable(router.aanthropic_messages)
    assert callable(router.aresponses)
    assert callable(router.responses)
    assert callable(router.aget_responses)
    assert callable(router.adelete_responses)


@pytest.mark.asyncio
async def test_avector_store_create_sends_only_the_callers_metadata_to_the_provider():
    def echo_vector_store(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "vs_echo",
                "object": "vector_store",
                "created_at": 0,
                "name": body["name"],
                "usage_bytes": 0,
                "file_counts": {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
                "status": "completed",
                "metadata": body.get("metadata"),
            },
        )

    router: Final = Router(
        model_list=[{"model_name": "gpt-5", "litellm_params": {"model": "openai/gpt-5", "api_key": "sk-test"}}]
    )

    response: Final = await router.avector_store_create(
        model="gpt-5",
        name="x",
        metadata={"team": "llmproxy"},
        client=AsyncHTTPHandler(transport=httpx.MockTransport(echo_vector_store)),
    )

    assert response["id"] == "vs_echo"
    assert response["metadata"] == {"team": "llmproxy"}


def test_apply_default_settings():
    """
    apply_default_settings initializes the default pre-call checks and leaves the optional
    callbacks the router already has in place.
    """

    router = Router()
    initial_optional_callbacks = router.optional_callbacks

    result = router.apply_default_settings()

    assert result is None

    assert router.optional_callbacks == initial_optional_callbacks

    router_with_callbacks = Router()
    mock_callback = MagicMock()
    router_with_callbacks.optional_callbacks = [mock_callback]

    result = router_with_callbacks.apply_default_settings()

    assert result is None

    assert mock_callback in router_with_callbacks.optional_callbacks


def test_initialize_core_endpoints():
    """
    Test that _initialize_core_endpoints correctly sets up all core router endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "anthropic/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    core_endpoints = [
        "amoderation",
        "aanthropic_messages",
        "agenerate_content",
        "aadapter_generate_content",
        "aresponses",
        "afile_delete",
        "afile_content",
        "responses",
        "aget_responses",
        "acancel_responses",
        "adelete_responses",
        "alist_input_items",
        "acreate_fine_tuning_job",
        "acancel_fine_tuning_job",
        "alist_fine_tuning_jobs",
        "aretrieve_fine_tuning_job",
        "afile_list",
        "aimage_edit",
        "allm_passthrough_route",
    ]

    for endpoint in core_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_specialized_endpoints():
    """
    Test that _initialize_specialized_endpoints correctly sets up specialized endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    specialized_endpoints = [
        "avector_store_search",
        "avector_store_create",
        "vector_store_search",
        "vector_store_create",
        "agenerate_content",
        "generate_content",
        "agenerate_content_stream",
        "generate_content_stream",
        "aocr",
        "ocr",
        "asearch",
        "search",
        "avideo_generation",
        "video_generation",
        "avideo_list",
        "video_list",
        "avideo_status",
        "video_status",
        "avideo_content",
        "video_content",
        "avideo_remix",
        "video_remix",
        "acreate_container",
        "create_container",
        "alist_containers",
        "list_containers",
        "aretrieve_container",
        "retrieve_container",
        "adelete_container",
        "delete_container",
        "acreate_skill",
        "alist_skills",
        "aget_skill",
        "adelete_skill",
    ]

    for endpoint in specialized_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_vector_store_endpoints():
    """
    Test that _initialize_vector_store_endpoints correctly sets up vector store endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    vector_store_endpoints = [
        "avector_store_search",
        "avector_store_create",
        "vector_store_search",
        "vector_store_create",
    ]

    for endpoint in vector_store_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_vector_store_file_endpoints():
    """
    Test that _initialize_vector_store_file_endpoints correctly sets up vector store file endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    vector_store_file_endpoints = [
        "avector_store_file_create",
        "vector_store_file_create",
        "avector_store_file_list",
        "vector_store_file_list",
        "avector_store_file_retrieve",
        "vector_store_file_retrieve",
        "avector_store_file_content",
        "vector_store_file_content",
        "avector_store_file_update",
        "vector_store_file_update",
        "avector_store_file_delete",
        "vector_store_file_delete",
    ]

    for endpoint in vector_store_file_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_google_genai_endpoints():
    """
    Test that _initialize_google_genai_endpoints correctly sets up Google GenAI endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    google_genai_endpoints = [
        "agenerate_content",
        "generate_content",
        "agenerate_content_stream",
        "generate_content_stream",
    ]

    for endpoint in google_genai_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_ocr_search_endpoints():
    """
    Test that _initialize_ocr_search_endpoints correctly sets up OCR and search endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    ocr_search_endpoints = [
        "aocr",
        "ocr",
        "asearch",
        "search",
    ]

    for endpoint in ocr_search_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_video_endpoints():
    """
    Test that _initialize_video_endpoints correctly sets up video endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    video_endpoints = [
        "avideo_generation",
        "video_generation",
        "avideo_list",
        "video_list",
        "avideo_status",
        "video_status",
        "avideo_content",
        "video_content",
        "avideo_remix",
        "video_remix",
    ]

    for endpoint in video_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_container_endpoints():
    """
    Test that _initialize_container_endpoints correctly sets up container endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    container_endpoints = [
        "acreate_container",
        "create_container",
        "alist_containers",
        "list_containers",
        "aretrieve_container",
        "retrieve_container",
        "adelete_container",
        "delete_container",
    ]

    for endpoint in container_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


def test_initialize_skills_endpoints():
    """
    Test that _initialize_skills_endpoints correctly sets up skills endpoints.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "anthropic/test-model",
                    "api_key": "fake-api-key",
                },
            }
        ]
    )

    skills_endpoints = [
        "acreate_skill",
        "alist_skills",
        "aget_skill",
        "adelete_skill",
    ]

    for endpoint in skills_endpoints:
        assert hasattr(router, endpoint)
        assert callable(getattr(router, endpoint))


_CONTAINER_FILE: Final = {
    "id": "cfile_xyz",
    "object": "container.file",
    "container_id": "cfile_upstream_abc",
    "created_at": 0,
    "bytes": 1,
    "path": "/a",
    "source": "user",
}


async def _container_file_request(router: Router, container_id: str) -> httpx.Request:
    with respx.mock(assert_all_called=False) as upstream:
        route = upstream.route().mock(return_value=httpx.Response(200, json=_CONTAINER_FILE))
        await router.aretrieve_container_file(container_id=container_id, file_id="cfile_xyz")
    assert route.call_count == 1
    return route.calls.last.request


@pytest.mark.asyncio
async def test_containers_managed_id_routes_to_its_deployment_with_its_credentials(monkeypatch):
    """
    Managed ``cntr_`` IDs embed ``model_id``; the router decodes it so the deployment's
    credentials and api_base apply, and the upstream gets the unwrapped container ID.
    """
    from litellm.responses.utils import ResponsesAPIRequestUtils

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router = Router(
        model_list=[
            {
                "model_name": "azure-router-model",
                "litellm_params": {
                    "model": "azure/gpt-5.5",
                    "api_key": "fake-key",
                    "api_base": "https://westus.api.cognitive.microsoft.com",
                },
            }
        ]
    )
    managed_id = ResponsesAPIRequestUtils.build_container_id(
        custom_llm_provider="azure",
        model_id="azure-router-model",
        container_id="cfile_upstream_abc",
    )

    request = await _container_file_request(router, managed_id)

    assert request.url.host == "westus.api.cognitive.microsoft.com"
    assert request.url.path.endswith("/containers/cfile_upstream_abc/files/cfile_xyz")
    assert request.headers["api-key"] == "fake-key"


@pytest.mark.asyncio
async def test_containers_managed_id_without_model_id_unwraps(monkeypatch):
    """
    Managed ``cntr_`` IDs may be encoded with an empty ``model_id`` (e.g. when a
    streaming response had no router metadata). The router must still unwrap the
    managed ID before calling the upstream provider — otherwise the raw
    ``cntr_...`` token leaks downstream and the provider rejects it.
    """
    from litellm.responses.utils import ResponsesAPIRequestUtils

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-global")
    router = Router(model_list=[])
    managed_id = ResponsesAPIRequestUtils.build_container_id(
        custom_llm_provider="openai",
        model_id=None,
        container_id="cfile_upstream_abc",
    )

    request = await _container_file_request(router, managed_id)

    assert request.url.path == "/v1/containers/cfile_upstream_abc/files/cfile_xyz"


@pytest.mark.asyncio
async def test_containers_create_with_model_uses_deployment_credentials(monkeypatch):
    """
    ``POST /v1/containers`` carries no container ID, so a ``model`` in the request
    body is the only way to pick a deployment. The upstream call must receive that
    deployment's ``api_key``/``api_base`` instead of falling back to the global
    ``OPENAI_API_KEY`` (which may be unset on the proxy).
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("litellm.containers.acreate_container", new_callable=AsyncMock) as create:
        create.return_value = {"id": "cntr_test", "name": "Test Container"}
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-5.4",
                    "litellm_params": {
                        "model": "openai/gpt-5.4",
                        "api_key": "sk-model-list-key",
                        "api_base": "https://custom.openai.example/v1",
                    },
                }
            ]
        )
        await router.acreate_container(name="Test Container", model="gpt-5.4")

    create.assert_called_once()
    call_kw = create.call_args.kwargs
    assert call_kw["api_key"] == "sk-model-list-key"
    assert call_kw["api_base"] == "https://custom.openai.example/v1"
    assert call_kw["model"] == "openai/gpt-5.4"
    assert call_kw["name"] == "Test Container"


@pytest.mark.asyncio
async def test_containers_create_without_model_calls_directly():
    """
    Without ``model``, create must keep calling the handler directly with global provider credentials.
    """
    with patch("litellm.containers.acreate_container", new_callable=AsyncMock) as create:
        create.return_value = {"id": "cntr_test"}
        router = Router(model_list=[])
        await router.acreate_container(name="Test Container")

    create.assert_called_once_with(name="Test Container")


@pytest.mark.asyncio
async def test_containers_create_with_unknown_model_passes_through(monkeypatch):
    """
    A ``model`` that names no configured deployment must not turn into a 400. The call
    falls through to the handler with the caller's model and no injected deployment
    credentials, matching the behaviour before model-based routing existed.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("litellm.containers.acreate_container", new_callable=AsyncMock) as create:
        create.return_value = {"id": "cntr_test"}
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-5.4",
                    "litellm_params": {"model": "openai/gpt-5.4", "api_key": "sk-model-list-key"},
                }
            ]
        )
        await router.acreate_container(name="Test Container", model="does-not-exist")

    create.assert_called_once()
    call_kw = create.call_args.kwargs
    assert call_kw["model"] == "does-not-exist"
    assert call_kw["name"] == "Test Container"
    assert "api_key" not in call_kw
    assert "api_base" not in call_kw


def test_router_model_group_encrypted_content_affinity_callback_registration():
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
        DeploymentAffinityCheck,
    )
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    model_group = "openai.gpt-5.1-codex"
    model_group_affinity_config = {
        model_group: ["encrypted_content_affinity"],
    }
    router = Router(
        model_list=[
            {
                "model_name": model_group,
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key",
                },
            }
        ],
        model_group_affinity_config=model_group_affinity_config,
        num_retries=0,
    )

    try:
        callbacks = router.optional_callbacks or []
        encrypted_content_callbacks = [cb for cb in callbacks if isinstance(cb, EncryptedContentAffinityCheck)]
        deployment_callback = next(cb for cb in callbacks if isinstance(cb, DeploymentAffinityCheck))
        assert len(encrypted_content_callbacks) == 1
        assert encrypted_content_callbacks[0].enable_global_affinity is False
        assert encrypted_content_callbacks[0].model_group_affinity_config == model_group_affinity_config
        assert callbacks.index(encrypted_content_callbacks[0]) < callbacks.index(deployment_callback)
    finally:
        router.discard()
