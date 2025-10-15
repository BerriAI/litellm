#### What this tests ####
# This tests utils used by llm router -> like llmrouter.get_settings()

import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params
from litellm.types.router import ModelInfo
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock, AsyncMock

load_dotenv()


def test_returned_settings():
    # this tests if the router raises an exception when invalid params are set
    # in this test both deployments have bad keys - Keep this test. It validates if the router raises the most recent exception
    litellm.set_verbose = True
    import openai

    try:
        print("testing if router raises an exception")
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-nano",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  #
                    "model": "gpt-3.5-turbo",
                    "api_key": "bad-key",
                },
                "tpm": 240000,
                "rpm": 1800,
            },
        ]
        router = Router(
            model_list=model_list,
            redis_host=os.getenv("REDIS_HOST"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_port=int(os.getenv("REDIS_PORT")),
            routing_strategy="latency-based-routing",
            routing_strategy_args={"ttl": 10},
            set_verbose=False,
            num_retries=3,
            retry_after=5,
            allowed_fails=1,
            cooldown_time=30,
        )  # type: ignore

        settings = router.get_settings()
        print(settings)

        """
        routing_strategy: "simple-shuffle"
        routing_strategy_args: {"ttl": 10} # Average the last 10 calls to compute avg latency per model
        allowed_fails: 1
        num_retries: 3
        retry_after: 5 # seconds to wait before retrying a failed request
        cooldown_time: 30 # seconds to cooldown a deployment after failure
        """
        assert settings["routing_strategy"] == "latency-based-routing"
        assert settings["routing_strategy_args"]["ttl"] == 10
        assert settings["allowed_fails"] == 1
        assert settings["num_retries"] == 3
        assert settings["retry_after"] == 5
        assert settings["cooldown_time"] == 30

    except Exception:
        print(traceback.format_exc())
        pytest.fail("An error occurred - " + traceback.format_exc())


from litellm.types.utils import CallTypes


def test_update_kwargs_before_fallbacks_unit_test():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-nano",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ],
    )

    kwargs = {"messages": [{"role": "user", "content": "write 1 sentence poem"}]}

    router._update_kwargs_before_fallbacks(
        model="gpt-3.5-turbo",
        kwargs=kwargs,
    )

    assert kwargs["litellm_trace_id"] is not None


@pytest.mark.parametrize(
    "call_type",
    [
        CallTypes.acompletion,
        CallTypes.atext_completion,
        CallTypes.aembedding,
        CallTypes.arerank,
        CallTypes.atranscription,
    ],
)
@pytest.mark.asyncio
async def test_update_kwargs_before_fallbacks(call_type):

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-nano",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ],
    )

    if call_type.value.startswith("a"):
        with patch.object(router, "async_function_with_fallbacks") as mock_client:
            if call_type.value == "acompletion":
                input_kwarg = {
                    "messages": [{"role": "user", "content": "Hello, how are you?"}],
                }
            elif (
                call_type.value == "atext_completion"
                or call_type.value == "aimage_generation"
            ):
                input_kwarg = {
                    "prompt": "Hello, how are you?",
                }
            elif call_type.value == "aembedding" or call_type.value == "arerank":
                input_kwarg = {
                    "input": "Hello, how are you?",
                }
            elif call_type.value == "atranscription":
                input_kwarg = {
                    "file": "path/to/file",
                }
            else:
                input_kwarg = {}

            await getattr(router, call_type.value)(
                model="gpt-3.5-turbo",
                **input_kwarg,
            )

            mock_client.assert_called_once()

            print(mock_client.call_args.kwargs)
            assert mock_client.call_args.kwargs["litellm_trace_id"] is not None


def test_router_get_model_info_wildcard_routes():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1},
            },
        ]
    )
    model_info = router.get_router_model_info(
        deployment=None, received_model_name="gemini/gemini-1.5-flash", id="1"
    )
    print(model_info)
    assert model_info is not None
    assert model_info["tpm"] is not None
    assert model_info["rpm"] is not None


@pytest.mark.asyncio
async def test_router_get_model_group_usage_wildcard_routes():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1},
            },
        ]
    )

    resp = await router.acompletion(
        model="gemini/gemini-1.5-flash",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="Hello, I'm good.",
    )
    print(resp)

    await asyncio.sleep(1)

    tpm, rpm = await router.get_model_group_usage(model_group="gemini/gemini-1.5-flash")

    assert tpm is not None, "tpm is None"
    assert rpm is not None, "rpm is None"


@pytest.mark.asyncio
async def test_call_router_callbacks_on_success():
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1},
            },
        ]
    )

    with patch.object(
        router.cache, "async_increment_cache_pipeline", new=AsyncMock()
    ) as mock_callback:
        await router.acompletion(
            model="gemini/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            mock_response="Hello, I'm good.",
        )
        await asyncio.sleep(1)
        assert mock_callback.call_count == 1

        increment_list = mock_callback.call_args_list[0].kwargs["increment_list"]
        assert len(increment_list) == 2

        for increment in increment_list:
            if "tpm" in increment["key"]:
                assert increment["key"].startswith(
                    "global_router:1:gemini/gemini-1.5-flash:tpm"
                )
                assert increment["increment_value"] == 30
            elif "rpm" in increment["key"]:
                assert increment["key"].startswith(
                    "global_router:1:gemini/gemini-1.5-flash:rpm"
                )
                assert increment["increment_value"] == 1


@pytest.mark.asyncio
async def test_call_router_callbacks_on_failure():
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1},
            },
        ]
    )

    with patch.object(
        router.cache, "async_increment_cache", new=AsyncMock()
    ) as mock_callback:
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(
                model="gemini/gemini-1.5-flash",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                mock_response="litellm.RateLimitError",
                num_retries=0,
            )
        await asyncio.sleep(1)
        print(mock_callback.call_args_list)
        assert mock_callback.call_count == 1

        assert (
            mock_callback.call_args_list[0]
            .kwargs["key"]
            .startswith("global_router:1:gemini/gemini-1.5-flash:rpm")
        )


@pytest.mark.asyncio
async def test_router_model_group_headers():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    from litellm.types.utils import OPENAI_RESPONSE_HEADERS

    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1},
            }
        ]
    )

    for _ in range(2):
        resp = await router.acompletion(
            model="gemini/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            mock_response="Hello, I'm good.",
        )
        await asyncio.sleep(1)

    assert (
        resp._hidden_params["additional_headers"]["x-litellm-model-group"]
        == "gemini/gemini-1.5-flash"
    )

    assert "x-ratelimit-remaining-requests" in resp._hidden_params["additional_headers"]
    assert "x-ratelimit-remaining-tokens" in resp._hidden_params["additional_headers"]


@pytest.mark.asyncio
async def test_get_remaining_model_group_usage():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    from litellm.types.utils import OPENAI_RESPONSE_HEADERS

    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1},
            }
        ]
    )
    for _ in range(2):
        resp = await router.acompletion(
            model="gemini/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            mock_response="Hello, I'm good.",
        )
        assert (
            "x-ratelimit-remaining-tokens" in resp._hidden_params["additional_headers"]
        )
        assert (
            "x-ratelimit-remaining-requests"
            in resp._hidden_params["additional_headers"]
        )
        await asyncio.sleep(1)

    remaining_usage = await router.get_remaining_model_group_usage(
        model_group="gemini/gemini-1.5-flash"
    )
    assert remaining_usage is not None
    assert "x-ratelimit-remaining-requests" in remaining_usage
    assert "x-ratelimit-remaining-tokens" in remaining_usage


@pytest.mark.parametrize(
    "potential_access_group, expected_result",
    [("gemini-models", True), ("gemini-models-2", False), ("gemini/*", False)],
)
def test_router_get_model_access_groups(potential_access_group, expected_result):
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1, "access_groups": ["gemini-models"]},
            },
        ]
    )
    access_groups = router._is_model_access_group_for_wildcard_route(
        model_access_group=potential_access_group
    )
    assert access_groups == expected_result


def test_router_redis_cache():
    router = Router(
        model_list=[{"model_name": "gemini/*", "litellm_params": {"model": "gemini/*"}}]
    )

    redis_cache = MagicMock()

    router._update_redis_cache(cache=redis_cache)

    assert router.cache.redis_cache == redis_cache


def test_router_handle_clientside_credential():
    deployment = {
        "model_name": "gemini/*",
        "litellm_params": {"model": "gemini/*"},
        "model_info": {
            "id": "1",
        },
    }
    router = Router(model_list=[deployment])

    new_deployment = router._handle_clientside_credential(
        deployment=deployment,
        kwargs={
            "api_key": "123",
            "metadata": {"model_group": "gemini/gemini-1.5-flash"},
        },
        function_name="acompletion",
    )

    assert new_deployment.litellm_params.api_key == "123"
    assert len(router.get_model_list()) == 2


def test_router_get_async_openai_model_client():
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {
                    "model": "gemini/*",
                    "api_base": "https://api.gemini.com",
                },
            }
        ]
    )
    model_client = router._get_async_openai_model_client(
        deployment=MagicMock(), kwargs={}
    )
    assert model_client is None


def test_router_get_deployment_credentials():
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*", "api_key": "123"},
                "model_info": {"id": "1"},
            }
        ]
    )
    credentials = router.get_deployment_credentials(model_id="1")
    assert credentials is not None
    assert credentials["api_key"] == "123"


def test_router_get_deployment_model_info():
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": "1"},
            }
        ]
    )
    model_info = router.get_deployment_model_info(
        model_id="1", model_name="gemini/gemini-1.5-flash"
    )
    assert model_info is not None
