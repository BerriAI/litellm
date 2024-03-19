#### What this tests ####
#    This tests caching on the router
import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router

## Scenarios
## 1. 2 models - openai + azure - 1 model group "gpt-3.5-turbo",
## 2. 2 models - openai, azure - 2 diff model groups, 1 caching group


@pytest.mark.asyncio
async def test_acompletion_caching_on_router():
    # tests acompletion + caching on router
    try:
        litellm.set_verbose = True
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-0613",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 100000,
                "rpm": 10000,
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                },
                "tpm": 100000,
                "rpm": 10000,
            },
        ]

        messages = [
            {"role": "user", "content": f"write a one sentence poem {time.time()}?"}
        ]
        start_time = time.time()
        router = Router(
            model_list=model_list,
            redis_host=os.environ["REDIS_HOST"],
            redis_password=os.environ["REDIS_PASSWORD"],
            redis_port=os.environ["REDIS_PORT"],
            cache_responses=True,
            timeout=30,
            routing_strategy="simple-shuffle",
        )
        response1 = await router.acompletion(
            model="gpt-3.5-turbo", messages=messages, temperature=1
        )
        print(f"response1: {response1}")
        await asyncio.sleep(1)  # add cache is async, async sleep for cache to get set
        response2 = await router.acompletion(
            model="gpt-3.5-turbo", messages=messages, temperature=1
        )
        print(f"response2: {response2}")
        assert response1.id == response2.id
        assert len(response1.choices[0].message.content) > 0
        assert (
            response1.choices[0].message.content == response2.choices[0].message.content
        )
        router.reset()
    except litellm.Timeout as e:
        end_time = time.time()
        print(f"timeout error occurred: {end_time - start_time}")
        pass
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_acompletion_caching_with_ttl_on_router():
    # tests acompletion + caching on router
    try:
        litellm.set_verbose = True
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-0613",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 100000,
                "rpm": 10000,
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                },
                "tpm": 100000,
                "rpm": 10000,
            },
        ]

        messages = [
            {"role": "user", "content": f"write a one sentence poem {time.time()}?"}
        ]
        start_time = time.time()
        router = Router(
            model_list=model_list,
            redis_host=os.environ["REDIS_HOST"],
            redis_password=os.environ["REDIS_PASSWORD"],
            redis_port=os.environ["REDIS_PORT"],
            cache_responses=True,
            timeout=30,
            routing_strategy="simple-shuffle",
        )
        response1 = await router.acompletion(
            model="gpt-3.5-turbo", messages=messages, temperature=1, ttl=0
        )
        print(f"response1: {response1}")
        await asyncio.sleep(1)  # add cache is async, async sleep for cache to get set
        response2 = await router.acompletion(
            model="gpt-3.5-turbo", messages=messages, temperature=1, ttl=0
        )
        print(f"response2: {response2}")
        assert response1.id != response2.id
        assert len(response1.choices[0].message.content) > 0
        assert (
            response1.choices[0].message.content != response2.choices[0].message.content
        )
        router.reset()
    except litellm.Timeout as e:
        end_time = time.time()
        print(f"timeout error occurred: {end_time - start_time}")
        pass
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_acompletion_caching_on_router_caching_groups():
    # tests acompletion + caching on router
    try:
        litellm.set_verbose = True
        model_list = [
            {
                "model_name": "openai-gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-0613",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 100000,
                "rpm": 10000,
            },
            {
                "model_name": "azure-gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                },
                "tpm": 100000,
                "rpm": 10000,
            },
        ]

        messages = [
            {"role": "user", "content": f"write a one sentence poem {time.time()}?"}
        ]
        start_time = time.time()
        router = Router(
            model_list=model_list,
            redis_host=os.environ["REDIS_HOST"],
            redis_password=os.environ["REDIS_PASSWORD"],
            redis_port=os.environ["REDIS_PORT"],
            cache_responses=True,
            timeout=30,
            routing_strategy="simple-shuffle",
            caching_groups=[("openai-gpt-3.5-turbo", "azure-gpt-3.5-turbo")],
        )
        response1 = await router.acompletion(
            model="openai-gpt-3.5-turbo", messages=messages, temperature=1
        )
        print(f"response1: {response1}")
        await asyncio.sleep(1)  # add cache is async, async sleep for cache to get set
        response2 = await router.acompletion(
            model="azure-gpt-3.5-turbo", messages=messages, temperature=1
        )
        print(f"response2: {response2}")
        assert response1.id == response2.id
        assert len(response1.choices[0].message.content) > 0
        assert (
            response1.choices[0].message.content == response2.choices[0].message.content
        )
        router.reset()
    except litellm.Timeout as e:
        end_time = time.time()
        print(f"timeout error occurred: {end_time - start_time}")
        pass
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")
