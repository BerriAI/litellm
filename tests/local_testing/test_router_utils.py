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
from litellm.router import Deployment, LiteLLM_Params, ModelInfo
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
                    "model": "azure/chatgpt-v-2",
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
                    "model": "azure/chatgpt-v-2",
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
                    "model": "azure/chatgpt-v-2",
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
