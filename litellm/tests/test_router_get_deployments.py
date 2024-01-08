# Tests for router.get_available_deployment
# specifically test if it can pick the correct LLM when rpm/tpm set
# These are fast Tests, and make no API calls
import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


def test_weighted_selection_router():
    # this tests if load balancing works based on the provided rpms in the router
    # it's a fast test, only tests get_available_deployment
    # users can pass rpms as a litellm_param
    try:
        litellm.set_verbose = False
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-0613",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "rpm": 6,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "rpm": 1440,
                },
            },
        ]
        router = Router(
            model_list=model_list,
        )
        selection_counts = defaultdict(int)

        # call get_available_deployment 1k times, it should pick azure/chatgpt-v-2 about 90% of the time
        for _ in range(1000):
            selected_model = router.get_available_deployment("gpt-3.5-turbo")
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'azure/chatgpt-v-2' has about 90% of the total requests
        assert (
            selection_counts["azure/chatgpt-v-2"] / total_requests > 0.89
        ), f"Assertion failed: 'azure/chatgpt-v-2' does not have about 90% of the total requests in the weighted load balancer. Selection counts {selection_counts}"

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_weighted_selection_router()


def test_weighted_selection_router_tpm():
    # this tests if load balancing works based on the provided tpms in the router
    # it's a fast test, only tests get_available_deployment
    # users can pass rpms as a litellm_param
    try:
        print("\ntest weighted selection based on TPM\n")
        litellm.set_verbose = False
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-0613",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "tpm": 5,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "tpm": 90,
                },
            },
        ]
        router = Router(
            model_list=model_list,
        )
        selection_counts = defaultdict(int)

        # call get_available_deployment 1k times, it should pick azure/chatgpt-v-2 about 90% of the time
        for _ in range(1000):
            selected_model = router.get_available_deployment("gpt-3.5-turbo")
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'azure/chatgpt-v-2' has about 90% of the total requests
        assert (
            selection_counts["azure/chatgpt-v-2"] / total_requests > 0.89
        ), f"Assertion failed: 'azure/chatgpt-v-2' does not have about 90% of the total requests in the weighted load balancer. Selection counts {selection_counts}"

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_weighted_selection_router_tpm()


def test_weighted_selection_router_tpm_as_router_param():
    # this tests if load balancing works based on the provided tpms in the router
    # it's a fast test, only tests get_available_deployment
    # users can pass rpms as a litellm_param
    try:
        print("\ntest weighted selection based on TPM\n")
        litellm.set_verbose = False
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-0613",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 5,
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                },
                "tpm": 90,
            },
        ]
        router = Router(
            model_list=model_list,
        )
        selection_counts = defaultdict(int)

        # call get_available_deployment 1k times, it should pick azure/chatgpt-v-2 about 90% of the time
        for _ in range(1000):
            selected_model = router.get_available_deployment("gpt-3.5-turbo")
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'azure/chatgpt-v-2' has about 90% of the total requests
        assert (
            selection_counts["azure/chatgpt-v-2"] / total_requests > 0.89
        ), f"Assertion failed: 'azure/chatgpt-v-2' does not have about 90% of the total requests in the weighted load balancer. Selection counts {selection_counts}"

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


test_weighted_selection_router_tpm_as_router_param()


def test_weighted_selection_router_rpm_as_router_param():
    # this tests if load balancing works based on the provided tpms in the router
    # it's a fast test, only tests get_available_deployment
    # users can pass rpms as a litellm_param
    try:
        print("\ntest weighted selection based on RPM\n")
        litellm.set_verbose = False
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-0613",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "rpm": 5,
                "tpm": 5,
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                },
                "rpm": 90,
                "tpm": 90,
            },
        ]
        router = Router(
            model_list=model_list,
        )
        selection_counts = defaultdict(int)

        # call get_available_deployment 1k times, it should pick azure/chatgpt-v-2 about 90% of the time
        for _ in range(1000):
            selected_model = router.get_available_deployment("gpt-3.5-turbo")
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'azure/chatgpt-v-2' has about 90% of the total requests
        assert (
            selection_counts["azure/chatgpt-v-2"] / total_requests > 0.89
        ), f"Assertion failed: 'azure/chatgpt-v-2' does not have about 90% of the total requests in the weighted load balancer. Selection counts {selection_counts}"

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_weighted_selection_router_tpm_as_router_param()


def test_weighted_selection_router_no_rpm_set():
    # this tests if we can do selection when no rpm is provided too
    # it's a fast test, only tests get_available_deployment
    # users can pass rpms as a litellm_param
    try:
        litellm.set_verbose = False
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-0613",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "rpm": 6,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "rpm": 1440,
                },
            },
            {
                "model_name": "claude-1",
                "litellm_params": {
                    "model": "bedrock/claude1.2",
                    "rpm": 1440,
                },
            },
        ]
        router = Router(
            model_list=model_list,
        )
        selection_counts = defaultdict(int)

        # call get_available_deployment 1k times, it should pick azure/chatgpt-v-2 about 90% of the time
        for _ in range(1000):
            selected_model = router.get_available_deployment("claude-1")
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'azure/chatgpt-v-2' has about 90% of the total requests
        assert (
            selection_counts["bedrock/claude1.2"] / total_requests == 1
        ), f"Assertion failed: Selection counts {selection_counts}"

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_weighted_selection_router_no_rpm_set()


def test_model_group_aliases():
    try:
        litellm.set_verbose = False
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo-0613",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "tpm": 1,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "tpm": 99,
                },
            },
            {
                "model_name": "claude-1",
                "litellm_params": {
                    "model": "bedrock/claude1.2",
                    "tpm": 1,
                },
            },
        ]
        router = Router(
            model_list=model_list,
            model_group_alias={
                "gpt-4": "gpt-3.5-turbo"
            },  # gpt-4 requests sent to gpt-3.5-turbo
        )

        # test that gpt-4 requests are sent to gpt-3.5-turbo
        for _ in range(20):
            selected_model = router.get_available_deployment("gpt-4")
            print("\n selected model", selected_model)
            selected_model_name = selected_model.get("model_name")
            if selected_model_name != "gpt-3.5-turbo":
                pytest.fail(
                    f"Selected model {selected_model_name} is not gpt-3.5-turbo"
                )

        # test that
        # call get_available_deployment 1k times, it should pick azure/chatgpt-v-2 about 90% of the time
        selection_counts = defaultdict(int)
        for _ in range(1000):
            selected_model = router.get_available_deployment("gpt-3.5-turbo")
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'azure/chatgpt-v-2' has about 90% of the total requests
        assert (
            selection_counts["azure/chatgpt-v-2"] / total_requests > 0.89
        ), f"Assertion failed: 'azure/chatgpt-v-2' does not have about 90% of the total requests in the weighted load balancer. Selection counts {selection_counts}"

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_model_group_aliases()
