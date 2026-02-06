# Tests for router.get_available_deployment
# specifically test if it can pick the correct LLM when rpm/tpm set
# These are fast Tests, and make no API calls
import asyncio
import os
import sys
import time
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

import litellm
from litellm import Router

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
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "rpm": 6,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
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

        # call get_available_deployment 1k times, it should pick azure/gpt-4.1-mini about 90% of the time
        for _ in range(1000):
            selected_model = router.get_available_deployment("gpt-3.5-turbo")
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'azure/gpt-4.1-mini' has about 90% of the total requests
        assert (
            selection_counts["azure/gpt-4.1-mini"] / total_requests > 0.89
        ), f"Assertion failed: 'azure/gpt-4.1-mini' does not have about 90% of the total requests in the weighted load balancer. Selection counts {selection_counts}"

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
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "tpm": 5,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
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

        # call get_available_deployment 1k times, it should pick azure/gpt-4.1-mini about 90% of the time
        for _ in range(1000):
            selected_model = router.get_available_deployment("gpt-3.5-turbo")
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'azure/gpt-4.1-mini' has about 90% of the total requests
        assert (
            selection_counts["azure/gpt-4.1-mini"] / total_requests > 0.89
        ), f"Assertion failed: 'azure/gpt-4.1-mini' does not have about 90% of the total requests in the weighted load balancer. Selection counts {selection_counts}"

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
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 5,
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
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

        # call get_available_deployment 1k times, it should pick azure/gpt-4.1-mini about 90% of the time
        for _ in range(1000):
            selected_model = router.get_available_deployment("gpt-3.5-turbo")
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'azure/gpt-4.1-mini' has about 90% of the total requests
        assert (
            selection_counts["azure/gpt-4.1-mini"] / total_requests > 0.89
        ), f"Assertion failed: 'azure/gpt-4.1-mini' does not have about 90% of the total requests in the weighted load balancer. Selection counts {selection_counts}"

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_weighted_selection_router_tpm_as_router_param()


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
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "rpm": 5,
                "tpm": 5,
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
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

        # call get_available_deployment 1k times, it should pick azure/gpt-4.1-mini about 90% of the time
        for _ in range(1000):
            selected_model = router.get_available_deployment("gpt-3.5-turbo")
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'azure/gpt-4.1-mini' has about 90% of the total requests
        assert (
            selection_counts["azure/gpt-4.1-mini"] / total_requests > 0.89
        ), f"Assertion failed: 'azure/gpt-4.1-mini' does not have about 90% of the total requests in the weighted load balancer. Selection counts {selection_counts}"

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
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "rpm": 6,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
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

        # call get_available_deployment 1k times, it should pick azure/gpt-4.1-mini about 90% of the time
        for _ in range(1000):
            selected_model = router.get_available_deployment("claude-1")
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'azure/gpt-4.1-mini' has about 90% of the total requests
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
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "tpm": 1,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
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
        # call get_available_deployment 1k times, it should pick azure/gpt-4.1-mini about 90% of the time
        selection_counts = defaultdict(int)
        for _ in range(1000):
            selected_model = router.get_available_deployment("gpt-3.5-turbo")
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'azure/gpt-4.1-mini' has about 90% of the total requests
        assert (
            selection_counts["azure/gpt-4.1-mini"] / total_requests > 0.89
        ), f"Assertion failed: 'azure/gpt-4.1-mini' does not have about 90% of the total requests in the weighted load balancer. Selection counts {selection_counts}"

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


# test_model_group_aliases()


@pytest.mark.flaky(retries=3, delay=2)
def test_usage_based_routing():
    """
    in this test we, have a model group with two models in it, model-a and model-b.
    Then at some point, we exceed the TPM limit (set in the litellm_params)
    for model-a only; but for model-b we are still under the limit
    """
    try:

        def get_azure_params(deployment_name: str):
            params = {
                "model": f"azure/{deployment_name}",
                "api_key": os.environ["AZURE_API_KEY"],
                "api_version": os.environ["AZURE_API_VERSION"],
                "api_base": "https://fake-api.openai.com/v1",
            }
            return params

        model_list = [
            {
                "model_name": "azure/gpt-4",
                "litellm_params": get_azure_params("chatgpt-low-tpm"),
                "tpm": 100,
            },
            {
                "model_name": "azure/gpt-4",
                "litellm_params": get_azure_params("chatgpt-high-tpm"),
                "tpm": 1000,
            },
        ]

        router = Router(
            model_list=model_list,
            set_verbose=True,
            debug_level="DEBUG",
            routing_strategy="usage-based-routing",
            redis_host=os.environ["REDIS_HOST"],
            redis_port=os.environ["REDIS_PORT"],
        )

        messages = [
            {"content": "Tell me a joke.", "role": "user"},
        ]

        selection_counts = defaultdict(int)
        for _ in range(25):
            response = router.completion(
                model="azure/gpt-4",
                messages=messages,
                timeout=5,
                mock_response="good morning",
            )

            # print("response", response)

            selection_counts[response["model"]] += 1

        print("selection counts", selection_counts)

        total_requests = sum(selection_counts.values())

        # Assert that 'chatgpt-low-tpm' has more than 2 requests
        assert (
            selection_counts["chatgpt-low-tpm"] > 2
        ), f"Assertion failed: 'chatgpt-low-tpm' does not have more than 2 request in the weighted load balancer. Selection counts {selection_counts}"

        # Assert that 'chatgpt-high-tpm' has about 70% of the total requests [DO NOT MAKE THIS LOWER THAN 70%]
        assert (
            selection_counts["chatgpt-high-tpm"] / total_requests > 0.70
        ), f"Assertion failed: 'chatgpt-high-tpm' does not have about 80% of the total requests in the weighted load balancer. Selection counts {selection_counts}"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_wildcard_openai_routing():
    """
    Initialize router with *, all models go through * and use OPENAI_API_KEY
    """
    try:
        model_list = [
            {
                "model_name": "*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
                "tpm": 100,
            },
        ]

        router = Router(
            model_list=model_list,
        )

        messages = [
            {"content": "Tell me a joke.", "role": "user"},
        ]

        selection_counts = defaultdict(int)
        for _ in range(25):
            response = await router.acompletion(
                model="gpt-4",
                messages=messages,
                mock_response="good morning",
            )
            # print("response1", response)

            selection_counts[response["model"]] += 1

            response = await router.acompletion(
                model="gpt-3.5-turbo",
                messages=messages,
                mock_response="good morning",
            )
            # print("response2", response)

            selection_counts[response["model"]] += 1

            response = await router.acompletion(
                model="gpt-4-turbo-preview",
                messages=messages,
                mock_response="good morning",
            )
            # print("response3", response)

            # print("response", response)

            selection_counts[response["model"]] += 1

        assert selection_counts["gpt-4"] == 25
        assert selection_counts["gpt-3.5-turbo"] == 25
        assert selection_counts["gpt-4-turbo-preview"] == 25

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


"""
Test async router get deployment (Simpl-shuffle)
"""

rpm_list = [[None, None], [6, 1440]]
tpm_list = [[None, None], [6, 1440]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rpm_list, tpm_list",
    [(rpm, tpm) for rpm in rpm_list for tpm in tpm_list],
)
async def test_weighted_selection_router_async(rpm_list, tpm_list):
    # this tests if load balancing works based on the provided rpms in the router
    # it's a fast test, only tests get_available_deployment
    # users can pass rpms as a litellm_param
    try:
        litellm.set_verbose = False
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "rpm": rpm_list[0],
                    "tpm": tpm_list[0],
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "rpm": rpm_list[1],
                    "tpm": tpm_list[1],
                },
            },
        ]
        router = Router(
            model_list=model_list,
        )
        selection_counts = defaultdict(int)

        # call get_available_deployment 1k times, it should pick azure/gpt-4.1-mini about 90% of the time
        for _ in range(1000):
            selected_model = await router.async_get_available_deployment(
                "gpt-3.5-turbo", request_kwargs={}
            )
            selected_model_id = selected_model["litellm_params"]["model"]
            selected_model_name = selected_model_id
            selection_counts[selected_model_name] += 1
        print(selection_counts)

        total_requests = sum(selection_counts.values())

        if rpm_list[0] is not None or tpm_list[0] is not None:
            # Assert that 'azure/gpt-4.1-mini' has about 90% of the total requests
            assert (
                selection_counts["azure/gpt-4.1-mini"] / total_requests > 0.89
            ), f"Assertion failed: 'azure/gpt-4.1-mini' does not have about 90% of the total requests in the weighted load balancer. Selection counts {selection_counts}"
        else:
            # Assert both are used
            assert selection_counts["azure/gpt-4.1-mini"] > 0
            assert selection_counts["gpt-3.5-turbo"] > 0
        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


def test_get_available_deployment_for_pass_through():
    """
    Test get_available_deployment_for_pass_through function
    - Tests that only deployments with use_in_pass_through=True are returned
    - Tests that BadRequestError is raised when no pass-through deployments exist
    """
    try:
        litellm.set_verbose = False
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "use_in_pass_through": True,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "use_in_pass_through": False,
                },
            },
        ]
        router = Router(
            model_list=model_list,
        )

        # Test that only pass-through deployment is returned
        selected_model = router.get_available_deployment_for_pass_through(
            "gpt-3.5-turbo"
        )
        assert selected_model["litellm_params"]["model"] == "gpt-3.5-turbo"
        assert selected_model["litellm_params"]["use_in_pass_through"] is True

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


def test_get_available_deployment_for_pass_through_no_deployments():
    """
    Test get_available_deployment_for_pass_through raises BadRequestError
    when no deployments have use_in_pass_through=True
    """
    try:
        litellm.set_verbose = False
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "use_in_pass_through": False,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "use_in_pass_through": False,
                },
            },
        ]
        router = Router(
            model_list=model_list,
        )

        # Test that BadRequestError is raised when no pass-through deployments exist
        try:
            router.get_available_deployment_for_pass_through("gpt-3.5-turbo")
            pytest.fail(
                "Expected BadRequestError when no pass-through deployments exist"
            )
        except litellm.BadRequestError as e:
            assert "use_in_pass_through=True" in str(e)

        router.reset()
    except Exception as e:
        if isinstance(e, litellm.BadRequestError):
            pass  # Expected error
        else:
            traceback.print_exc()
            pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_async_get_available_deployment_for_pass_through():
    """
    Test async_get_available_deployment_for_pass_through function
    - Tests that only deployments with use_in_pass_through=True are returned
    - Tests async version works correctly
    """
    try:
        litellm.set_verbose = False
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "use_in_pass_through": True,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "use_in_pass_through": False,
                },
            },
        ]
        router = Router(
            model_list=model_list,
        )

        # Test that only pass-through deployment is returned
        selected_model = await router.async_get_available_deployment_for_pass_through(
            model="gpt-3.5-turbo", request_kwargs={}
        )
        assert selected_model["litellm_params"]["model"] == "gpt-3.5-turbo"
        assert selected_model["litellm_params"]["use_in_pass_through"] is True

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")


def test_filter_pass_through_deployments():
    """
    Test _filter_pass_through_deployments function
    - Tests that it correctly filters deployments with use_in_pass_through=True
    """
    try:
        litellm.set_verbose = False
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "use_in_pass_through": True,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "use_in_pass_through": False,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-35-turbo",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "use_in_pass_through": True,
                },
            },
        ]
        router = Router(
            model_list=model_list,
        )

        # Get all healthy deployments
        healthy_deployments = router.get_model_list()

        # Filter pass-through deployments
        pass_through_deployments = router._filter_pass_through_deployments(
            healthy_deployments
        )

        # Should only have 2 deployments with use_in_pass_through=True
        assert len(pass_through_deployments) == 2

        # Verify all returned deployments have use_in_pass_through=True
        for deployment in pass_through_deployments:
            assert deployment["litellm_params"]["use_in_pass_through"] is True

        router.reset()
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")
