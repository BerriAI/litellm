from __future__ import annotations

from typing import Final

import pytest

import litellm
from litellm import Router
import os
import traceback
from collections import defaultdict


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
        with pytest.raises(litellm.BadRequestError) as exc_info:
            router.get_available_deployment_for_pass_through("gpt-3.5-turbo")
        e = exc_info.value
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
