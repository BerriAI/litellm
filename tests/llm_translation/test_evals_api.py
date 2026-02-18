"""
Tests for Evals API operations across providers
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.types.llms.openai_evals import (
    CancelEvalResponse,
    DeleteEvalResponse,
    Eval,
    ListEvalsResponse,
)


class BaseEvalsAPITest(ABC):
    """
    Base test class for Evals API operations.
    Tests create, list, get, update, delete, and cancel operations.
    """

    @abstractmethod
    def get_custom_llm_provider(self) -> str:
        """Return the provider name (e.g., 'openai')"""
        pass

    @abstractmethod
    def get_api_key(self) -> Optional[str]:
        """Return the API key for the provider"""
        pass

    @abstractmethod
    def get_api_base(self) -> Optional[str]:
        """Return the API base URL for the provider"""
        pass

    def test_create_eval(self):
        """
        Test creating an evaluation.
        """
        import time

        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        litellm.set_verbose = True

        # Create eval with stored_completions data source
        unique_name = f"Test Eval {int(time.time())}"

        response = litellm.create_eval(
            name=unique_name,
            data_source_config={
                "type": "stored_completions",
                "metadata": {"usecase": "chatbot"},
            },
            testing_criteria=[
                {
                    "type": "label_model",
                    "model": "gpt-4o",
                    "input": [
                        {
                            "role": "developer",
                            "content": "Classify the sentiment as 'positive' or 'negative'",
                        },
                        {"role": "user", "content": "Statement: {{item.input}}"},
                    ],
                    "passing_labels": ["positive"],
                    "labels": ["positive", "negative"],
                    "name": "Sentiment grader",
                }
            ],
            custom_llm_provider=custom_llm_provider,
            api_key=api_key,
            api_base=api_base,
        )

        assert response is not None
        assert isinstance(response, Eval)
        assert response.id is not None
        assert response.name == unique_name
        print(f"Created eval: {response}")
        print(f"Eval ID: {response.id}")

    def test_list_evals(self):
        """
        Test listing evaluations.
        """
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        litellm.set_verbose = True

        response = litellm.list_evals(
            limit=10,
            custom_llm_provider=custom_llm_provider,
            api_key=api_key,
            api_base=api_base,
        )

        assert response is not None
        assert isinstance(response, ListEvalsResponse)
        assert hasattr(response, "data")
        assert hasattr(response, "has_more")
        print(f"Listed evals: {len(response.data)} evaluations")

    def test_get_eval(self):
        """
        Test getting a specific evaluation by ID.
        """
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        litellm.set_verbose = True

        # First list existing evals to get an ID
        list_response = litellm.list_evals(
            limit=1,
            custom_llm_provider=custom_llm_provider,
            api_key=api_key,
            api_base=api_base,
        )

        assert isinstance(list_response, ListEvalsResponse)

        if list_response.data and len(list_response.data) > 0:
            eval_id = list_response.data[0].id
            print(f"Testing with eval ID: {eval_id}")

            # Get the eval
            response = litellm.get_eval(
                eval_id=eval_id,
                custom_llm_provider=custom_llm_provider,
                api_key=api_key,
                api_base=api_base,
            )

            assert response is not None
            assert isinstance(response, Eval)
            assert response.id == eval_id
            print(f"Retrieved eval: {response}")
        else:
            pytest.skip("No existing evals to test with")

    def test_update_eval(self):
        """
        Test updating an evaluation.
        """
        import time

        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        litellm.set_verbose = True

        # First list existing evals
        list_response = litellm.list_evals(
            limit=1,
            custom_llm_provider=custom_llm_provider,
            api_key=api_key,
            api_base=api_base,
        )

        assert isinstance(list_response, ListEvalsResponse)

        if list_response.data and len(list_response.data) > 0:
            eval_id = list_response.data[0].id
            updated_name = f"Updated Eval {int(time.time())}"

            # Update the eval
            response = litellm.update_eval(
                eval_id=eval_id,
                name=updated_name,
                custom_llm_provider=custom_llm_provider,
                api_key=api_key,
                api_base=api_base,
            )

            assert response is not None
            assert isinstance(response, Eval)
            assert response.id == eval_id
            assert response.name == updated_name
            print(f"Updated eval: {response}")
        else:
            pytest.skip("No existing evals to test with")

    def test_delete_eval(self):
        """
        Test deleting an evaluation.
        """
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        # Skip this test to avoid deleting production evals
        pytest.skip("Skipping delete test to preserve existing evals")


class TestOpenAIEvalsAPI(BaseEvalsAPITest):
    """
    Test OpenAI Evals API implementation.
    """

    def get_custom_llm_provider(self) -> str:
        return "openai"

    def get_api_key(self) -> Optional[str]:
        return os.environ.get("OPENAI_API_KEY")

    def get_api_base(self) -> Optional[str]:
        return os.environ.get("OPENAI_API_BASE")
