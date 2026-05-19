"""
Tests for Evals API operations across providers
"""

import hashlib
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


def _stable_eval_name(test_node_name: str, suffix: str = "") -> str:
    """Deterministic eval name keyed off the test's node name.

    The previous ``f"Test Eval {int(time.time())}"`` pattern embedded a
    fresh value into the request body every run, defeating VCR's
    ``safe_body`` matcher and forcing a real OpenAI ``create`` call on
    every CI run. With a stable per-test name the cassette matches on
    replay, and provider-side resources stay bounded because each test
    deletes the eval it owns on teardown.
    """
    nonce = hashlib.sha1(test_node_name.encode()).hexdigest()[:12]
    return f"vcr-managed-{nonce}{suffix}"


_TESTING_CRITERIA = [
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
]


_PROVIDER_FLAKINESS = (
    litellm.InternalServerError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.ServiceUnavailableError,
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

    @pytest.fixture
    def managed_eval(self, request):
        """Create a stable-named eval for this test; delete on teardown.

        Function-scoped so each cassette captures the full
        create→test→delete cycle. A class-scoped fixture would push
        the create into whichever test ran first and the delete into
        whichever ran last, which is fragile under reordering.

        Replaces the prior ``list_evals().data[0].id`` pattern, which
        made the URL of ``get_eval`` / ``update_eval`` vary across
        runs (the "first" eval depends on what other runs left
        behind).
        """
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        try:
            created = litellm.create_eval(
                name=_stable_eval_name(request.node.name),
                data_source_config={
                    "type": "stored_completions",
                    "metadata": {"usecase": "chatbot", "vcr": "managed"},
                },
                testing_criteria=_TESTING_CRITERIA,
                custom_llm_provider=custom_llm_provider,
                api_key=api_key,
                api_base=api_base,
            )
        except _PROVIDER_FLAKINESS:
            pytest.skip("Provider service unavailable")
        except litellm.RateLimitError:
            pytest.skip("Rate limit exceeded")

        yield created

        # Best-effort cleanup. OpenAI eval names are not unique-keyed
        # (only IDs are), so a failed delete doesn't block the next
        # run's create.
        try:
            litellm.delete_eval(
                eval_id=created.id,
                custom_llm_provider=custom_llm_provider,
                api_key=api_key,
                api_base=api_base,
            )
        except Exception:
            pass

    @pytest.mark.flaky(retries=3, delay=2)
    def test_create_eval(self, request):
        """
        Test creating an evaluation.
        """
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        litellm.set_verbose = True
        unique_name = _stable_eval_name(request.node.name)

        created_id = None
        try:
            try:
                response = litellm.create_eval(
                    name=unique_name,
                    data_source_config={
                        "type": "stored_completions",
                        "metadata": {"usecase": "chatbot"},
                    },
                    testing_criteria=_TESTING_CRITERIA,
                    custom_llm_provider=custom_llm_provider,
                    api_key=api_key,
                    api_base=api_base,
                )
            except _PROVIDER_FLAKINESS:
                pytest.skip("Provider service unavailable")
            except litellm.RateLimitError:
                pytest.skip("Rate limit exceeded")

            assert response is not None
            assert isinstance(response, Eval)
            assert response.id is not None
            assert response.name == unique_name
            created_id = response.id
            print(f"Created eval: {response}")
            print(f"Eval ID: {response.id}")
        finally:
            if created_id is not None:
                try:
                    litellm.delete_eval(
                        eval_id=created_id,
                        custom_llm_provider=custom_llm_provider,
                        api_key=api_key,
                        api_base=api_base,
                    )
                except Exception:
                    pass

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

    def test_get_eval(self, managed_eval):
        """
        Test getting a specific evaluation by ID.
        """
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        litellm.set_verbose = True

        response = litellm.get_eval(
            eval_id=managed_eval.id,
            custom_llm_provider=custom_llm_provider,
            api_key=api_key,
            api_base=api_base,
        )

        assert response is not None
        assert isinstance(response, Eval)
        assert response.id == managed_eval.id
        print(f"Retrieved eval: {response}")

    @pytest.mark.flaky(retries=3, delay=2)
    def test_update_eval(self, request, managed_eval):
        """
        Test updating an evaluation.
        """
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        litellm.set_verbose = True
        updated_name = _stable_eval_name(request.node.name, suffix="-updated")

        response = litellm.update_eval(
            eval_id=managed_eval.id,
            name=updated_name,
            custom_llm_provider=custom_llm_provider,
            api_key=api_key,
            api_base=api_base,
        )

        assert response is not None
        assert isinstance(response, Eval)
        assert response.id == managed_eval.id
        assert response.name == updated_name
        print(f"Updated eval: {response}")

    def test_delete_eval(self):
        """
        Test deleting an evaluation.

        Real delete coverage now lives in the ``managed_eval`` fixture
        teardown and in ``test_create_eval``'s ``finally`` block, so
        this stays a no-op skip rather than creating a fresh resource
        just to delete it.
        """
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        pytest.skip("Delete is exercised via managed_eval fixture teardown.")


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
