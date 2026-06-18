"""
Routing strategies -- one per scenario.

Each strategy knows how to issue every batch-lifecycle call for one routing
mode. The lifecycle test never branches on the scenario; it drives a strategy.
The three strategies correspond directly to the proxy's file-create routing
priority (litellm/proxy/openai_files_endpoints/files_endpoints.py):

  ModelParamStrategy        - `model=` on file create. The proxy encodes the
                              model into the returned file/batch id, so NO extra
                              routing hint is needed on downstream calls.
  TargetModelNamesStrategy  - managed files (DB-backed, load-balanced). The
                              `target_model_names` hint is repeated on list.
  CustomLlmProviderStrategy - the `custom-llm-provider` header, repeated on
                              every call.
"""

from typing import Any, Dict

from .config import (
    STRATEGY_CUSTOM_LLM_PROVIDER,
    STRATEGY_MODEL_PARAM,
    STRATEGY_TARGET_MODEL_NAMES,
    Case,
)


class _BaseStrategy:
    def __init__(self, case: Case, endpoint: str):
        self.case = case
        self.endpoint = endpoint

    # --- routing hints injected per call; overridden by subclasses --------
    def _create_file_kwargs(self) -> Dict[str, Any]:
        return {}

    def _call_kwargs(self) -> Dict[str, Any]:
        """Hints repeated on retrieve/cancel/delete/create-batch."""
        return {}

    def _list_kwargs(self) -> Dict[str, Any]:
        return {}

    # --- lifecycle operations --------------------------------------------
    def create_file(self, client, input_path: str):
        return client.files.create(
            file=open(input_path, "rb"),
            purpose="batch",
            **self._create_file_kwargs(),
        )

    def create_batch(self, client, input_file_id: str):
        return client.batches.create(
            input_file_id=input_file_id,
            endpoint=self.endpoint,
            completion_window="24h",
            metadata={"e2e": self.case.id},
            **self._call_kwargs(),
        )

    def retrieve_batch(self, client, batch_id: str):
        return client.batches.retrieve(batch_id, **self._call_kwargs())

    def list_batches(self, client):
        return client.batches.list(**self._list_kwargs())

    def cancel_batch(self, client, batch_id: str):
        return client.batches.cancel(batch_id, **self._call_kwargs())

    def delete_file(self, client, file_id: str):
        return client.files.delete(file_id, **self._call_kwargs())


class ModelParamStrategy(_BaseStrategy):
    """Scenario 1: `model=` -> model credentials, id carries the model."""

    def _create_file_kwargs(self) -> Dict[str, Any]:
        # `model` is not a first-class param on files.create in the OpenAI SDK;
        # the proxy reads it from the request body.
        return {"extra_body": {"model": self.case.model}}

    # downstream calls need nothing: the model is encoded in the id.


class TargetModelNamesStrategy(_BaseStrategy):
    """Scenario 2: managed files via `target_model_names` (requires DB)."""

    def _create_file_kwargs(self) -> Dict[str, Any]:
        return {"extra_body": {"target_model_names": self.case.target_model_names}}

    def _list_kwargs(self) -> Dict[str, Any]:
        return {"extra_query": {"target_model_names": self.case.target_model_names}}


class CustomLlmProviderStrategy(_BaseStrategy):
    """Scenario 3: route purely via the `custom-llm-provider` header."""

    def _header(self) -> Dict[str, Any]:
        return {"extra_headers": {"custom-llm-provider": self.case.provider}}

    def _create_file_kwargs(self) -> Dict[str, Any]:
        return self._header()

    def _call_kwargs(self) -> Dict[str, Any]:
        return self._header()

    def _list_kwargs(self) -> Dict[str, Any]:
        return self._header()


_REGISTRY = {
    STRATEGY_MODEL_PARAM: ModelParamStrategy,
    STRATEGY_TARGET_MODEL_NAMES: TargetModelNamesStrategy,
    STRATEGY_CUSTOM_LLM_PROVIDER: CustomLlmProviderStrategy,
}


def build_strategy(case: Case, endpoint: str) -> _BaseStrategy:
    return _REGISTRY[case.strategy](case, endpoint)
