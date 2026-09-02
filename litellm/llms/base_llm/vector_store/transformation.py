from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, NoReturn, Protocol, runtime_checkable

import httpx
from pydantic import TypeAdapter

from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import (
    VECTOR_STORE_OPENAI_PARAMS,
    BaseVectorStoreAuthCredentials,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreIndexEndpoints,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.router import Router

    from ..chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any


@runtime_checkable
class VectorStoreEmbeddingExecutor(Protocol):
    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse: ...

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse: ...


@dataclass(frozen=True, slots=True)
class LiteLLMVectorStoreEmbeddingExecutor:
    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        import litellm

        return litellm.embedding(  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportUnknownVariableType]  # provider kwargs are intentionally dynamic
            model=model,
            input=[query],  # mutable-ok: LiteLLM embedding requires a mutable input list
            **dict(configuration),  # pyright: ignore[reportArgumentType]  # provider-specific embedding config is validated downstream  # mutable-ok: kwargs require a concrete dict
        )

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        import litellm

        return await litellm.aembedding(  # pyright: ignore[reportUnknownMemberType]  # provider kwargs are intentionally dynamic
            model=model,
            input=[query],  # mutable-ok: LiteLLM embedding requires a mutable input list
            **dict(configuration),  # pyright: ignore[reportArgumentType]  # provider-specific embedding config is validated downstream  # mutable-ok: kwargs require a concrete dict
        )


@dataclass(frozen=True, slots=True)
class RouterVectorStoreEmbeddingExecutor:
    router: Router
    metadata: Mapping[str, object]

    def _embedding_kwargs(self, configuration: Mapping[str, object]) -> Mapping[str, object]:
        configured_metadata: Final = configuration.get("metadata")
        metadata: Final = {
            **(configured_metadata if isinstance(configured_metadata, Mapping) else {}),
            **self.metadata,
        }
        return {
            **{key: value for key, value in configuration.items() if key not in ("input", "metadata", "model")},
            "metadata": metadata,
        }

    def _router_serves(self, model: str) -> bool:
        team_id: Final = self.metadata.get("user_api_key_team_id")
        resolved: Final = self.router.resolved_litellm_models(model, team_id if isinstance(team_id, str) else None)
        deployment_models: Final = (
            deployment.get("litellm_params", {}).get("model") for deployment in self.router.get_model_list() or ()
        )
        return bool(resolved) or model in deployment_models

    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        embedding_kwargs: Final = self._embedding_kwargs(configuration)
        if not self._router_serves(model):
            return LiteLLMVectorStoreEmbeddingExecutor().embed(model, query, embedding_kwargs)
        return self.router.embedding(  # pyright: ignore[reportUnknownMemberType]  # Router embedding input retains a legacy untyped list
            model=model,
            input=[query],  # mutable-ok: Router embedding requires a mutable input list
            **embedding_kwargs,  # pyright: ignore[reportArgumentType]  # provider kwargs are intentionally dynamic
        )

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        embedding_kwargs: Final = self._embedding_kwargs(configuration)
        if not self._router_serves(model):
            return await LiteLLMVectorStoreEmbeddingExecutor().aembed(model, query, embedding_kwargs)
        return await self.router.aembedding(  # pyright: ignore[reportUnknownMemberType]  # Router embedding input retains a legacy untyped list
            model=model,
            input=[query],  # mutable-ok: Router embedding requires a mutable input list
            **embedding_kwargs,  # pyright: ignore[reportArgumentType]  # provider kwargs are intentionally dynamic
        )


class BaseVectorStoreConfig:
    def get_supported_openai_params(self, model: str) -> list[VECTOR_STORE_OPENAI_PARAMS]:
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        drop_params: bool,
    ) -> dict:
        return optional_params

    @abstractmethod
    def get_auth_credentials(self, litellm_params: dict) -> BaseVectorStoreAuthCredentials:
        pass

    @abstractmethod
    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        pass

    @abstractmethod
    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | list[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        pass

    async def atransform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | list[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        """
        Optional async version of transform_search_vector_store_request.
        If not implemented, the handler will fall back to the sync version.
        Providers that need to make async calls (e.g., generating embeddings) should override this.
        """
        # Default implementation: call the sync version
        return self.transform_search_vector_store_request(
            vector_store_id=vector_store_id,
            query=query,
            vector_store_search_optional_params=vector_store_search_optional_params,
            api_base=api_base,
            litellm_logging_obj=litellm_logging_obj,
            litellm_params=litellm_params,
            extra_body=extra_body,
        )

    @abstractmethod
    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        pass

    @abstractmethod
    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> tuple[str, dict]:
        pass

    @abstractmethod
    def transform_create_vector_store_response(self, response: httpx.Response) -> VectorStoreCreateResponse:
        pass

    @abstractmethod
    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        return {}

    @abstractmethod
    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        if api_base is None:
            raise ValueError("api_base is required")
        return api_base

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        from ..chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: str | None = None,
    ) -> tuple[dict, bytes | None]:
        """Optionally sign or modify the request before sending.

        Providers like AWS Bedrock require SigV4 signing. Providers that don't
        require any signing can simply return the headers unchanged and ``None``
        for the signed body.
        """
        return headers, None

    def calculate_vector_store_cost(
        self,
        response: VectorStoreSearchResponse,
    ) -> tuple[float, float]:
        return 0.0, 0.0


_EMPTY_EMBEDDING_CONFIGURATION: Final[Mapping[str, object]] = MappingProxyType({})
_QUERY_VECTOR: Final = TypeAdapter(list[float])


class BaseQueryEmbeddingVectorStoreConfig(BaseVectorStoreConfig):
    @abstractmethod
    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> tuple[str, dict[str, object]]:
        pass

    async def atransform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> tuple[str, dict[str, object]]:
        return self.transform_search_vector_store_request(
            vector_store_id=vector_store_id,
            query=query,
            vector_store_search_optional_params=vector_store_search_optional_params,
            api_base=api_base,
            litellm_logging_obj=litellm_logging_obj,
            litellm_params=litellm_params,
            extra_body=extra_body,
            embedding_executor=embedding_executor,
        )

    @staticmethod
    def query_text(query: str | Sequence[str]) -> str:
        return query if isinstance(query, str) else " ".join(query)

    @staticmethod
    def query_embedding_model(litellm_params: Mapping[str, object]) -> str:
        embedding_model: Final = litellm_params.get("litellm_embedding_model")
        if isinstance(embedding_model, str) and embedding_model:
            return embedding_model
        raise ValueError(
            "litellm_embedding_model is required in litellm_params for this vector store. "
            "Example: litellm_params['litellm_embedding_model'] = 'openai/text-embedding-3-small'"
        )

    @staticmethod
    def query_embedding_configuration(litellm_params: Mapping[str, object]) -> Mapping[str, object]:
        configuration: Final = litellm_params.get("litellm_embedding_config")
        if isinstance(configuration, Mapping):
            return {str(key): value for key, value in configuration.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # litellm_params is an untyped dict, keys are re-validated as str here
        return _EMPTY_EMBEDDING_CONFIGURATION

    def embed_query(
        self,
        query_text: str,
        litellm_params: Mapping[str, object],
        embedding_executor: VectorStoreEmbeddingExecutor | None,
    ) -> Sequence[float]:
        model: Final = self.query_embedding_model(litellm_params)
        configuration: Final = self.query_embedding_configuration(litellm_params)
        executor: Final = (
            embedding_executor if embedding_executor is not None else LiteLLMVectorStoreEmbeddingExecutor()
        )
        try:
            response: Final = executor.embed(model, query_text, configuration)
        except Exception as e:
            raise Exception(f"Failed to generate embedding for query: {e}")
        return _QUERY_VECTOR.validate_python(response.data[0]["embedding"])  # pyright: ignore[reportUnknownMemberType]  # EmbeddingResponse.data is an untyped list, the vector is validated here

    async def aembed_query(
        self,
        query_text: str,
        litellm_params: Mapping[str, object],
        embedding_executor: VectorStoreEmbeddingExecutor | None,
    ) -> Sequence[float]:
        model: Final = self.query_embedding_model(litellm_params)
        configuration: Final = self.query_embedding_configuration(litellm_params)
        executor: Final = (
            embedding_executor if embedding_executor is not None else LiteLLMVectorStoreEmbeddingExecutor()
        )
        try:
            response: Final = await executor.aembed(model, query_text, configuration)
        except Exception as e:
            raise Exception(f"Failed to generate embedding for query: {e}")
        return _QUERY_VECTOR.validate_python(response.data[0]["embedding"])  # pyright: ignore[reportUnknownMemberType]  # EmbeddingResponse.data is an untyped list, the vector is validated here


class BaseDirectVectorStoreConfig(BaseVectorStoreConfig):
    """
    Base config for vector store providers whose datastore has no HTTP API
    (e.g. Valkey over RESP). Instead of transforming to an httpx request, the
    config executes the search itself via (a)execute_search_vector_store_request.
    """

    @abstractmethod
    def execute_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> VectorStoreSearchResponse:
        pass

    @abstractmethod
    async def aexecute_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> VectorStoreSearchResponse:
        pass

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
    ) -> NoReturn:
        raise NotImplementedError("Direct vector store providers execute the search themselves; no HTTP request shape")

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> NoReturn:
        raise NotImplementedError("Direct vector store providers execute the search themselves; no HTTP response shape")

    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> NoReturn:
        raise NotImplementedError

    def transform_create_vector_store_response(self, response: httpx.Response) -> NoReturn:
        raise NotImplementedError

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        return api_base or ""

    def get_auth_credentials(self, litellm_params: Mapping[str, object]) -> BaseVectorStoreAuthCredentials:
        return BaseVectorStoreAuthCredentials()

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return VectorStoreIndexEndpoints(read=[], write=[])  # mutable-ok: the TypedDict declares list fields
