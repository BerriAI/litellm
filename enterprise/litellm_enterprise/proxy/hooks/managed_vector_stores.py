# What is this?
## This hook is used to manage vector stores with target_model_names support
## It allows creating vector stores across multiple models and managing them with unified IDs

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, Final, List, Optional, Union, cast

from fastapi import HTTPException

from litellm import Router, verbose_logger
from litellm._uuid import uuid
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.base_llm.managed_resources import BaseManagedResource
from litellm.llms.base_llm.managed_resources.isolation import can_access_resource
from litellm.llms.base_llm.managed_resources.utils import (
    extract_model_id_from_unified_id,
    extract_provider_resource_id_from_unified_id,
    extract_target_model_names_from_unified_id,
    generate_unified_id_string,
    is_base64_encoded_unified_id,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import LLMResponseTypes
from litellm.types.vector_stores import (
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.caching.caching import DualCache
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache
    from litellm.proxy.utils import PrismaClient as _PrismaClient

    Span = _Span
    InternalUsageCache = _InternalUsageCache
    PrismaClient = _PrismaClient
else:
    Span = Any
    InternalUsageCache = Any
    PrismaClient = Any


_WRITE_VECTOR_STORE_CALL_TYPES: Final = frozenset({"avector_store_update", "avector_store_delete"})
_ROUTED_VECTOR_STORE_CALL_TYPES: Final = _WRITE_VECTOR_STORE_CALL_TYPES | {
    "avector_store_search",
    "avector_store_retrieve",
}


def _route_to_provider_vector_store(data: dict[str, object], decoded_id: str) -> dict[str, object]:
    routing_model: Final = extract_model_id_from_unified_id(decoded_id) or next(
        iter(extract_target_model_names_from_unified_id(decoded_id)), None
    )
    provider_resource_id: Final = extract_provider_resource_id_from_unified_id(decoded_id)
    return {
        **data,
        **({"model": routing_model} if routing_model else {}),
        **({"vector_store_id": provider_resource_id} if provider_resource_id else {}),
    }


class _PROXY_LiteLLMManagedVectorStores(
    CustomLogger, BaseManagedResource[VectorStoreCreateResponse]
):
    """
    Managed vector stores with target_model_names support.
    
    This class provides functionality to:
    - Create vector stores across multiple models
    - Retrieve vector stores by unified ID
    - Update or delete a vector store created on one model by unified ID
    - List vector stores created by a user
    """

    def __init__(
        self, internal_usage_cache: InternalUsageCache, prisma_client: PrismaClient
    ):
        CustomLogger.__init__(self)
        BaseManagedResource.__init__(self, internal_usage_cache, prisma_client)

    # ============================================================================
    #                     ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================================

    @property
    def resource_type(self) -> str:
        """Return the resource type identifier."""
        return "vector_store"

    @property
    def table_name(self) -> str:
        """Return the database table name for vector stores."""
        # Prisma converts model name LiteLLM_ManagedVectorStoreTable to litellm_managedvectorstoretable
        return "litellm_managedvectorstoretable"

    def get_unified_resource_id_format(
        self,
        resource_object: VectorStoreCreateResponse,
        target_model_names_list: List[str],
    ) -> str:
        """
        Generate the format string for the unified vector store ID.
        
        Format:
        litellm_proxy:vector_store;unified_id,<uuid>;target_model_names,<models>;resource_id,<vs_id>;model_id,<model_id>
        """
        # VectorStoreCreateResponse is a TypedDict, so resource_object is a dictionary
        # Extract provider resource ID from the response
        provider_resource_id = resource_object.get("id", "")
        
        # Model ID is stored in hidden params if the response object supports it
        # For TypedDict responses, we need to check if _hidden_params was added
        hidden_params: Mapping[str, object] = {}
        if hasattr(resource_object, "_hidden_params"):
            hidden_params = cast(Mapping[str, object], getattr(resource_object, "_hidden_params", {}) or {})
        model_id: Final = cast(str, hidden_params.get("model_id", ""))

        return generate_unified_id_string(
            resource_type=self.resource_type,
            unified_uuid=str(uuid.uuid4()),
            target_model_names=target_model_names_list,
            provider_resource_id=provider_resource_id,
            model_id=model_id,
        )

    async def create_resource_for_model(
        self,
        llm_router: Router,
        model: str,
        request_data: dict[str, object] | VectorStoreCreateOptionalRequestParams,
        litellm_parent_otel_span: Span | None,
    ) -> VectorStoreCreateResponse:
        """
        Create a vector store for a specific model.
        
        Args:
            llm_router: LiteLLM router instance
            model: Model name to create vector store for
            request_data: Request data for vector store creation
            litellm_parent_otel_span: OpenTelemetry span for tracing
            
        Returns:
            VectorStoreCreateResponse from the provider
        """
        # Use the router to create the vector store
        response: Final = await llm_router.avector_store_create(model=model, **request_data)
        return cast(VectorStoreCreateResponse, response)

    # ============================================================================
    #                     VECTOR STORE CRUD OPERATIONS
    # ============================================================================

    async def acreate_vector_store(
        self,
        create_request: VectorStoreCreateOptionalRequestParams,
        llm_router: Router,
        target_model_names_list: List[str],
        litellm_parent_otel_span: Span | None,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> VectorStoreCreateResponse:
        """
        Create a vector store across multiple models.
        
        Args:
            create_request: Vector store creation request parameters
            llm_router: LiteLLM router instance
            target_model_names_list: List of target model names
            litellm_parent_otel_span: OpenTelemetry span for tracing
            user_api_key_dict: User API key authentication details
            
        Returns:
            VectorStoreCreateResponse with unified ID
        """
        verbose_logger.info(
            f"Creating managed vector store for models: {target_model_names_list}"
        )

        # Create vector store for each model
        # Convert TypedDict to Dict[str, Any] for base class compatibility
        request_data_dict: Dict[str, object] = dict(create_request)
        responses = await self.create_resource_for_each_model(
            llm_router=llm_router,
            request_data=request_data_dict,
            target_model_names_list=target_model_names_list,
            litellm_parent_otel_span=litellm_parent_otel_span,
        )

        # Generate unified ID
        unified_id = self.generate_unified_resource_id(
            resource_objects=responses,
            target_model_names_list=target_model_names_list,
        )

        # Extract model mappings from responses
        model_mappings: Dict[str, str] = {}
        for response in responses:
            hidden_params = getattr(response, "_hidden_params", {}) or {}
            model_id = hidden_params.get("model_id")
            if model_id:
                # VectorStoreCreateResponse is a TypedDict, use dict access
                model_mappings[model_id] = response["id"]

        verbose_logger.debug(
            f"Created vector stores with model mappings: {model_mappings}"
        )

        # Store in database
        await self.store_unified_resource_id(
            unified_resource_id=unified_id,
            resource_object=responses[0],  # Store first response as template
            litellm_parent_otel_span=litellm_parent_otel_span,
            model_mappings=model_mappings,
            user_api_key_dict=user_api_key_dict,
        )

        # Return response with unified ID
        # VectorStoreCreateResponse is a TypedDict, so we need to create a new dict with the unified ID
        response = responses[0].copy()
        response["id"] = unified_id
        
        verbose_logger.info(
            f"Successfully created managed vector store with unified ID: {unified_id}"
        )

        return response

    async def alist_vector_stores(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        limit: Optional[int] = None,
        after: Optional[str] = None,
        order: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        List vector stores created by a user.
        
        Args:
            user_api_key_dict: User API key authentication details
            limit: Maximum number of vector stores to return
            after: Cursor for pagination
            order: Sort order ('asc' or 'desc')
            
        Returns:
            Dictionary with list of vector stores and pagination info
        """
        # Use the base class method
        return await self.list_user_resources(
            user_api_key_dict=user_api_key_dict,
            limit=limit,
            after=after,
        )

    # ============================================================================
    #                     ACCESS CONTROL
    # ============================================================================

    async def check_vector_store_access(
        self, vector_store_id: str, user_api_key_dict: UserAPIKeyAuth
    ) -> bool:
        """
        Check if user has access to a vector store.
        
        Args:
            vector_store_id: The unified vector store ID
            user_api_key_dict: User API key authentication details
            
        Returns:
            True if user has access, False otherwise
        """
        is_unified_id = is_base64_encoded_unified_id(vector_store_id)
        
        if is_unified_id:
            # Check access for managed vector store
            return await self.can_user_access_unified_resource_id(
                vector_store_id,
                user_api_key_dict,
            )
        
        # Not a managed vector store, allow access
        return True

    async def check_managed_vector_store_access(
        self, data: Dict, user_api_key_dict: UserAPIKeyAuth
    ) -> bool:
        """
        Check if user has access to a managed vector store in request data.
        
        Args:
            data: Request data containing vector_store_id
            user_api_key_dict: User API key authentication details
            
        Returns:
            True if this is a managed vector store and user has access
            
        Raises:
            HTTPException: If user doesn't have access
        """
        vector_store_id = cast(Optional[str], data.get("vector_store_id"))
        is_unified_id = (
            is_base64_encoded_unified_id(vector_store_id)
            if vector_store_id
            else False
        )
        
        if not (is_unified_id and vector_store_id):
            return False
        managed_store: Final = await self.get_unified_resource_id(vector_store_id, user_api_key_dict.parent_otel_span)
        if managed_store is None:
            raise HTTPException(status_code=404, detail=f"Vector store not found: {vector_store_id}")
        if can_access_resource(
            user_api_key_dict=user_api_key_dict,
            created_by=managed_store.get("created_by"),
            resource_team_id=managed_store.get("team_id"),
        ):
            return True
        raise HTTPException(
            status_code=403,
            detail=f"User {user_api_key_dict.user_id} does not have access to vector store {vector_store_id}",
        )

    # ============================================================================
    #                     PRE-CALL HOOK (For Router Integration)
    # ============================================================================

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: "DualCache",
        data: Dict,
        call_type: str,
    ) -> Union[Exception, str, Dict, None]:
        """
        Pre-call hook to handle vector store operations.
        
        This hook intercepts vector store requests and:
        - Validates access for managed vector stores
        - Transforms unified IDs to provider-specific IDs
        - Adds model routing information
        
        Args:
            user_api_key_dict: User API key authentication details
            cache: Cache instance
            data: Request data
            call_type: Type of call being made
            
        Returns:
            Modified request data or None
        """
        if call_type not in _ROUTED_VECTOR_STORE_CALL_TYPES:
            return data
        await self.check_managed_vector_store_access(data, user_api_key_dict)
        vector_store_id: Final = data.get("vector_store_id")
        decoded_id: Final = is_base64_encoded_unified_id(vector_store_id) if isinstance(vector_store_id, str) else False
        if not decoded_id:
            return data
        created_on_several_models: Final = len(extract_target_model_names_from_unified_id(decoded_id)) > 1
        if call_type in _WRITE_VECTOR_STORE_CALL_TYPES and created_on_several_models:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Vector store {vector_store_id} was created on several models, so updating or deleting it by "
                    "this id would only change the first model's store. Update or delete each model's store by its "
                    "provider id instead."
                ),
            )
        return _route_to_provider_vector_store(data, decoded_id)

    # ============================================================================
    #                     POST-CALL HOOK (For Response Transformation)
    # ============================================================================

    async def async_post_call_success_hook(
        self,
        data: Dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        """
        Post-call hook to transform responses.
        
        This hook can be used to transform responses if needed.
        For now, it just passes through the response.
        
        Args:
            data: Request data
            user_api_key_dict: User API key authentication details
            response: Response from the provider
            
        Returns:
            Potentially modified response
        """
        # Currently no transformation needed
        return response

    # ============================================================================
    #                     DEPLOYMENT FILTERING
    # ============================================================================

    async def async_filter_deployments(  # type: ignore[override]
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List] = None,
        request_kwargs: Optional[Dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> List[Dict]:
        """
        Filter deployments based on vector store availability.
        
        This is used by the router to select only deployments that have
        the vector store available.
        
        Note: This method signature is a compromise between CustomLogger and BaseManagedResource
        parent classes which have incompatible signatures. The type: ignore[override] is necessary
        due to this multiple inheritance conflict.
        
        Args:
            model: Model name
            healthy_deployments: List of healthy deployments
            messages: Messages (unused for vector stores, required by CustomLogger interface)
            request_kwargs: Request kwargs containing vector_store_id and mappings
            parent_otel_span: OpenTelemetry span for tracing
            
        Returns:
            Filtered list of deployments
        """
        return await BaseManagedResource.async_filter_deployments(
            self,
            model=model,
            healthy_deployments=healthy_deployments,
            request_kwargs=request_kwargs,
            parent_otel_span=parent_otel_span,
            resource_id_key="vector_store_id",
        )
PROXY_LiteLLMManagedVectorStores = _PROXY_LiteLLMManagedVectorStores
