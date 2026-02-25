# What is this?
## Base class for managing resources (files, vector stores, etc.) with target_model_names support
## This provides common functionality for creating, retrieving, and managing resources across multiple models

import base64
import json
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)

from litellm import verbose_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import SpecialEnums

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache
    from litellm.proxy.utils import PrismaClient as _PrismaClient
    from litellm.router import Router as _Router

    Span = Union[_Span, Any]
    InternalUsageCache = _InternalUsageCache
    PrismaClient = _PrismaClient
    Router = _Router
else:
    Span = Any
    InternalUsageCache = Any
    PrismaClient = Any
    Router = Any

# Generic type for resource objects
ResourceObjectType = TypeVar('ResourceObjectType')


class BaseManagedResource(ABC, Generic[ResourceObjectType]):
    """
    Base class for managing resources with target_model_names support.
    
    This class provides common functionality for:
    - Storing unified resource IDs with model mappings
    - Retrieving resources by unified ID
    - Deleting resources across multiple models
    - Creating resources for multiple models
    - Filtering deployments based on model mappings
    
    Subclasses should implement:
    - resource_type: str property
    - table_name: str property
    - create_resource_for_model: method to create resource on a specific model
    - get_unified_resource_id_format: method to generate unified ID format
    """
    
    def __init__(
        self,
        internal_usage_cache: InternalUsageCache,
        prisma_client: PrismaClient,
    ):
        self.internal_usage_cache = internal_usage_cache
        self.prisma_client = prisma_client

    # ============================================================================
    #                          ABSTRACT METHODS
    # ============================================================================

    @property
    @abstractmethod
    def resource_type(self) -> str:
        """
        Return the resource type identifier (e.g., 'file', 'vector_store', 'vector_store_file').
        Used for logging and unified ID generation.
        """
        pass

    @property
    @abstractmethod
    def table_name(self) -> str:
        """
        Return the database table name for this resource type.
        Example: 'litellm_managedfiletable', 'litellm_managedvectorstoretable'
        """
        pass

    @abstractmethod
    def get_unified_resource_id_format(
        self,
        resource_object: ResourceObjectType,
        target_model_names_list: List[str],
    ) -> str:
        """
        Generate the format string for the unified resource ID.
        
        This should return a string that will be base64 encoded.
        Example for files:
            "litellm_proxy:application/json;unified_id,{uuid};target_model_names,{models};..."
        
        Args:
            resource_object: The resource object returned from the provider
            target_model_names_list: List of target model names
            
        Returns:
            Format string to be base64 encoded
        """
        pass

    @abstractmethod
    async def create_resource_for_model(
        self,
        llm_router: Router,
        model: str,
        request_data: Dict[str, Any],
        litellm_parent_otel_span: Span,
    ) -> ResourceObjectType:
        """
        Create a resource for a specific model.
        
        Args:
            llm_router: LiteLLM router instance
            model: Model name to create resource for
            request_data: Request data for resource creation
            litellm_parent_otel_span: OpenTelemetry span for tracing
            
        Returns:
            Resource object from the provider
        """
        pass

    # ============================================================================
    #                     COMMON STORAGE OPERATIONS
    # ============================================================================

    async def store_unified_resource_id(
        self,
        unified_resource_id: str,
        resource_object: Optional[ResourceObjectType],
        litellm_parent_otel_span: Optional[Span],
        model_mappings: Dict[str, str],
        user_api_key_dict: UserAPIKeyAuth,
        additional_db_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store unified resource ID with model mappings in cache and database.
        
        Args:
            unified_resource_id: The unified resource ID (base64 encoded)
            resource_object: The resource object to store (can be None)
            litellm_parent_otel_span: OpenTelemetry span for tracing
            model_mappings: Dictionary mapping model_id -> provider_resource_id
            user_api_key_dict: User API key authentication details
            additional_db_fields: Additional fields to store in database
        """
        verbose_logger.info(
            f"Storing LiteLLM Managed {self.resource_type} with id={unified_resource_id} in cache"
        )
        
        # Prepare cache data
        cache_data = {
            "unified_resource_id": unified_resource_id,
            "resource_object": resource_object,
            "model_mappings": model_mappings,
            "flat_model_resource_ids": list(model_mappings.values()),
            "created_by": user_api_key_dict.user_id,
            "updated_by": user_api_key_dict.user_id,
        }
        
        # Add additional fields if provided
        if additional_db_fields:
            cache_data.update(additional_db_fields)
        
        # Store in cache
        if resource_object is not None:
            await self.internal_usage_cache.async_set_cache(
                key=unified_resource_id,
                value=cache_data,
                litellm_parent_otel_span=litellm_parent_otel_span,
            )

        # Prepare database data
        db_data = {
            "unified_resource_id": unified_resource_id,
            "model_mappings": json.dumps(model_mappings),
            "flat_model_resource_ids": list(model_mappings.values()),
            "created_by": user_api_key_dict.user_id,
            "updated_by": user_api_key_dict.user_id,
        }
        
        # Add resource object if available
        if resource_object is not None:
            # Handle both dict and Pydantic models
            if hasattr(resource_object, "model_dump_json"):
                db_data["resource_object"] = resource_object.model_dump_json()  # type: ignore
            elif isinstance(resource_object, dict):
                db_data["resource_object"] = json.dumps(resource_object)
            
            # Extract storage metadata from hidden params if present
            hidden_params = getattr(resource_object, "_hidden_params", {}) or {}
            if "storage_backend" in hidden_params:
                db_data["storage_backend"] = hidden_params["storage_backend"]
            if "storage_url" in hidden_params:
                db_data["storage_url"] = hidden_params["storage_url"]
        
        # Add additional fields to database
        if additional_db_fields:
            db_data.update(additional_db_fields)

        # Store in database
        table = getattr(self.prisma_client.db, self.table_name)
        result = await table.create(data=db_data)
        
        verbose_logger.debug(
            f"LiteLLM Managed {self.resource_type} with id={unified_resource_id} stored in db: {result}"
        )

    async def get_unified_resource_id(
        self,
        unified_resource_id: str,
        litellm_parent_otel_span: Optional[Span] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve unified resource by ID from cache or database.
        
        Args:
            unified_resource_id: The unified resource ID to retrieve
            litellm_parent_otel_span: OpenTelemetry span for tracing
            
        Returns:
            Dictionary containing resource data or None if not found
        """
        # Check cache first
        result = cast(
            Optional[dict],
            await self.internal_usage_cache.async_get_cache(
                key=unified_resource_id,
                litellm_parent_otel_span=litellm_parent_otel_span,
            ),
        )

        if result:
            return result

        # Check database
        table = getattr(self.prisma_client.db, self.table_name)
        db_object = await table.find_first(
            where={"unified_resource_id": unified_resource_id}
        )

        if db_object:
            return db_object.model_dump()
        
        return None

    async def delete_unified_resource_id(
        self,
        unified_resource_id: str,
        litellm_parent_otel_span: Optional[Span] = None,
    ) -> Optional[ResourceObjectType]:
        """
        Delete unified resource from cache and database.
        
        Args:
            unified_resource_id: The unified resource ID to delete
            litellm_parent_otel_span: OpenTelemetry span for tracing
            
        Returns:
            The deleted resource object or None if not found
        """
        # Get old value from database
        table = getattr(self.prisma_client.db, self.table_name)
        initial_value = await table.find_first(
            where={"unified_resource_id": unified_resource_id}
        )
        
        if initial_value is None:
            raise Exception(
                f"LiteLLM Managed {self.resource_type} with id={unified_resource_id} not found"
            )
        
        # Delete from cache
        await self.internal_usage_cache.async_set_cache(
            key=unified_resource_id,
            value=None,
            litellm_parent_otel_span=litellm_parent_otel_span,
        )
        
        # Delete from database
        await table.delete(where={"unified_resource_id": unified_resource_id})
        
        return initial_value.resource_object

    async def can_user_access_unified_resource_id(
        self,
        unified_resource_id: str,
        user_api_key_dict: UserAPIKeyAuth,
        litellm_parent_otel_span: Optional[Span] = None,
    ) -> bool:
        """
        Check if user has access to the unified resource ID.
        
        Uses get_unified_resource_id() which checks cache first before hitting the database,
        avoiding direct DB queries in the critical request path.
        
        Args:
            unified_resource_id: The unified resource ID to check
            user_api_key_dict: User API key authentication details
            litellm_parent_otel_span: OpenTelemetry span for tracing
            
        Returns:
            True if user has access, False otherwise
        """
        user_id = user_api_key_dict.user_id
        
        # Use cached method instead of direct DB query
        resource = await self.get_unified_resource_id(
            unified_resource_id, litellm_parent_otel_span
        )

        if resource:
            return resource.get("created_by") == user_id
        
        return False

    # ============================================================================
    #                     MODEL MAPPING OPERATIONS
    # ============================================================================

    async def get_model_resource_id_mapping(
        self,
        resource_ids: List[str],
        litellm_parent_otel_span: Span,
    ) -> Dict[str, Dict[str, str]]:
        """
        Get model-specific resource IDs for a list of unified resource IDs.
        
        Args:
            resource_ids: List of unified resource IDs
            litellm_parent_otel_span: OpenTelemetry span for tracing
            
        Returns:
            Dictionary mapping unified_resource_id -> model_id -> provider_resource_id
            
        Example:
            {
                "unified_resource_id_1": {
                    "model_id_1": "provider_resource_id_1",
                    "model_id_2": "provider_resource_id_2"
                }
            }
        """
        resource_id_mapping: Dict[str, Dict[str, str]] = {}

        for resource_id in resource_ids:
            # Get unified resource from cache/db
            unified_resource_object = await self.get_unified_resource_id(
                resource_id, litellm_parent_otel_span
            )

            if unified_resource_object:
                model_mappings = unified_resource_object.get("model_mappings", {})
                
                # Handle both JSON string and dict
                if isinstance(model_mappings, str):
                    model_mappings = json.loads(model_mappings)
                
                resource_id_mapping[resource_id] = model_mappings

        return resource_id_mapping

    # ============================================================================
    #                     RESOURCE CREATION OPERATIONS
    # ============================================================================

    async def create_resource_for_each_model(
        self,
        llm_router: Router,
        request_data: Dict[str, Any],
        target_model_names_list: List[str],
        litellm_parent_otel_span: Span,
    ) -> List[ResourceObjectType]:
        """
        Create a resource for each model in the target list.
        
        Args:
            llm_router: LiteLLM router instance
            request_data: Request data for resource creation
            target_model_names_list: List of target model names
            litellm_parent_otel_span: OpenTelemetry span for tracing
            
        Returns:
            List of resource objects created for each model
        """
        if llm_router is None:
            raise Exception("LLM Router not initialized. Ensure models added to proxy.")
        
        responses = []
        for model in target_model_names_list:
            individual_response = await self.create_resource_for_model(
                llm_router=llm_router,
                model=model,
                request_data=request_data,
                litellm_parent_otel_span=litellm_parent_otel_span,
            )
            responses.append(individual_response)
        return responses

    def generate_unified_resource_id(
        self,
        resource_objects: List[ResourceObjectType],
        target_model_names_list: List[str],
    ) -> str:
        """
        Generate a unified resource ID from multiple resource objects.
        
        Args:
            resource_objects: List of resource objects from different models
            target_model_names_list: List of target model names
            
        Returns:
            Base64 encoded unified resource ID
        """
        # Use the first resource object to generate the format
        unified_id_format = self.get_unified_resource_id_format(
            resource_object=resource_objects[0],
            target_model_names_list=target_model_names_list,
        )
        
        # Convert to URL-safe base64 and strip padding
        base64_unified_id = (
            base64.urlsafe_b64encode(unified_id_format.encode()).decode().rstrip("=")
        )
        
        return base64_unified_id

    def extract_model_mappings_from_responses(
        self,
        resource_objects: List[ResourceObjectType],
    ) -> Dict[str, str]:
        """
        Extract model mappings from resource objects.
        
        Args:
            resource_objects: List of resource objects from different models
            
        Returns:
            Dictionary mapping model_id -> provider_resource_id
        """
        model_mappings: Dict[str, str] = {}

        for resource_object in resource_objects:
            # Get hidden params if available
            hidden_params = getattr(resource_object, "_hidden_params", {}) or {}
            model_resource_id_mapping = hidden_params.get("model_resource_id_mapping")
            
            if model_resource_id_mapping and isinstance(model_resource_id_mapping, dict):
                model_mappings.update(model_resource_id_mapping)

        return model_mappings

    # ============================================================================
    #                     DEPLOYMENT FILTERING
    # ============================================================================

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        request_kwargs: Optional[Dict] = None,
        parent_otel_span: Optional[Span] = None,
        resource_id_key: str = "resource_id",
    ) -> List[Dict]:
        """
        Filter deployments based on model mappings for a resource.
        
        This is used by the router to select only deployments that have
        the resource available.
        
        Args:
            model: Model name
            healthy_deployments: List of healthy deployments
            request_kwargs: Request kwargs containing resource_id and mappings
            parent_otel_span: OpenTelemetry span for tracing
            resource_id_key: Key to use for resource ID in request_kwargs
            
        Returns:
            Filtered list of deployments
        """
        if request_kwargs is None:
            return healthy_deployments

        resource_id = cast(Optional[str], request_kwargs.get(resource_id_key))
        model_resource_id_mapping = cast(
            Optional[Dict[str, Dict[str, str]]],
            request_kwargs.get("model_resource_id_mapping"),
        )
        
        allowed_model_ids = []
        if resource_id and model_resource_id_mapping:
            model_id_dict = model_resource_id_mapping.get(resource_id, {})
            allowed_model_ids = list(model_id_dict.keys())

        if len(allowed_model_ids) == 0:
            return healthy_deployments

        return [
            deployment
            for deployment in healthy_deployments
            if deployment.get("model_info", {}).get("id") in allowed_model_ids
        ]

    # ============================================================================
    #                     UTILITY METHODS
    # ============================================================================

    def get_unified_id_prefix(self) -> str:
        """
        Get the prefix for unified IDs for this resource type.
        
        Returns:
            Prefix string (e.g., "litellm_proxy:")
        """
        return SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value

    async def list_user_resources(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        limit: Optional[int] = None,
        after: Optional[str] = None,
        additional_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        List resources created by a user.
        
        Args:
            user_api_key_dict: User API key authentication details
            limit: Maximum number of resources to return
            after: Cursor for pagination
            additional_filters: Additional filters to apply
            
        Returns:
            Dictionary with list of resources and pagination info
        """
        where_clause: Dict[str, Any] = {}
        
        # Filter by user who created the resource
        if user_api_key_dict.user_id:
            where_clause["created_by"] = user_api_key_dict.user_id
        
        if after:
            where_clause["id"] = {"gt": after}
        
        # Add additional filters
        if additional_filters:
            where_clause.update(additional_filters)
        
        # Fetch resources
        fetch_limit = limit or 20
        table = getattr(self.prisma_client.db, self.table_name)
        resources = await table.find_many(
            where=where_clause,
            take=fetch_limit,
            order={"created_at": "desc"},
        )
        
        resource_objects: List[Any] = []
        for resource in resources:
            try:
                # Stop once we have enough
                if len(resource_objects) >= (limit or 20):
                    break

                # Parse resource object
                resource_data = resource.resource_object
                if isinstance(resource_data, str):
                    resource_data = json.loads(resource_data)
                
                # Set unified ID
                if hasattr(resource_data, "id"):
                    resource_data.id = resource.unified_resource_id
                elif isinstance(resource_data, dict):
                    resource_data["id"] = resource.unified_resource_id
                
                resource_objects.append(resource_data)

            except Exception as e:
                verbose_logger.warning(
                    f"Failed to parse {self.resource_type} object "
                    f"{resource.unified_resource_id}: {e}"
                )
                continue
        
        return {
            "object": "list",
            "data": resource_objects,
            "first_id": resource_objects[0].id if resource_objects else None,
            "last_id": resource_objects[-1].id if resource_objects else None,
            "has_more": len(resource_objects) == (limit or 20),
        }
