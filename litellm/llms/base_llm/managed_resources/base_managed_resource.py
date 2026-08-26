# What is this?
## Base class for managing resources (files, vector stores, etc.) with target_model_names support
## This provides common functionality for creating, retrieving, and managing resources across multiple models

import base64
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Final, Generic, TypeVar, cast

from litellm import verbose_logger
from litellm.llms.base_llm.managed_resources.isolation import (
    build_list_page,
    build_owner_filter,
    can_access_resource,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import SpecialEnums

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache
    from litellm.proxy.utils import PrismaClient as _PrismaClient
    from litellm.router import Router as _Router

    Span = _Span | Any
    InternalUsageCache = _InternalUsageCache
    PrismaClient = _PrismaClient
    Router = _Router
else:
    Span = Any
    InternalUsageCache = Any
    PrismaClient = Any
    Router = Any

# Generic type for resource objects
ResourceObjectType = TypeVar("ResourceObjectType")


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

    @property
    @abstractmethod
    def table_name(self) -> str:
        """
        Return the database table name for this resource type.
        Example: 'litellm_managedfiletable', 'litellm_managedvectorstoretable'
        """

    @abstractmethod
    def get_unified_resource_id_format(
        self,
        resource_object: ResourceObjectType,
        target_model_names_list: list[str],
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

    @abstractmethod
    async def create_resource_for_model(
        self,
        llm_router: Router,
        model: str,
        request_data: dict[str, Any],
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

    # ============================================================================
    #                     COMMON STORAGE OPERATIONS
    # ============================================================================

    async def store_unified_resource_id(
        self,
        unified_resource_id: str,
        resource_object: ResourceObjectType | None,
        litellm_parent_otel_span: Span | None,
        model_mappings: dict[str, str],
        user_api_key_dict: UserAPIKeyAuth,
        additional_db_fields: dict[str, Any] | None = None,
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
        verbose_logger.info("Storing LiteLLM Managed %s with id=%s in cache", self.resource_type, unified_resource_id)

        # Prepare cache data
        cache_data: Final = {
            "unified_resource_id": unified_resource_id,
            "resource_object": resource_object,
            "model_mappings": model_mappings,
            "flat_model_resource_ids": list(model_mappings.values()),
            "created_by": user_api_key_dict.user_id,
            "team_id": user_api_key_dict.team_id,
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
        db_data: Final = {
            "unified_resource_id": unified_resource_id,
            "model_mappings": json.dumps(model_mappings),
            "flat_model_resource_ids": list(model_mappings.values()),
            "created_by": user_api_key_dict.user_id,
            "team_id": user_api_key_dict.team_id,
            "updated_by": user_api_key_dict.user_id,
        }

        # Add resource object if available
        if resource_object is not None:
            # Handle both dict and Pydantic models
            if hasattr(resource_object, "model_dump_json"):
                db_data["resource_object"] = resource_object.model_dump_json()
            elif isinstance(resource_object, dict):
                db_data["resource_object"] = json.dumps(resource_object)

            # Extract storage metadata from hidden params if present
            hidden_params: Final = getattr(resource_object, "_hidden_params", {}) or {}
            if "storage_backend" in hidden_params:
                db_data["storage_backend"] = hidden_params["storage_backend"]
            if "storage_url" in hidden_params:
                db_data["storage_url"] = hidden_params["storage_url"]

        # Add additional fields to database
        if additional_db_fields:
            db_data.update(additional_db_fields)

        # Store in database
        table: Final = getattr(self.prisma_client.db, self.table_name)
        result: Final = await table.create(data=db_data)

        verbose_logger.debug(
            "LiteLLM Managed %s with id=%s stored in db: %s", self.resource_type, unified_resource_id, result
        )

    async def get_unified_resource_id(
        self,
        unified_resource_id: str,
        litellm_parent_otel_span: Span | None = None,
    ) -> dict[str, Any] | None:
        """
        Retrieve unified resource by ID from cache or database.

        Args:
            unified_resource_id: The unified resource ID to retrieve
            litellm_parent_otel_span: OpenTelemetry span for tracing

        Returns:
            Dictionary containing resource data or None if not found
        """
        # Check cache first
        result: Final = cast(
            dict | None,
            await self.internal_usage_cache.async_get_cache(
                key=unified_resource_id,
                litellm_parent_otel_span=litellm_parent_otel_span,
            ),
        )

        if result:
            return result

        # Check database
        table: Final = getattr(self.prisma_client.db, self.table_name)
        db_object: Final = await table.find_first(where={"unified_resource_id": unified_resource_id})

        if db_object:
            return db_object.model_dump()

        return None

    async def delete_unified_resource_id(
        self,
        unified_resource_id: str,
        litellm_parent_otel_span: Span | None = None,
    ) -> ResourceObjectType | None:
        """
        Delete unified resource from cache and database.

        Args:
            unified_resource_id: The unified resource ID to delete
            litellm_parent_otel_span: OpenTelemetry span for tracing

        Returns:
            The deleted resource object or None if not found
        """
        # Get old value from database
        table: Final = getattr(self.prisma_client.db, self.table_name)
        initial_value: Final = await table.find_first(where={"unified_resource_id": unified_resource_id})

        if initial_value is None:
            raise Exception(f"LiteLLM Managed {self.resource_type} with id={unified_resource_id} not found")

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
        litellm_parent_otel_span: Span | None = None,
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
        # Use cached method instead of direct DB query
        resource: Final = await self.get_unified_resource_id(unified_resource_id, litellm_parent_otel_span)

        if resource:
            return can_access_resource(
                user_api_key_dict=user_api_key_dict,
                created_by=resource.get("created_by"),
                resource_team_id=resource.get("team_id"),
            )

        return False

    # ============================================================================
    #                     MODEL MAPPING OPERATIONS
    # ============================================================================

    async def get_model_resource_id_mapping(
        self,
        resource_ids: list[str],
        litellm_parent_otel_span: Span,
    ) -> dict[str, dict[str, str]]:
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
        resource_id_mapping: Final[dict[str, dict[str, str]]] = {}

        for resource_id in resource_ids:
            # Get unified resource from cache/db
            unified_resource_object = await self.get_unified_resource_id(resource_id, litellm_parent_otel_span)

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
        request_data: dict[str, Any],
        target_model_names_list: list[str],
        litellm_parent_otel_span: Span,
    ) -> list[ResourceObjectType]:
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

        responses: Final = []
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
        resource_objects: list[ResourceObjectType],
        target_model_names_list: list[str],
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
        unified_id_format: Final = self.get_unified_resource_id_format(
            resource_object=resource_objects[0],
            target_model_names_list=target_model_names_list,
        )

        # Convert to URL-safe base64 and strip padding
        base64_unified_id: Final = base64.urlsafe_b64encode(unified_id_format.encode()).decode().rstrip("=")

        return base64_unified_id

    def extract_model_mappings_from_responses(
        self,
        resource_objects: list[ResourceObjectType],
    ) -> dict[str, str]:
        """
        Extract model mappings from resource objects.

        Args:
            resource_objects: List of resource objects from different models

        Returns:
            Dictionary mapping model_id -> provider_resource_id
        """
        model_mappings: Final[dict[str, str]] = {}

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
        healthy_deployments: list,
        request_kwargs: dict | None = None,
        parent_otel_span: Span | None = None,
        resource_id_key: str = "resource_id",
    ) -> list[dict]:
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

        resource_id: Final = cast(str | None, request_kwargs.get(resource_id_key))
        model_resource_id_mapping: Final = cast(
            dict[str, dict[str, str]] | None,
            request_kwargs.get("model_resource_id_mapping"),
        )

        allowed_model_ids = []
        if resource_id and model_resource_id_mapping:
            model_id_dict: Final = model_resource_id_mapping.get(resource_id, {})
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
        limit: int | None = None,
        after: str | None = None,
        additional_filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        owner_filter: Final = build_owner_filter(user_api_key_dict)
        if owner_filter is None:
            return build_list_page([])

        where_clause: Final[dict[str, Any]] = {**owner_filter}

        if after:
            where_clause["id"] = {"gt": after}

        # Add additional filters
        if additional_filters:
            where_clause.update(additional_filters)

        # Fetch resources
        fetch_limit: Final = limit or 20
        table: Final = getattr(self.prisma_client.db, self.table_name)
        resources: Final = await table.find_many(
            where=where_clause,
            take=fetch_limit,
            order={"created_at": "desc"},
        )

        resource_objects: Final[list[Any]] = []
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
                    "Failed to parse %s object %s: %s", self.resource_type, resource.unified_resource_id, e
                )
                continue

        return build_list_page(resource_objects, has_more=len(resource_objects) == (limit or 20))
