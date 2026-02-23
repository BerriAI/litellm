"""
Search Tool Registry for managing search tool configurations.
"""
from datetime import datetime, timezone
from typing import List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.utils import PrismaClient
from litellm.types.search import SearchTool


class SearchToolRegistry:
    """    
    Handles adding, removing, and getting search tools in DB + in memory.
    """

    def __init__(self):
        pass

    @staticmethod
    def _convert_prisma_to_dict(prisma_obj) -> dict:
        """
        Convert Prisma result to dict with datetime objects as ISO format strings.
        
        Args:
            prisma_obj: Prisma model instance
            
        Returns:
            Dict with datetime fields converted to ISO strings
        """
        result = dict(prisma_obj)
        # Convert datetime objects to ISO format strings
        if "created_at" in result and result["created_at"]:
            result["created_at"] = result["created_at"].isoformat()
        if "updated_at" in result and result["updated_at"]:
            result["updated_at"] = result["updated_at"].isoformat()
        return result

    ###########################################################
    ########### DB management helpers for search tools ########
    ###########################################################
    
    async def add_search_tool_to_db(
        self, search_tool: SearchTool, prisma_client: PrismaClient
    ):
        """
        Add a search tool to the database.
        
        Args:
            search_tool: Search tool configuration
            prisma_client: Prisma client instance
            
        Returns:
            Dict with created search tool data
        """
        try:
            search_tool_name = search_tool.get("search_tool_name")
            litellm_params: str = safe_dumps(dict(search_tool.get("litellm_params", {})))
            search_tool_info: str = safe_dumps(search_tool.get("search_tool_info", {}))

            # Create search tool in DB
            created_search_tool = await prisma_client.db.litellm_searchtoolstable.create(
                data={
                    "search_tool_name": search_tool_name,
                    "litellm_params": litellm_params,
                    "search_tool_info": search_tool_info,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

            # Add search_tool_id to the returned search tool object
            search_tool_dict = dict(search_tool)
            search_tool_dict["search_tool_id"] = created_search_tool.search_tool_id
            search_tool_dict["created_at"] = created_search_tool.created_at.isoformat()
            search_tool_dict["updated_at"] = created_search_tool.updated_at.isoformat()

            return search_tool_dict
        except Exception as e:
            verbose_proxy_logger.exception(f"Error adding search tool to DB: {str(e)}")
            raise Exception(f"Error adding search tool to DB: {str(e)}")

    async def delete_search_tool_from_db(
        self, search_tool_id: str, prisma_client: PrismaClient
    ):
        """
        Delete a search tool from the database.
        
        Args:
            search_tool_id: ID of search tool to delete
            prisma_client: Prisma client instance
            
        Returns:
            Dict with success message
        """
        try:
            # Get search tool before deletion for response
            existing_tool = await prisma_client.db.litellm_searchtoolstable.find_unique(
                where={"search_tool_id": search_tool_id}
            )
            
            if not existing_tool:
                raise Exception(f"Search tool with ID {search_tool_id} not found")
            
            # Delete from DB
            await prisma_client.db.litellm_searchtoolstable.delete(
                where={"search_tool_id": search_tool_id}
            )

            return {
                "message": f"Search tool {search_tool_id} deleted successfully",
                "search_tool_name": existing_tool.search_tool_name,
            }
        except Exception as e:
            verbose_proxy_logger.exception(f"Error deleting search tool from DB: {str(e)}")
            raise Exception(f"Error deleting search tool from DB: {str(e)}")

    async def update_search_tool_in_db(
        self, search_tool_id: str, search_tool: SearchTool, prisma_client: PrismaClient
    ):
        """
        Update a search tool in the database.
        
        Args:
            search_tool_id: ID of search tool to update
            search_tool: Updated search tool configuration
            prisma_client: Prisma client instance
            
        Returns:
            Dict with updated search tool data
        """
        try:
            search_tool_name = search_tool.get("search_tool_name")
            litellm_params: str = safe_dumps(dict(search_tool.get("litellm_params", {})))
            search_tool_info: str = safe_dumps(search_tool.get("search_tool_info", {}))

            # Update in DB
            updated_search_tool = await prisma_client.db.litellm_searchtoolstable.update(
                where={"search_tool_id": search_tool_id},
                data={
                    "search_tool_name": search_tool_name,
                    "litellm_params": litellm_params,
                    "search_tool_info": search_tool_info,
                    "updated_at": datetime.now(timezone.utc),
                },
            )

            # Convert to dict with ISO formatted datetimes
            return self._convert_prisma_to_dict(updated_search_tool)
        except Exception as e:
            verbose_proxy_logger.exception(f"Error updating search tool in DB: {str(e)}")
            raise Exception(f"Error updating search tool in DB: {str(e)}")

    @staticmethod
    async def get_all_search_tools_from_db(
        prisma_client: PrismaClient,
    ) -> List[SearchTool]:
        """
        Get all search tools from the database.
        
        Args:
            prisma_client: Prisma client instance
            
        Returns:
            List of search tool configurations
        """
        try:
            search_tools_from_db = (
                await prisma_client.db.litellm_searchtoolstable.find_many(
                    order={"created_at": "desc"},
                )
            )

            search_tools: List[SearchTool] = []
            for search_tool in search_tools_from_db:
                # Convert Prisma result to dict with ISO formatted datetimes
                search_tool_dict = SearchToolRegistry._convert_prisma_to_dict(search_tool)
                search_tools.append(SearchTool(**search_tool_dict))  # type: ignore

            return search_tools
        except Exception as e:
            verbose_proxy_logger.exception(f"Error getting search tools from DB: {str(e)}")
            raise Exception(f"Error getting search tools from DB: {str(e)}")

    async def get_search_tool_by_id_from_db(
        self, search_tool_id: str, prisma_client: PrismaClient
    ) -> Optional[SearchTool]:
        """
        Get a search tool by its ID from the database.
        
        Args:
            search_tool_id: ID of search tool to retrieve
            prisma_client: Prisma client instance
            
        Returns:
            Search tool configuration or None if not found
        """
        try:
            search_tool = await prisma_client.db.litellm_searchtoolstable.find_unique(
                where={"search_tool_id": search_tool_id}
            )

            if not search_tool:
                return None

            # Convert Prisma result to dict with ISO formatted datetimes
            search_tool_dict = self._convert_prisma_to_dict(search_tool)
            return SearchTool(**search_tool_dict)  # type: ignore
        except Exception as e:
            verbose_proxy_logger.exception(f"Error getting search tool from DB: {str(e)}")
            raise Exception(f"Error getting search tool from DB: {str(e)}")

    async def get_search_tool_by_name_from_db(
        self, search_tool_name: str, prisma_client: PrismaClient
    ) -> Optional[SearchTool]:
        """
        Get a search tool by its name from the database.
        
        Args:
            search_tool_name: Name of search tool to retrieve
            prisma_client: Prisma client instance
            
        Returns:
            Search tool configuration or None if not found
        """
        try:
            search_tool = await prisma_client.db.litellm_searchtoolstable.find_unique(
                where={"search_tool_name": search_tool_name}
            )

            if not search_tool:
                return None

            # Convert Prisma result to dict with ISO formatted datetimes
            search_tool_dict = self._convert_prisma_to_dict(search_tool)
            return SearchTool(**search_tool_dict)  # type: ignore
        except Exception as e:
            verbose_proxy_logger.exception(f"Error getting search tool from DB: {str(e)}")
            raise Exception(f"Error getting search tool from DB: {str(e)}")

