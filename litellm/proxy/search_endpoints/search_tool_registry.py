"""
Search Tool Registry for managing search tool configurations.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional

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

            # Convert to dict and return
            return dict(updated_search_tool)
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
                search_tools.append(SearchTool(**(dict(search_tool))))  # type: ignore

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

            return SearchTool(**(dict(search_tool)))  # type: ignore
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

            return SearchTool(**(dict(search_tool)))  # type: ignore
        except Exception as e:
            verbose_proxy_logger.exception(f"Error getting search tool from DB: {str(e)}")
            raise Exception(f"Error getting search tool from DB: {str(e)}")


class InMemorySearchToolHandler:
    """
    Class that handles caching search tools in memory.
    """

    def __init__(self):
        self.IN_MEMORY_SEARCH_TOOLS: Dict[str, SearchTool] = {}
        """
        Search tool id to SearchTool object mapping
        """

    def add_search_tool(self, search_tool: SearchTool) -> None:
        """
        Add a search tool to in-memory cache.
        
        Args:
            search_tool: Search tool configuration
        """
        search_tool_id = search_tool.get("search_tool_id")
        if search_tool_id:
            self.IN_MEMORY_SEARCH_TOOLS[search_tool_id] = search_tool
            verbose_proxy_logger.debug(
                f"Added search tool '{search_tool.get('search_tool_name')}' to in-memory cache"
            )

    def update_search_tool(self, search_tool_id: str, search_tool: SearchTool) -> None:
        """
        Update a search tool in in-memory cache.
        
        Args:
            search_tool_id: ID of search tool to update
            search_tool: Updated search tool configuration
        """
        self.IN_MEMORY_SEARCH_TOOLS[search_tool_id] = search_tool
        verbose_proxy_logger.debug(
            f"Updated search tool '{search_tool.get('search_tool_name')}' in in-memory cache"
        )

    def delete_search_tool(self, search_tool_id: str) -> None:
        """
        Delete a search tool from in-memory cache.
        
        Args:
            search_tool_id: ID of search tool to delete
        """
        self.IN_MEMORY_SEARCH_TOOLS.pop(search_tool_id, None)
        verbose_proxy_logger.debug(
            f"Deleted search tool with ID '{search_tool_id}' from in-memory cache"
        )

    def list_search_tools(self) -> List[SearchTool]:
        """
        List all search tools in in-memory cache.
        
        Returns:
            List of search tool configurations
        """
        return list(self.IN_MEMORY_SEARCH_TOOLS.values())

    def get_search_tool_by_id(self, search_tool_id: str) -> Optional[SearchTool]:
        """
        Get a search tool by its ID from in-memory cache.
        
        Args:
            search_tool_id: ID of search tool to retrieve
            
        Returns:
            Search tool configuration or None if not found
        """
        return self.IN_MEMORY_SEARCH_TOOLS.get(search_tool_id)


########################################################
# In Memory Search Tool Handler for LiteLLM Proxy
########################################################
IN_MEMORY_SEARCH_TOOL_HANDLER = InMemorySearchToolHandler()
########################################################

