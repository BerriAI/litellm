"""
Fetch Tool Registry for managing fetch tool configurations.
"""

from datetime import datetime, timezone
from typing import List

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.utils import PrismaClient
from litellm.types.fetch import FetchTool


class FetchToolRegistry:
    """
    Handles adding, removing, and getting fetch tools in DB + in memory.
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
    ########### DB management helpers for fetch tools ########
    ###########################################################

    async def add_fetch_tool_to_db(
        self, fetch_tool: FetchTool, prisma_client: PrismaClient
    ):
        """
        Add a fetch tool to the database.

        Args:
            fetch_tool: Fetch tool configuration
            prisma_client: Prisma client instance

        Returns:
            Dict with created fetch tool data
        """
        try:
            fetch_tool_name = fetch_tool.get("fetch_tool_name")
            litellm_params: str = safe_dumps(dict(fetch_tool.get("litellm_params", {})))
            fetch_tool_info: str = safe_dumps(fetch_tool.get("fetch_tool_info", {}))

            # Create fetch tool in DB
            created_fetch_tool = await prisma_client.db.litellm_fetchtoolstable.create(
                data={
                    "fetch_tool_name": fetch_tool_name,
                    "litellm_params": litellm_params,
                    "fetch_tool_info": fetch_tool_info,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

            # Add fetch_tool_id to the returned fetch tool object
            fetch_tool_dict = dict(fetch_tool)
            fetch_tool_dict["fetch_tool_id"] = created_fetch_tool.fetch_tool_id
            fetch_tool_dict["created_at"] = created_fetch_tool.created_at.isoformat()
            fetch_tool_dict["updated_at"] = created_fetch_tool.updated_at.isoformat()

            return fetch_tool_dict
        except Exception as e:
            verbose_proxy_logger.exception(f"Error adding fetch tool to DB: {str(e)}")
            raise Exception(f"Error adding fetch tool to DB: {str(e)}")

    async def delete_fetch_tool_from_db(
        self, fetch_tool_id: str, prisma_client: PrismaClient
    ):
        """
        Delete a fetch tool from the database.

        Args:
            fetch_tool_id: ID of fetch tool to delete
            prisma_client: Prisma client instance

        Returns:
            Dict with success message
        """
        try:
            await prisma_client.db.litellm_fetchtoolstable.delete(
                where={"fetch_tool_id": fetch_tool_id}
            )
            return {"message": f"Fetch tool {fetch_tool_id} deleted successfully"}
        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error deleting fetch tool from DB: {str(e)}"
            )
            raise Exception(f"Error deleting fetch tool from DB: {str(e)}")

    async def get_fetch_tool_from_db(
        self, fetch_tool_id: str, prisma_client: PrismaClient
    ):
        """
        Get a fetch tool from the database by ID.

        Args:
            fetch_tool_id: ID of fetch tool to retrieve
            prisma_client: Prisma client instance

        Returns:
            Dict with fetch tool data or None if not found
        """
        try:
            fetch_tool = await prisma_client.db.litellm_fetchtoolstable.find_unique(
                where={"fetch_tool_id": fetch_tool_id}
            )
            if fetch_tool is None:
                return None
            return self._convert_prisma_to_dict(fetch_tool)
        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error getting fetch tool from DB: {str(e)}"
            )
            raise Exception(f"Error getting fetch tool from DB: {str(e)}")

    async def get_all_fetch_tools_from_db(self, prisma_client: PrismaClient):
        """
        Get all fetch tools from the database.

        Args:
            prisma_client: Prisma client instance

        Returns:
            List of dicts with fetch tool data
        """
        try:
            fetch_tools_from_db = (
                await prisma_client.db.litellm_fetchtoolstable.find_many()
            )
            fetch_tools: List[FetchTool] = []
            for fetch_tool in fetch_tools_from_db:
                fetch_tool_dict = self._convert_prisma_to_dict(fetch_tool)
                fetch_tools.append(FetchTool(**fetch_tool_dict))  # type: ignore
            return fetch_tools
        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error getting all fetch tools from DB: {str(e)}"
            )
            raise Exception(f"Error getting all fetch tools from DB: {str(e)}")

    async def update_fetch_tool_in_db(
        self,
        fetch_tool_id: str,
        fetch_tool: FetchTool,
        prisma_client: PrismaClient,
    ):
        """
        Update a fetch tool in the database.

        Args:
            fetch_tool_id: ID of fetch tool to update
            fetch_tool: Updated fetch tool configuration
            prisma_client: Prisma client instance

        Returns:
            Dict with updated fetch tool data
        """
        try:
            fetch_tool_name = fetch_tool.get("fetch_tool_name")
            litellm_params: str = safe_dumps(dict(fetch_tool.get("litellm_params", {})))
            fetch_tool_info: str = safe_dumps(fetch_tool.get("fetch_tool_info", {}))

            updated_fetch_tool = await prisma_client.db.litellm_fetchtoolstable.update(
                where={"fetch_tool_id": fetch_tool_id},
                data={
                    "fetch_tool_name": fetch_tool_name,
                    "litellm_params": litellm_params,
                    "fetch_tool_info": fetch_tool_info,
                    "updated_at": datetime.now(timezone.utc),
                },
            )

            fetch_tool_dict = dict(fetch_tool)
            fetch_tool_dict["fetch_tool_id"] = updated_fetch_tool.fetch_tool_id
            fetch_tool_dict["created_at"] = updated_fetch_tool.created_at.isoformat()
            fetch_tool_dict["updated_at"] = updated_fetch_tool.updated_at.isoformat()

            return fetch_tool_dict
        except Exception as e:
            verbose_proxy_logger.exception(f"Error updating fetch tool in DB: {str(e)}")
            raise Exception(f"Error updating fetch tool in DB: {str(e)}")
