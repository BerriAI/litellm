"""
Search Tool Registry for managing search tool configurations.
"""

import os
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.auth.master_key_boot_check import SALT_KEY_ENV_VAR
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    _get_salt_key,
    decrypt_if_encrypted_with,
    encrypt_value_helper,
)
from litellm.proxy.db.exception_handler import call_with_db_reconnect_retry
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import SearchToolsRepository
from litellm.types.search import SearchTool
from litellm.types.utils import SearchProviders


class SearchToolRecord(Protocol):
    search_tool_id: str
    search_tool_name: str
    created_at: datetime
    updated_at: datetime

    def __iter__(self) -> Iterator[tuple[str, object]]: ...


class SearchToolTableClient(Protocol):
    async def create(self, data: Mapping[str, object]) -> SearchToolRecord: ...

    async def find_unique(self, where: Mapping[str, object]) -> SearchToolRecord | None: ...

    async def find_many(self, order: Mapping[str, str] | None = None) -> Sequence[SearchToolRecord]: ...

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> SearchToolRecord: ...

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, object]) -> int: ...

    async def delete(self, where: Mapping[str, object]) -> SearchToolRecord: ...


class _SearchToolsRepositoryView(Protocol):
    @property
    def table(self) -> SearchToolTableClient: ...


def _search_tools_table_of(repository: _SearchToolsRepositoryView) -> SearchToolTableClient:
    return repository.table


def _search_tools_table(prisma_client: PrismaClient) -> SearchToolTableClient:
    return _search_tools_table_of(SearchToolsRepository(prisma_client))


_STORED_LITELLM_PARAMS: Final = TypeAdapter(Mapping[str, object])


def _stored_litellm_params(row: SearchToolRecord) -> Mapping[str, object] | None:
    try:
        return _STORED_LITELLM_PARAMS.validate_python(dict(row).get("litellm_params"))
    except ValidationError:
        return None


def _encrypted_search_tool_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return encrypt_value_helper(value=value)
    except Exception:  # noqa: BLE001  # no salt key or master key configured: store the value as written
        return value


def encrypt_search_tool_litellm_params(litellm_params: Mapping[str, object]) -> Mapping[str, object]:
    """Encrypt every string value of a search tool's litellm_params for storage."""
    return {key: _encrypted_search_tool_value(value) for key, value in litellm_params.items()}


def _search_tool_plaintext(value: str) -> str | None:
    signing_key: Final = _get_salt_key()
    return None if signing_key is None else decrypt_if_encrypted_with(value, signing_key)


def _decrypted_search_tool_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    plaintext: Final = _search_tool_plaintext(value)
    return value if plaintext is None else plaintext


def decrypt_search_tool_litellm_params(litellm_params: Mapping[str, object]) -> Mapping[str, object]:
    """Decrypt stored litellm_params values; values that are not ciphertext are returned unchanged."""
    return {key: _decrypted_search_tool_value(value) for key, value in litellm_params.items()}


def _reencrypt_search_tool_value(value: object, encryption_key: str) -> object:
    if not isinstance(value, str):
        return value
    plaintext: Final = _search_tool_plaintext(value)
    return value if plaintext is None else encrypt_value_helper(value=plaintext, new_encryption_key=encryption_key)


async def _rotate_search_tool_row(
    table: SearchToolTableClient, search_tool_id: str, stored_litellm_params: Mapping[str, object], encryption_key: str
) -> None:
    expected_litellm_params: Mapping[str, object] | None = stored_litellm_params
    while expected_litellm_params is not None:
        rows_updated = await table.update_many(
            where={
                "search_tool_id": search_tool_id,
                "litellm_params": {"equals": safe_dumps(expected_litellm_params)},
            },
            data={
                "litellm_params": safe_dumps(
                    {
                        key: _reencrypt_search_tool_value(value, encryption_key)
                        for key, value in expected_litellm_params.items()
                    }
                )
            },
        )
        if rows_updated:
            return
        reread = await table.find_unique(where={"search_tool_id": search_tool_id})
        reread_litellm_params = None if reread is None else _stored_litellm_params(reread)
        if reread_litellm_params == expected_litellm_params:
            verbose_proxy_logger.warning(
                "Search tool %s was not re-encrypted: its stored litellm_params did not match on write", search_tool_id
            )
            return
        expected_litellm_params = reread_litellm_params


async def rotate_search_tools_master_key(prisma_client: PrismaClient, new_master_key: str) -> None:
    """Re-encrypt the litellm_params values that decrypt under the current key with the key in force after
    rotation (LITELLM_SALT_KEY when set, otherwise new_master_key).

    Values that do not decrypt under the current key (plaintext rows written before encryption, or
    ciphertext under another key) are kept as stored. Each row is written only if it still holds the
    litellm_params that were read, and is re-read and rotated again while it keeps being edited in between.
    """
    salt_key: Final = os.environ.get(SALT_KEY_ENV_VAR)
    encryption_key: Final = new_master_key if salt_key is None else salt_key
    table: Final = _search_tools_table(prisma_client)
    for row in await table.find_many():
        stored_litellm_params = _stored_litellm_params(row)
        if stored_litellm_params is not None:
            await _rotate_search_tool_row(table, row.search_tool_id, stored_litellm_params, encryption_key)


_KNOWN_SEARCH_PROVIDERS: Final = frozenset(provider.value for provider in SearchProviders)
# An empty string encrypted with aes-256-gcm, the shortest ciphertext either algorithm produces
_SHORTEST_CIPHERTEXT_LENGTH: Final = 47


def _did_not_decrypt(search_tool: Mapping[str, object]) -> bool:
    litellm_params: Final = search_tool.get("litellm_params")
    search_provider: Final = litellm_params.get("search_provider") if isinstance(litellm_params, Mapping) else None
    return (
        isinstance(search_provider, str)
        and search_provider not in _KNOWN_SEARCH_PROVIDERS
        and len(search_provider) >= _SHORTEST_CIPHERTEXT_LENGTH
    )


def keep_loaded_search_tools_that_do_not_decrypt(
    db_search_tools: Sequence[Mapping[str, object]], loaded_search_tools: Sequence[Mapping[str, object]]
) -> Sequence[Mapping[str, object]]:
    """Replace each DB search tool whose params do not decrypt with the current key by its loaded version."""
    loaded_by_id: Final = {tool.get("search_tool_id"): tool for tool in loaded_search_tools}
    kept: Final = tuple(
        loaded_by_id.get(tool.get("search_tool_id"), tool) if _did_not_decrypt(tool) else tool
        for tool in db_search_tools
    )
    for db_tool, kept_tool in zip(db_search_tools, kept):
        if kept_tool is not db_tool:
            verbose_proxy_logger.warning(
                "Search tool %s has litellm_params that do not decrypt with the current key; keeping the loaded "
                "version. Restart the proxy if the master key was rotated.",
                db_tool.get("search_tool_id"),
            )
    return kept


class SearchToolRegistry:
    """
    Handles adding, removing, and getting search tools in DB + in memory.
    """

    def __init__(self):
        pass

    @staticmethod
    def _convert_prisma_to_dict(prisma_obj: SearchToolRecord) -> dict:
        """
        Convert Prisma result to dict with decrypted litellm_params and datetime objects as ISO format strings.

        Args:
            prisma_obj: Prisma model instance

        Returns:
            Dict with datetime fields converted to ISO strings
        """
        stored_litellm_params: Final = _stored_litellm_params(prisma_obj)
        result: Final = {
            **dict(prisma_obj),
            **(
                {"litellm_params": decrypt_search_tool_litellm_params(stored_litellm_params)}
                if stored_litellm_params is not None
                else {}
            ),
        }
        # Convert datetime objects to ISO format strings
        if "created_at" in result and result["created_at"]:
            result["created_at"] = prisma_obj.created_at.isoformat()
        if "updated_at" in result and result["updated_at"]:
            result["updated_at"] = prisma_obj.updated_at.isoformat()
        return result

    ###########################################################
    ########### DB management helpers for search tools ########
    ###########################################################

    async def add_search_tool_to_db(self, search_tool: SearchTool, prisma_client: PrismaClient):
        """
        Add a search tool to the database.

        Args:
            search_tool: Search tool configuration
            prisma_client: Prisma client instance

        Returns:
            Dict with created search tool data
        """
        try:
            search_tool_name: Final = search_tool.get("search_tool_name")
            litellm_params: Final[str] = safe_dumps(
                encrypt_search_tool_litellm_params(search_tool.get("litellm_params", {}))
            )
            search_tool_info: Final[str] = safe_dumps(search_tool.get("search_tool_info", {}))

            # Create search tool in DB
            created_search_tool: Final = await _search_tools_table(prisma_client).create(
                data={
                    "search_tool_name": search_tool_name,
                    "litellm_params": litellm_params,
                    "search_tool_info": search_tool_info,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

            # Add search_tool_id to the returned search tool object
            search_tool_dict: Final = dict(search_tool)
            search_tool_dict["search_tool_id"] = created_search_tool.search_tool_id
            search_tool_dict["created_at"] = created_search_tool.created_at.isoformat()
            search_tool_dict["updated_at"] = created_search_tool.updated_at.isoformat()

            return search_tool_dict
        except Exception as e:
            verbose_proxy_logger.exception("Error adding search tool to DB: %s", e)
            raise Exception(f"Error adding search tool to DB: {e}")

    async def delete_search_tool_from_db(self, search_tool_id: str, prisma_client: PrismaClient):
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
            existing_tool: Final = await _search_tools_table(prisma_client).find_unique(
                where={"search_tool_id": search_tool_id}
            )

            if not existing_tool:
                raise Exception(f"Search tool with ID {search_tool_id} not found")

            # Delete from DB
            await _search_tools_table(prisma_client).delete(where={"search_tool_id": search_tool_id})

            return {
                "message": f"Search tool {search_tool_id} deleted successfully",
                "search_tool_name": existing_tool.search_tool_name,
            }
        except Exception as e:
            verbose_proxy_logger.exception("Error deleting search tool from DB: %s", e)
            raise Exception(f"Error deleting search tool from DB: {e}")

    async def update_search_tool_in_db(self, search_tool_id: str, search_tool: SearchTool, prisma_client: PrismaClient):
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
            search_tool_name: Final = search_tool.get("search_tool_name")
            litellm_params: Final[str] = safe_dumps(
                encrypt_search_tool_litellm_params(search_tool.get("litellm_params", {}))
            )
            search_tool_info: Final[str] = safe_dumps(search_tool.get("search_tool_info", {}))

            # Update in DB
            updated_search_tool: Final = await _search_tools_table(prisma_client).update(
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
            verbose_proxy_logger.exception("Error updating search tool in DB: %s", e)
            raise Exception(f"Error updating search tool in DB: {e}")

    @staticmethod
    async def get_all_search_tools_from_db(
        prisma_client: PrismaClient,
    ) -> list[SearchTool]:
        """
        Get all search tools from the database.

        Args:
            prisma_client: Prisma client instance

        Returns:
            List of search tool configurations
        """
        try:
            search_tools_from_db: Final = await call_with_db_reconnect_retry(
                prisma_client,
                lambda: _search_tools_table(prisma_client).find_many(
                    order={"created_at": "desc"},
                ),
                reason="get_all_search_tools_from_db_lookup_failure",
            )

            search_tools: Final[list[SearchTool]] = []
            for search_tool in search_tools_from_db:
                # Convert Prisma result to dict with ISO formatted datetimes
                search_tool_dict = SearchToolRegistry._convert_prisma_to_dict(search_tool)
                search_tools.append(SearchTool(**search_tool_dict))

            return search_tools
        except Exception as e:
            verbose_proxy_logger.exception("Error getting search tools from DB: %s", e)
            raise Exception(f"Error getting search tools from DB: {e}")

    async def get_search_tool_by_id_from_db(
        self, search_tool_id: str, prisma_client: PrismaClient
    ) -> SearchTool | None:
        """
        Get a search tool by its ID from the database.

        Args:
            search_tool_id: ID of search tool to retrieve
            prisma_client: Prisma client instance

        Returns:
            Search tool configuration or None if not found
        """
        try:
            search_tool: Final = await _search_tools_table(prisma_client).find_unique(
                where={"search_tool_id": search_tool_id}
            )

            if not search_tool:
                return None

            # Convert Prisma result to dict with ISO formatted datetimes
            search_tool_dict: Final = self._convert_prisma_to_dict(search_tool)
            return SearchTool(**search_tool_dict)
        except Exception as e:
            verbose_proxy_logger.exception("Error getting search tool from DB: %s", e)
            raise Exception(f"Error getting search tool from DB: {e}")

    async def get_search_tool_by_name_from_db(
        self, search_tool_name: str, prisma_client: PrismaClient
    ) -> SearchTool | None:
        """
        Get a search tool by its name from the database.

        Args:
            search_tool_name: Name of search tool to retrieve
            prisma_client: Prisma client instance

        Returns:
            Search tool configuration or None if not found
        """
        try:
            search_tool: Final = await _search_tools_table(prisma_client).find_unique(
                where={"search_tool_name": search_tool_name}
            )

            if not search_tool:
                return None

            # Convert Prisma result to dict with ISO formatted datetimes
            search_tool_dict: Final = self._convert_prisma_to_dict(search_tool)
            return SearchTool(**search_tool_dict)
        except Exception as e:
            verbose_proxy_logger.exception("Error getting search tool from DB: %s", e)
            raise Exception(f"Error getting search tool from DB: {e}")
