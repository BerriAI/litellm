"""Container management functions for LiteLLM."""

# Auto-generated container file functions from endpoints.json
from .endpoint_factory import (
    adelete_container_file,
    alist_container_files,
    aretrieve_container_file,
    aretrieve_container_file_content,
    delete_container_file,
    list_container_files,
    retrieve_container_file,
    retrieve_container_file_content,
)
from .main import (
    acreate_container,
    adelete_container,
    alist_containers,
    aretrieve_container,
    create_container,
    delete_container,
    list_containers,
    retrieve_container,
)

__all__ = [
    # Core container operations
    "acreate_container",
    "adelete_container",
    "alist_containers",
    "aretrieve_container",
    "create_container",
    "delete_container",
    "list_containers",
    "retrieve_container",
    # Container file operations (auto-generated from endpoints.json)
    "adelete_container_file",
    "alist_container_files",
    "aretrieve_container_file",
    "aretrieve_container_file_content",
    "delete_container_file",
    "list_container_files",
    "retrieve_container_file",
    "retrieve_container_file_content",
]

