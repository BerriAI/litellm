import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock

from litellm.caching import DualCache
from litellm.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles
from litellm.types.utils import SpecialEnums


def test_get_file_ids_from_messages():
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=MagicMock()
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCxmYzdmMmVhNS0wZjUwLTQ5ZjYtODljMS03ZTZhNTRiMTIxMzg",
                    },
                },
            ],
        },
    ]
    file_ids = proxy_managed_files.get_file_ids_from_messages(messages)
    assert file_ids == [
        "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCxmYzdmMmVhNS0wZjUwLTQ5ZjYtODljMS03ZTZhNTRiMTIxMzg"
    ]


# def test_list_managed_files():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())

#     # Create some test files
#     file1 = proxy_managed_files.create_file(
#         file=("test1.txt", b"test content 1", "text/plain"),
#         purpose="assistants"
#     )
#     file2 = proxy_managed_files.create_file(
#         file=("test2.pdf", b"test content 2", "application/pdf"),
#         purpose="assistants"
#     )

#     # List all files
#     files = proxy_managed_files.list_files()

#     # Verify response
#     assert len(files) == 2
#     assert all(f.id.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value) for f in files)
#     assert any(f.filename == "test1.txt" for f in files)
#     assert any(f.filename == "test2.pdf" for f in files)
#     assert all(f.purpose == "assistants" for f in files)

# def test_retrieve_managed_file():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())

#     # Create a test file
#     test_content = b"test content for retrieve"
#     created_file = proxy_managed_files.create_file(
#         file=("test.txt", test_content, "text/plain"),
#         purpose="assistants"
#     )

#     # Retrieve the file
#     retrieved_file = proxy_managed_files.retrieve_file(created_file.id)

#     # Verify response
#     assert retrieved_file.id == created_file.id
#     assert retrieved_file.filename == "test.txt"
#     assert retrieved_file.purpose == "assistants"
#     assert retrieved_file.bytes == len(test_content)
#     assert retrieved_file.status == "uploaded"

# def test_delete_managed_file():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())

#     # Create a test file
#     created_file = proxy_managed_files.create_file(
#         file=("test.txt", b"test content", "text/plain"),
#         purpose="assistants"
#     )

#     # Delete the file
#     deleted_file = proxy_managed_files.delete_file(created_file.id)

#     # Verify deletion
#     assert deleted_file.id == created_file.id
#     assert deleted_file.deleted == True

#     # Verify file is no longer retrievable
#     with pytest.raises(Exception):
#         proxy_managed_files.retrieve_file(created_file.id)

#     # Verify file is not in list
#     files = proxy_managed_files.list_files()
#     assert created_file.id not in [f.id for f in files]

# def test_retrieve_nonexistent_file():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())

#     # Try to retrieve a non-existent file
#     with pytest.raises(Exception):
#         proxy_managed_files.retrieve_file("nonexistent-file-id")

# def test_delete_nonexistent_file():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())

#     # Try to delete a non-existent file
#     with pytest.raises(Exception):
#         proxy_managed_files.delete_file("nonexistent-file-id")

# def test_list_files_with_purpose_filter():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())

#     # Create files with different purposes
#     file1 = proxy_managed_files.create_file(
#         file=("test1.txt", b"test content 1", "text/plain"),
#         purpose="assistants"
#     )
#     file2 = proxy_managed_files.create_file(
#         file=("test2.pdf", b"test content 2", "application/pdf"),
#         purpose="batch"
#     )

#     # List files with purpose filter
#     assistant_files = proxy_managed_files.list_files(purpose="assistants")
#     batch_files = proxy_managed_files.list_files(purpose="batch")

#     # Verify filtering
#     assert len(assistant_files) == 1
#     assert len(batch_files) == 1
#     assert assistant_files[0].id == file1.id
#     assert batch_files[0].id == file2.id
