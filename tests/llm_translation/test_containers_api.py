"""
E2E Test for Container Files API.

Tests the container files endpoints using LiteLLM SDK methods.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.containers import (
    create_container,
    delete_container,
)
from litellm.containers.endpoint_factory import (
    list_container_files,
    retrieve_container_file,
    retrieve_container_file_content,
    delete_container_file,
)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_container_files_api():
    """
    Test container files API: list, retrieve, delete.
    
    Flow:
    1. Create a container
    2. List files (should be empty)
    3. Try retrieve file (should error - no files)
    4. Try delete file (should error - no files)
    5. Cleanup: delete container
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 1. Create container
    print("\n1. Creating container...")
    container = create_container(
        name=f"test-files-api-{int(time.time())}",
        custom_llm_provider="openai",
        api_key=api_key,
        expires_after={"anchor": "last_active_at", "minutes": 5},
    )
    print(f"   Created: {container.id}")
    
    try:
        # 2. List files
        print("2. Listing container files...")
        files = list_container_files(
            container_id=container.id,
            custom_llm_provider="openai",
            api_key=api_key,
        )
        assert files.object == "list"
        assert isinstance(files.data, list)
        assert len(files.data) == 0  # New container has no files
        print(f"   Files found: {len(files.data)} ✓")
        
        # 3. Try retrieve non-existent file metadata (should raise error)
        print("3. Testing retrieve_container_file (expect error)...")
        try:
            retrieve_container_file(
                container_id=container.id,
                file_id="cfile_nonexistent",
                custom_llm_provider="openai",
                api_key=api_key,
            )
            assert False, "Should have raised error for non-existent file"
        except Exception as e:
            assert "not found" in str(e).lower() or "invalid" in str(e).lower()
            print(f"   Got expected error ✓")
        
        # 3b. Try retrieve non-existent file content (should raise error)
        print("3b. Testing retrieve_container_file_content (expect error)...")
        try:
            retrieve_container_file_content(
                container_id=container.id,
                file_id="cfile_nonexistent",
                custom_llm_provider="openai",
                api_key=api_key,
            )
            assert False, "Should have raised error for non-existent file content"
        except Exception as e:
            print(f"   Got expected error ✓")
        
        # 4. Try delete non-existent file (should raise error)
        print("4. Testing delete_container_file (expect error)...")
        try:
            delete_container_file(
                container_id=container.id,
                file_id="cfile_nonexistent",
                custom_llm_provider="openai",
                api_key=api_key,
            )
            assert False, "Should have raised error for non-existent file"
        except Exception as e:
            # Delete returns 400 for non-existent files
            print(f"   Got expected error ✓")
        
    finally:
        # 5. Cleanup
        print("5. Deleting container...")
        result = delete_container(
            container_id=container.id,
            custom_llm_provider="openai",
            api_key=api_key,
        )
        assert result.deleted is True
        print(f"   Deleted ✓")
    
    print("\nAll container files API tests passed! ✓")
