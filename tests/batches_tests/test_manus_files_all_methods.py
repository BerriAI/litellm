"""
E2E test for all Manus Files API methods.
"""

import os
import pytest
import litellm


@pytest.mark.asyncio
async def test_manus_files_api_e2e_all_methods():
    """
    E2E test for Manus Files API: create, retrieve, list, delete.
    """
    litellm._turn_on_debug()

    api_key = os.getenv("MANUS_API_KEY")
    if api_key is None:
        pytest.skip("MANUS_API_KEY not set")

    # Create a simple test file content
    test_content = b"This is a test file for Manus Files API - all methods test."
    test_filename = "test_file_all_methods.txt"

    # Step 1: Create file
    print("Step 1: Creating file...")
    created_file = await litellm.acreate_file(
        file=(test_filename, test_content),
        purpose="assistants",
        custom_llm_provider="manus",
        api_key=api_key,
    )
    print(f"Created file: {created_file}")
    assert created_file.filename == test_filename
    assert created_file.status == "uploaded"
    # Note: Manus doesn't return bytes in initial response
    file_id = created_file.id

    # Step 2: Retrieve file
    print(f"\nStep 2: Retrieving file {file_id}...")
    retrieved_file = await litellm.afile_retrieve(
        file_id=file_id,
        custom_llm_provider="manus",
        api_key=api_key,
    )
    print(f"Retrieved file: {retrieved_file}")
    assert retrieved_file.id == file_id
    assert retrieved_file.filename == test_filename

    # Step 3: List files
    print("\nStep 3: Listing files...")
    files_list = await litellm.afile_list(
        custom_llm_provider="manus",
        api_key=api_key,
    )
    print(f"Files list: {files_list}")
    assert isinstance(files_list, list)
    assert any(f.id == file_id for f in files_list)

    # Step 4: Delete file
    print(f"\nStep 4: Deleting file {file_id}...")
    deleted_file = await litellm.afile_delete(
        file_id=file_id,
        custom_llm_provider="manus",
        api_key=api_key,
    )
    print(f"Deleted file: {deleted_file}")
    assert deleted_file.id == file_id
    assert deleted_file.deleted is True

    print("\n✅ All Manus Files API methods working!")
