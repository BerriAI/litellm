import requests
import pytest
import os
import sys
import pytest


sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system paths

import litellm
from litellm import completion
from vertex_ai_test_utils import get_vertex_ai_creds_json, load_vertex_ai_credentials





# Download and save the PDF locally
url = (
    "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
)
response = requests.get(url)
response.raise_for_status()

# Save the PDF locally
with open("2403.05530.pdf", "wb") as f:
    f.write(response.content)

@pytest.mark.asyncio()
async def test_async_file_upload_with_chat_completion_vertex_ai():
    """
    Test File creation and chat completion with a file PDF

    1. Create File for GCS
    2. Create Chat Completion with a file PDF
    """
    load_vertex_ai_credentials()
    #litellm._turn_on_debug()
    ###############################
    # Create File
    ###############################
    file_obj = await litellm.acreate_file(
        file=open("2403.05530.pdf", "rb"),
        purpose="user_data",

        ###############################
        # Vertex Specific Parameters
        ###############################
        custom_llm_provider="vertex_ai",
        gcs_bucket_name="litellm-local",
    )
    print("CREATED FILE RESPONSE=", file_obj)

    #########################################################
    # create chat completion with a file PDF
    #########################################################
    response = await litellm.acompletion(
        model="vertex_ai/gemini-1.5-flash",
        max_tokens=10,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "What is in this file?"}]},
            {"role": "user", "content": [{"type": "file", "file": {"file_id": file_obj.id}}]},
        ],
    )
    print("RESPONSE=", response)
