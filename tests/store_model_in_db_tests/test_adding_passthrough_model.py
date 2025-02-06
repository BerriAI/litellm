"""
Test adding a pass through assemblyai model + api key + api base to the db
wait 20 seconds 
make request 

Cases to cover 
1. user points api base to <proxy-base>/assemblyai 
2. user points api base to <proxy-base>/asssemblyai/us
3. user points api base to <proxy-base>/assemblyai/eu 
4. Bad API Key / credential - 401
"""

import time
import assemblyai as aai
import pytest
import httpx
import os

TEST_MASTER_KEY = "sk-1234"
PROXY_BASE_URL = "http://0.0.0.0:4000"
US_BASE_URL = f"{PROXY_BASE_URL}/assemblyai"
EU_BASE_URL = f"{PROXY_BASE_URL}/eu.assemblyai"
ASSEMBLYAI_API_KEY_ENV_VAR = "TEST_SPECIAL_ASSEMBLYAI_API_KEY"


def test_e2e_assemblyai_passthrough():
    """
    Test adding a pass through assemblyai model + api key + api base to the db
    wait 20 seconds
    make request
    """
    add_assembly_ai_model_to_db(api_base="https://api.assemblyai.com")
    virtual_key = create_virtual_key()
    # make request
    make_assemblyai_basic_transcribe_request(
        virtual_key=virtual_key, assemblyai_base_url=US_BASE_URL
    )

    pass


def test_e2e_assemblyai_passthrough_eu():
    """
    Test adding a pass through assemblyai model + api key + api base to the db
    wait 20 seconds
    make request
    """
    add_assembly_ai_model_to_db(api_base="https://api.eu.assemblyai.com")
    virtual_key = create_virtual_key()
    # make request
    make_assemblyai_basic_transcribe_request(
        virtual_key=virtual_key, assemblyai_base_url=EU_BASE_URL
    )

    pass


def create_virtual_key():
    """
    Create a virtual key
    """
    response = httpx.post(
        url=f"{PROXY_BASE_URL}/key/generate",
        headers={"Authorization": f"Bearer {TEST_MASTER_KEY}"},
        json={},
    )
    print(response.json())
    return response.json()["token"]


def add_assembly_ai_model_to_db(
    api_base: str,
):
    """
    Add the assemblyai model to the db - makes a http request to the /model/new endpoint on PROXY_BASE_URL
    """
    response = httpx.post(
        url=f"{PROXY_BASE_URL}/model/new",
        headers={"Authorization": f"Bearer {TEST_MASTER_KEY}"},
        json={
            "model_name": "assemblyai/*",
            "litellm_params": {
                "model": "assemblyai/*",
                "custom_llm_provider": "assemblyai",
                "api_key": os.getenv(ASSEMBLYAI_API_KEY_ENV_VAR),
                "api_base": api_base,
                "use_in_pass_through": True,
            },
            "model_info": {},
        },
    )
    print(response.json())
    pass


def make_assemblyai_basic_transcribe_request(
    virtual_key: str, assemblyai_base_url: str
):
    print("making basic transcribe request to assemblyai passthrough")

    # Replace with your API key
    aai.settings.api_key = f"Bearer {virtual_key}"
    aai.settings.base_url = assemblyai_base_url

    # URL of the file to transcribe
    FILE_URL = "https://assembly.ai/wildfires.mp3"

    # You can also transcribe a local file by passing in a file path
    # FILE_URL = './path/to/file.mp3'

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(FILE_URL)
    print(transcript)
    print(transcript.id)
    if transcript.id:
        transcript.delete_by_id(transcript.id)
    else:
        pytest.fail("Failed to get transcript id")

    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
        pytest.fail(f"Failed to transcribe file error: {transcript.error}")
    else:
        print(transcript.text)
