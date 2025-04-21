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
import json

TEST_MASTER_KEY = "sk-1234"
PROXY_BASE_URL = "http://0.0.0.0:4000"
US_BASE_URL = f"{PROXY_BASE_URL}/assemblyai"
EU_BASE_URL = f"{PROXY_BASE_URL}/eu.assemblyai"
ASSEMBLYAI_API_KEY_ENV_VAR = "TEST_SPECIAL_ASSEMBLYAI_API_KEY"


def _delete_all_assemblyai_models_from_db():
    """
    Delete all assemblyai models from the db
    """
    print("Deleting all assemblyai models from the db.......")
    model_list_response = httpx.get(
        url=f"{PROXY_BASE_URL}/v2/model/info",
        headers={"Authorization": f"Bearer {TEST_MASTER_KEY}"},
    )
    response_data = model_list_response.json()
    print("model list response", json.dumps(response_data, indent=4, default=str))
    # Filter for only AssemblyAI models
    assemblyai_models = [
        model
        for model in response_data["data"]
        if model.get("litellm_params", {}).get("custom_llm_provider") == "assemblyai"
    ]

    for model in assemblyai_models:
        model_id = model["model_info"]["id"]
        httpx.post(
            url=f"{PROXY_BASE_URL}/model/delete",
            headers={"Authorization": f"Bearer {TEST_MASTER_KEY}"},
            json={"id": model_id},
        )
    print("Deleted all assemblyai models from the db")


@pytest.fixture(autouse=True)
def cleanup_assemblyai_models():
    """
    Fixture to clean up AssemblyAI models before and after each test
    """
    # Clean up before test
    _delete_all_assemblyai_models_from_db()

    # Run the test
    yield

    # Clean up after test
    _delete_all_assemblyai_models_from_db()


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


def test_assemblyai_routes_with_bad_api_key():
    """
    Test AssemblyAI endpoints with invalid API key to ensure proper error handling
    """
    bad_api_key = "sk-12222"
    payload = {
        "audio_url": "https://assembly.ai/wildfires.mp3",
        "audio_end_at": 280,
        "audio_start_from": 10,
        "auto_chapters": True,
    }
    headers = {
        "Authorization": f"Bearer {bad_api_key}",
        "Content-Type": "application/json",
    }

    # Test EU endpoint
    eu_response = httpx.post(
        f"{PROXY_BASE_URL}/eu.assemblyai/v2/transcript", headers=headers, json=payload
    )
    assert (
        eu_response.status_code == 401
    ), f"Expected 401 unauthorized, got {eu_response.status_code}"

    # Test US endpoint
    us_response = httpx.post(
        f"{PROXY_BASE_URL}/assemblyai/v2/transcript", headers=headers, json=payload
    )
    assert (
        us_response.status_code == 401
    ), f"Expected 401 unauthorized, got {us_response.status_code}"


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
    return response.json()["key"]


def add_assembly_ai_model_to_db(
    api_base: str,
):
    """
    Add the assemblyai model to the db - makes a http request to the /model/new endpoint on PROXY_BASE_URL
    """
    print("assmbly ai api key", os.getenv(ASSEMBLYAI_API_KEY_ENV_VAR))
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
