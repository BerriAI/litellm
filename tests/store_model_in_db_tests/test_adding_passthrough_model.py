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
import pytest
import httpx
import os
import json

TEST_MASTER_KEY = "sk-1234"
PROXY_BASE_URL = "http://0.0.0.0:4000"
US_BASE_URL = f"{PROXY_BASE_URL}/assemblyai"
EU_BASE_URL = f"{PROXY_BASE_URL}/eu.assemblyai"
ASSEMBLYAI_API_KEY_ENV_VAR = "ASSEMBLYAI_API_KEY"


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
    file_url = "https://assembly.ai/wildfires.mp3"
    headers = {
        "Authorization": f"Bearer {virtual_key}",
        "Content-Type": "application/json",
    }
    create_payload = {
        "audio_url": file_url,
        "speech_models": ["universal-2"],
    }

    create_response = httpx.post(
        url=f"{assemblyai_base_url}/v2/transcript",
        headers=headers,
        json=create_payload,
        timeout=60.0,
    )
    if create_response.status_code != 200:
        pytest.fail(
            "Failed to create transcript request: "
            f"status={create_response.status_code}, body={create_response.text}"
        )

    transcript = create_response.json()
    transcript_id = transcript.get("id")
    if not transcript_id:
        pytest.fail("Failed to get transcript id")

    for _ in range(60):
        poll_response = httpx.get(
            url=f"{assemblyai_base_url}/v2/transcript/{transcript_id}",
            headers=headers,
            timeout=30.0,
        )
        if poll_response.status_code != 200:
            pytest.fail(
                "Failed to poll transcript status: "
                f"status={poll_response.status_code}, body={poll_response.text}"
            )
        transcript = poll_response.json()
        if transcript.get("status") in ("completed", "error"):
            break
        time.sleep(1)

    httpx.delete(
        url=f"{assemblyai_base_url}/v2/transcript/{transcript_id}",
        headers=headers,
        timeout=30.0,
    )

    if transcript.get("status") == "error":
        pytest.fail(f"Failed to transcribe file error: {transcript.get('error')}")

    print(transcript.get("text"))
