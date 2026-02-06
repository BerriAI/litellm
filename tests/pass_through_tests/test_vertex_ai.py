"""
Test Vertex AI Pass Through

1. use Credentials client side, Assert SpendLog was created
"""

import vertexai
from vertexai.preview.generative_models import GenerativeModel
import tempfile
import json
import os
import pytest
import asyncio


# Path to your service account JSON file
SERVICE_ACCOUNT_FILE = "path/to/your/service-account.json"


def load_vertex_ai_credentials():
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/vertex_key.json"

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    # print(f"service_account_key_data: {service_account_key_data}")
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary files
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)


async def call_spend_logs_endpoint():
    """
    Call this
    curl -X GET "http://0.0.0.0:4000/spend/logs" -H "Authorization: Bearer sk-1234"
    """
    import datetime
    import requests

    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
    url = f"http://0.0.0.0:4000/global/spend/logs?api_key=best-api-key-ever"
    headers = {"Authorization": f"Bearer sk-1234"}
    response = requests.get(url, headers=headers)
    print("response from call_spend_logs_endpoint", response)

    json_response = response.json()

    # get spend for today
    """
    json response looks like this

    [{'date': '2024-08-30', 'spend': 0.00016600000000000002, 'api_key': 'best-api-key-ever'}]
    """
    print("json_response", json_response)

    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
    for spend_log in json_response:
        if spend_log["date"] == todays_date:
            return spend_log["spend"]


LITE_LLM_ENDPOINT = "http://localhost:4000"


@pytest.mark.asyncio()
async def test_basic_vertex_ai_pass_through_with_spendlog():

    spend_before = await call_spend_logs_endpoint() or 0.0
    load_vertex_ai_credentials()

    vertexai.init(
        project="pathrise-convert-1606954137718",
        location="us-central1",
        api_endpoint=f"{LITE_LLM_ENDPOINT}/vertex_ai",
        api_transport="rest",
    )

    model = GenerativeModel(model_name="gemini-2.5-flash-lite")
    response = model.generate_content("hi")

    print("response", response)

    await asyncio.sleep(20)
    spend_after = await call_spend_logs_endpoint()
    print("spend_after", spend_after)
    assert (
        spend_after > spend_before
    ), "Spend should be greater than before. spend_before: {}, spend_after: {}".format(
        spend_before, spend_after
    )

    pass


@pytest.mark.asyncio()
@pytest.mark.skip(reason="skip flaky test - vertex pass through streaming is flaky")
async def test_basic_vertex_ai_pass_through_streaming_with_spendlog():

    spend_before = await call_spend_logs_endpoint() or 0.0
    print("spend_before", spend_before)
    load_vertex_ai_credentials()

    vertexai.init(
        project="pathrise-convert-1606954137718",
        location="us-central1",
        api_endpoint=f"{LITE_LLM_ENDPOINT}/vertex_ai",
        api_transport="rest",
    )

    model = GenerativeModel(model_name="gemini-2.5-flash-lite")
    response = model.generate_content("hi", stream=True)

    for chunk in response:
        print("chunk", chunk)

    print("response", response)

    await asyncio.sleep(20)
    spend_after = await call_spend_logs_endpoint()
    print("spend_after", spend_after)
    assert (
        spend_after > spend_before
    ), "Spend should be greater than before. spend_before: {}, spend_after: {}".format(
        spend_before, spend_after
    )

    pass


@pytest.mark.skip(
    reason="skip flaky test - google context caching is flaky and not reliable."
)
@pytest.mark.asyncio
async def test_vertex_ai_pass_through_endpoint_context_caching():
    import vertexai
    from vertexai.generative_models import Part
    from vertexai.preview import caching
    import datetime

    # load_vertex_ai_credentials()

    vertexai.init(
        project="pathrise-convert-1606954137718",
        location="us-central1",
        api_endpoint=f"{LITE_LLM_ENDPOINT}/vertex_ai",
        api_transport="rest",
    )

    system_instruction = """
    You are an expert researcher. You always stick to the facts in the sources provided, and never make up new facts.
    Now look at these research papers, and answer the following questions.
    """

    contents = [
        Part.from_uri(
            "gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf",
            mime_type="application/pdf",
        ),
        Part.from_uri(
            "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
            mime_type="application/pdf",
        ),
    ]

    cached_content = caching.CachedContent.create(
        model_name="gemini-2.5-flash-lite-001",
        system_instruction=system_instruction,
        contents=contents,
        ttl=datetime.timedelta(minutes=60),
        # display_name="example-cache",
    )

    print(cached_content.name)
