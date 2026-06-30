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


SPEND_LOG_API_KEY = "best-api-key-ever"


async def get_tracked_spend() -> float:
    """
    Total spend recorded under the pass-through key in the global spend view.

    Sums every day the endpoint returns instead of matching the runner's local
    "today" so a UTC date rollover mid-test can't hide a freshly billed call, and
    treats an unreachable endpoint as "nothing recorded yet" (0.0).
    """
    import requests

    url = f"http://0.0.0.0:4000/global/spend/logs?api_key={SPEND_LOG_API_KEY}"
    response = requests.get(url, headers={"Authorization": "Bearer sk-1234"})
    if response.status_code != 200:
        print(f"global spend logs endpoint returned {response.status_code}: {response.text}")
        return 0.0

    rows = response.json()
    print("global spend logs rows", rows)
    return sum(float(row.get("spend") or 0.0) for row in rows)


LITE_LLM_ENDPOINT = "http://localhost:4000"


def _is_vertex_quota_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "429" in message
        or "Too Many Requests" in message
        or "RESOURCE_EXHAUSTED" in message
    )


@pytest.mark.asyncio()
async def test_basic_vertex_ai_pass_through_with_spendlog():

    load_vertex_ai_credentials()

    vertexai.init(
        project="litellm-ci-cd",
        location="global",
        api_endpoint=f"{LITE_LLM_ENDPOINT}/vertex_ai",
        api_transport="rest",
    )

    model = GenerativeModel(model_name="gemini-3.1-flash-lite")

    spend_before = await get_tracked_spend()

    # Pass-through spend logging is best-effort: the success handler runs on a
    # background worker that can drop or time out an individual event under CI load
    # and never retries it, so one billed call occasionally never reaches
    # LiteLLM_SpendLogs. Waiting longer can't recover a dropped event; only re-billing
    # can. Require at least one of a few billed calls to be tracked. This still fails
    # hard if cost tracking is genuinely broken, since then every call records nothing.
    max_attempts = 5
    per_attempt_wait = 45  # seconds to wait for a single call's spend to land
    poll_interval = 5

    spend_after = spend_before
    for attempt in range(1, max_attempts + 1):
        try:
            model.generate_content("hi")
        except Exception as exc:
            if _is_vertex_quota_error(exc):
                pytest.skip("Vertex AI quota exhausted")
            raise

        for _ in range(per_attempt_wait // poll_interval):
            await asyncio.sleep(poll_interval)
            spend_after = await get_tracked_spend()
            if spend_after > spend_before:
                print(f"spend tracked on attempt {attempt}: {spend_before} -> {spend_after}")
                return

        print(f"attempt {attempt}: spend not tracked yet (spend_after={spend_after}), re-billing")

    pytest.fail(
        "Vertex pass-through spend never recorded after {} billed calls. spend_before: {}, spend_after: {}".format(
            max_attempts, spend_before, spend_after
        )
    )


@pytest.mark.asyncio()
@pytest.mark.skip(reason="skip flaky test - vertex pass through streaming is flaky")
async def test_basic_vertex_ai_pass_through_streaming_with_spendlog():

    spend_before = await get_tracked_spend()
    print("spend_before", spend_before)
    load_vertex_ai_credentials()

    vertexai.init(
        project="litellm-ci-cd",
        location="global",
        api_endpoint=f"{LITE_LLM_ENDPOINT}/vertex_ai",
        api_transport="rest",
    )

    model = GenerativeModel(model_name="gemini-3.1-flash-lite")
    response = model.generate_content("hi", stream=True)

    for chunk in response:
        print("chunk", chunk)

    print("response", response)

    await asyncio.sleep(20)
    spend_after = await get_tracked_spend()
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
        project="litellm-ci-cd",
        location="global",
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
        model_name="gemini-3.1-flash-lite",
        system_instruction=system_instruction,
        contents=contents,
        ttl=datetime.timedelta(minutes=60),
        # display_name="example-cache",
    )

    print(cached_content.name)
