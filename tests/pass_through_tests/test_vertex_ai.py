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

    if response.status_code != 200:
        print(f"spend logs endpoint returned {response.status_code}: {response.text}")
        return None

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


def _is_vertex_quota_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "429" in message
        or "Too Many Requests" in message
        or "RESOURCE_EXHAUSTED" in message
    )


def _parse_spend_log_start_time(value):
    import datetime

    if not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


async def get_request_spend_log(
    model_name: str,
    after_time,
    max_wait: int = 120,
    poll_interval: int = 10,
):
    """Poll /spend/logs for the individual row produced by this request: one for
    ``model_name``, started at/after ``after_time``, with spend > 0.

    Asserting on this request's own row instead of a shared daily spend total is what
    keeps the test deterministic. The total is mutated by every other passthrough test
    on the same key and is bucketed by UTC day, so a strict "total went up" check races
    both of those; a request that lands on the far side of midnight UTC or whose tiny
    delta is masked by a concurrent write would flake. Spend logging is async and
    batched, so the row still shows up a few seconds after the response returns; poll
    until it does.
    """
    import datetime
    import requests

    after_date = after_time.date()
    start_date = (after_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (after_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"{LITE_LLM_ENDPOINT}/spend/logs"
    params = {
        "api_key": "best-api-key-ever",
        "start_date": start_date,
        "end_date": end_date,
        "summarize": "false",
    }
    headers = {"Authorization": "Bearer sk-1234"}

    elapsed = 0
    while elapsed < max_wait:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"/spend/logs returned {response.status_code}: {response.text}")
            continue
        for row in response.json():
            if model_name not in (row.get("model") or ""):
                continue
            if float(row.get("spend") or 0.0) <= 0:
                continue
            row_start = _parse_spend_log_start_time(row.get("startTime"))
            if row_start is None:
                print(
                    f"matching row has unparseable startTime {row.get('startTime')!r}",
                    row,
                )
                continue
            if row_start >= after_time:
                print(f"found spend log (elapsed={elapsed}s)", row)
                return row
    return None


@pytest.mark.asyncio()
async def test_basic_vertex_ai_pass_through_with_spendlog():
    import datetime

    load_vertex_ai_credentials()

    vertexai.init(
        project="litellm-ci-cd",
        location="global",
        api_endpoint=f"{LITE_LLM_ENDPOINT}/vertex_ai",
        api_transport="rest",
    )

    model = GenerativeModel(model_name="gemini-3.1-flash-lite")
    # Capture the request time before the call (with a small buffer for clock skew) so
    # we only match the spend log this request produces, not an earlier one.
    request_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        seconds=5
    )
    try:
        response = model.generate_content("hi")
    except Exception as exc:
        if _is_vertex_quota_error(exc):
            pytest.skip("Vertex AI quota exhausted")
        raise

    print("response", response)

    spend_log = await get_request_spend_log(
        model_name="gemini-3.1-flash-lite", after_time=request_time
    )

    assert spend_log is not None, (
        "No spend log with spend > 0 was recorded for the vertex passthrough request "
        "within 120s. The request itself succeeded, so its cost-tracking write was lost "
        "or its cost was computed as zero."
    )


@pytest.mark.asyncio()
@pytest.mark.skip(reason="skip flaky test - vertex pass through streaming is flaky")
async def test_basic_vertex_ai_pass_through_streaming_with_spendlog():

    spend_before = await call_spend_logs_endpoint() or 0.0
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
