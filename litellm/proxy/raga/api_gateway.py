import requests
import time
import uuid


async def call_api_gateway(data):
    api_data = data.get("api_data", {})
    api_response = make_api_call(api_data)

    if "error" in api_response:
        return get_llm_formatted_error(api_response.get("error"))

    response = extract_response_field(api_data, api_response)
    return get_llm_formatted_output(response)


def make_api_call(api_data):
    url = api_data.get("endpoint", "")
    method = api_data.get("method", "").upper()
    headers = api_data.get("headers", {})
    payload = api_data.get("payload", {})

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=payload)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=payload)
        else:
            print(f"Unsupported method: {method}")
            return None

        # Parse the response based on the output schema
        if response.ok:
            return response.json()  # Returning JSON response
        else:
            return {"error": f"Request failed with status code {response.status_code}"}

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def extract_response_field(api_data, api_response):
    # return formatted response
    output_schema = api_data.get("output_schema", {})
    response_schema = output_schema.get("response")

    if response_schema in api_response:
        response = api_response.get(response_schema)
    else:
        response = str(api_response)
    return response


def get_llm_formatted_error(error_message):
    return {
        "error": {
            "message": error_message,
            "code": 500
        }
    }


def get_llm_formatted_output(message):
    choices = [{
        "finish_reason": "",
        "index": 0,
        "message": {
            "content": message,
            "role": "assistant",
            "tool_calls": None,
            "function_call": None,
        }
    }]

    return {
        "id": f"api-gateway-{uuid.uuid4()}",
        "choices": choices,
        "created": int(time.time()),
        "model": "api-gateway",
        "object": "chat.completion",
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
    }
