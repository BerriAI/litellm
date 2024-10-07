import traceback
import requests
import uuid
import time
import tiktoken
import logging
import litellm.proxy.vault as vault

logger = logging.getLogger(__name__)


async def call_workday_gateway(data):
    try:
        vault_secrets = vault.get_api_keys(data['user_id'])
        api_gateway = vault_secrets.get("__custom::api_base")

        payload = prepare_payload(data)
        print(f"calling workday gateway: {api_gateway}, payload: {payload}")
        response = requests.post(url=api_gateway, json=payload)
        print(f"gateway response code : {response.status_code}")
        if response.status_code != 200:
            raise Exception(f"Gateway error: {response.status_code}, {response.text}")

        return get_formatted_output(data, response.json())
    except Exception as e:
        print(f"Gateway error {str(e)}")
        traceback.print_exc()
        return {
            "error": {
                "message": f"Internal Server Error",
                "code": 500
            }
        }


def prepare_payload(data):
    """Convert Litellm input format to gateway payload format"""

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]

    generation_config = {
        "temperature": data.get("temperature", 0),
        "maxOutputTokens": 8000,
        "topK": 40,
        "topP": data.get("top_p", 0.95),
        "stopSequences": [],
        "candidateCount": 1
    }

    contents = []
    for message in data.get("messages", []):
        contents.append({
            "role": message.get("role"),
            "parts": [
                {
                    "text": message.get("content")
                }
            ]
        })

    return {
        "target": {
            "provider": "gcp",
            "model": "gemini-1.5-flash"
        },
        "task": {
            "type": "gcp-multimodal-v1",
            "prediction_type": "gcp-multimodal-v1",
            "input": {
                "contents": contents,
                "safetySettings": safety_settings,
                "generationConfig": generation_config
            }
        }
    }


def convert_output(response_json):
    if response_json.get('prediction', {}).get('type', '') == 'generic-text-generation-v1':
        return response_json['prediction']['output']
    elif response_json.get('prediction', {}).get('type', '') == 'gcp-multimodal-v1':
        full_response = ''
        for chunk in response_json['prediction']['output']['chunks']:
            candidate = chunk['candidates'][0]
            if candidate['finishReason'] and candidate['finishReason'] not in ['STOP']:
                raise ValueError(candidate['finishReason'])
            part = candidate['content']['parts'][0]
            full_response += part['text']
        return full_response
    else:
        raise ValueError('Invalid prediction type passed in config')


def get_formatted_output(input_data, response_json):
    encoder = tiktoken.get_encoding("cl100k_base")
    prompt_tokens = 0
    for message in input_data.get("messages", []):
        message_content = message.get("content", "workday message")
        prompt_tokens += len(encoder.encode(message_content))

    final_message = convert_output(response_json)
    completion_tokens = len(encoder.encode(final_message))
    choices = [{
        "finish_reason": "",
        "index": 0,
        "message": {
            "content": final_message,
            "role": "assistant",
            "tool_calls": None,
            "function_call": None
        }
    }]

    return {
        "id": f"workday-{uuid.uuid4()}",
        "choices": choices,
        "created": int(time.time()),
        "model": "gemini-1.5-flash",
        "object": "chat.completion",
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
