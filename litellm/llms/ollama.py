import requests
import json

try:
    from async_generator import async_generator, yield_  # optional dependency
    async_generator_imported = True
except ImportError:
    async_generator_imported = False  # this should not throw an error, it will impact the 'import litellm' statement

# ollama implementation
def get_ollama_response_stream(
        api_base="http://localhost:11434",
        model="llama2",
        prompt="Why is the sky blue?"
    ):
    url = f"{api_base}/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
    }
    session = requests.Session()

    with session.post(url, json=data, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                try:
                    json_chunk = line.decode("utf-8")
                    chunks = json_chunk.split("\n")
                    for chunk in chunks:
                        if chunk.strip() != "":
                            j = json.loads(chunk)
                            if "response" in j:
                                completion_obj = {
                                    "role": "assistant",
                                    "content": "",
                                }
                                completion_obj["content"] = j["response"]
                                yield {"choices": [{"delta": completion_obj}]}
                except Exception as e:
                    print(f"Error decoding JSON: {e}")
    session.close()

if async_generator_imported:
    # ollama implementation
    @async_generator
    async def async_get_ollama_response_stream(
            api_base="http://localhost:11434",
            model="llama2",
            prompt="Why is the sky blue?"
        ):
        url = f"{api_base}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
        }
        session = requests.Session()

        with session.post(url, json=data, stream=True) as resp:
            for line in resp.iter_lines():
                if line:
                    try:
                        json_chunk = line.decode("utf-8")
                        chunks = json_chunk.split("\n")
                        for chunk in chunks:
                            if chunk.strip() != "":
                                j = json.loads(chunk)
                                if "response" in j:
                                    completion_obj = {
                                        "role": "assistant",
                                        "content": "",
                                    }
                                    completion_obj["content"] = j["response"]
                                    await yield_({"choices": [{"delta": completion_obj}]})
                    except Exception as e:
                        print(f"Error decoding JSON: {e}")
        session.close()