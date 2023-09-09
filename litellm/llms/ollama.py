import requests
import json

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