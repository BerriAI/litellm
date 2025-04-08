# Notes - on how to do sagemaker streaming using boto3
import json
import boto3

import sys
import os
from dotenv import load_dotenv

load_dotenv()
import io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


class TokenIterator:
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line) + 1
                full_line = line[:-1].decode("utf-8")
                line_data = json.loads(full_line.lstrip("data:").rstrip("/n"))
                return line_data["token"]["text"]
            chunk = next(self.byte_iterator)
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])


payload = {
    "inputs": "How do I build a website?",
    "parameters": {"max_new_tokens": 256},
    "stream": True,
}


client = boto3.client("sagemaker-runtime", region_name="us-west-2")
response = client.invoke_endpoint_with_response_stream(
    EndpointName="berri-benchmarking-Llama-2-70b-chat-hf-4",
    Body=json.dumps(payload),
    ContentType="application/json",
)

# for token in TokenIterator(response["Body"]):
#     print(token)
