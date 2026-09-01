"""Mock OpenAI-compatible provider that responds instantly."""
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        request = json.loads(body)

        response = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1234567890,
            "model": request.get("model", "mock-model"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "mock response"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

        response_body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        pass  # suppress logs

if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 11434), MockHandler)
    print("Mock provider listening on http://127.0.0.1:11434")
    server.serve_forever()
