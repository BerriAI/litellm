import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import litellm
from litellm.caching.caching import Cache
from litellm.rust_bridge import _native
from litellm.rust_bridge.messages.entrypoints import LiteLLMMessagesRequest


class Upstream(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.calls.append(body)
        usage = {"input_tokens": 2, "output_tokens": 3, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 5}
        message = {"id": "msg_native", "type": "message", "role": "assistant", "model": "test-native-sdk",
                   "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn", "stop_sequence": None,
                   "usage": usage}
        if body.get("stream"):
            events = ({"type": "message_start", "message": {**message, "usage": {**usage, "output_tokens": 0}}},
                      {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
                      {"type": "message_stop"})
            response = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
        else:
            response = json.dumps(message).encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream" if body.get("stream") else "application/json")
        self.send_header("content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args):
        pass


class MessagesLifecycle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.original = (litellm.cache, litellm.max_budget, litellm._current_cost, litellm.modify_params)
        litellm.cache = Cache(type="local", supported_call_types=["anthropic_messages"])
        litellm.max_budget = 100.0
        litellm._current_cost = 0.0
        litellm.modify_params = False
        litellm.register_model({"test-native-sdk": {"litellm_provider": "anthropic", "mode": "chat",
            "input_cost_per_token": 2.0, "output_cost_per_token": 3.0,
            "cache_read_input_token_cost": 0.5, "cache_creation_input_token_cost": 4.0,
            "max_input_tokens": 200, "max_output_tokens": 50}})
        self.before = len(Upstream.calls)

    async def asyncTearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        litellm.cache, litellm.max_budget, litellm._current_cost, litellm.modify_params = self.original
        litellm.model_cost.pop("test-native-sdk", None)

    def call(self, asynchronous=True, **extra):
        kwargs = {"max_tokens": 100, "messages": [{"role": "user", "content": "hello"}],
                  "model": "test-native-sdk", "custom_llm_provider": "anthropic", "api_key": "test",
                  "api_base": f"http://127.0.0.1:{self.server.server_port}", **extra}
        request = LiteLLMMessagesRequest(model=kwargs["model"], messages=kwargs["messages"], max_tokens=100,
            stream=kwargs.get("stream"), api_key="test", api_base=kwargs["api_base"],
            custom_llm_provider="anthropic", kwargs=kwargs)
        return (_native.amessages if asynchronous else _native.messages)(request, (), kwargs)

    async def test_cache_hit_charges_once_and_budget_precedes_cache(self):
        first = await self.call()
        second = await self.call()
        self.assertEqual(first, second)
        self.assertEqual(len(Upstream.calls) - self.before, 1)
        cost = 2 * 2.0 + 3 * 3.0 + 4 * 0.5 + 5 * 4.0
        self.assertEqual(litellm._current_cost, cost)
        litellm.max_budget = cost - 1
        with self.assertRaises(litellm.BudgetExceededError):
            await self.call()
        self.assertEqual(len(Upstream.calls) - self.before, 1)

    async def test_stream_cache_replays_and_charges_once(self):
        first = b"".join([chunk async for chunk in await self.call(stream=True)])
        second = b"".join([chunk async for chunk in await self.call(stream=True)])
        self.assertEqual(first, second)
        self.assertEqual(len(Upstream.calls) - self.before, 1)
        self.assertEqual(litellm._current_cost, 2 * 2.0 + 3 * 3.0 + 4 * 0.5 + 5 * 4.0)

    async def test_opt_in_output_cap_is_sent_and_cached_under_original_request(self):
        litellm.modify_params = True
        await self.call()
        await self.call()
        self.assertEqual(len(Upstream.calls) - self.before, 1)
        self.assertEqual(Upstream.calls[-1]["max_tokens"], 50)

    async def test_sync_messages_use_the_same_cache_and_budget(self):
        first = self.call(asynchronous=False)
        second = await self.call()
        self.assertEqual(first, second)
        self.assertEqual(len(Upstream.calls) - self.before, 1)


if __name__ == "__main__":
    print(f"native extension: {_native.__file__}")
    unittest.main()
