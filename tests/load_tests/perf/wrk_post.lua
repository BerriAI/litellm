-- wrk script: POST /v1/chat/completions with a JSON body.
-- Used to drive the Rust mock gateway and LiteLLM with a non-Python client,
-- so the driver isn't the bottleneck.

wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"model":"fake-gpt","messages":[{"role":"user","content":"hi"}]}'
