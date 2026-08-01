from langfuse import Langfuse

langfuse = Langfuse(
    host="http://localhost:4000",
    public_key="anything",
    secret_key="anything",
)

print("sending langfuse trace request")
with langfuse.start_as_current_observation(name="test-trace-litellm-proxy-passthrough"):
    pass
print("flushing langfuse request")
langfuse.flush()

print("flushed langfuse request")
