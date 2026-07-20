# Routing Utils

Shared helpers for deciding how a LiteLLM model routes to an LLM provider.
Keep provider-name parsing, explicit `custom_llm_provider` handling, and model-prefix normalization here.
Do not put deployment selection or load-balancing logic here; that belongs in `router`.
Do not put provider HTTP transformation logic here; that belongs in `providers`.
Helpers in this folder should be deterministic and easy to unit test without network calls.
