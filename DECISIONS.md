# Decisions

The cooldown regression uses fixed logging timestamps to prove a provider 408 after the caller's TTFT deadline still counts for non-streaming and synchronous streaming calls. The async stream tests use the existing controlled clock.

The expiry helper checks both the exception and a MidStreamFallbackError's original exception. Only the boolean True qualifies, so truthy values cannot grant the exemption.

The existing deployment-stall test now covers caller-selected deadlines too and checks both cooldown state and failure counts. The stream cleanup test also verifies that the real router expiry exception grants the exemption without elapsed-time metadata.

Live QA uses a local scripted provider because a real provider cannot reliably produce a delayed 408 on demand. The same proxy configuration, two workers and dedicated Redis instance cover the original PR tip and the fix. Provider 408s are checked through Chat Completions, Messages and Responses. Caller-selected and deployment-configured TTFT expiry are checked through streaming Chat Completions.

The touched files already have strict lint and pyright diagnostics. Validation compares their diagnostics with the unchanged PR tip in a temporary checkout, using the same tools and environment, and requires no new diagnostics.
