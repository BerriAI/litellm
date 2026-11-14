"""
Vertex AI Agent Engine A2A provider.

Maps the A2A contextId onto Agent Engine user/session ids so multi-turn agents
keep their session across A2A calls, then uses the completion bridge.
"""
