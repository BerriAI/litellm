"""
Anthropic Messages API Structured Outputs Test Suite

E2E tests for structured outputs functionality across different providers:
- Direct Anthropic API
- Azure AI Foundry Anthropic models
- AWS Bedrock Invoke API
- AWS Bedrock Converse API

All tests validate that the output_format parameter works correctly
and returns valid JSON instead of Markdown text.
"""