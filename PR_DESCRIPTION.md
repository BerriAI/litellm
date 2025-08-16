## Summary

This PR adds conditional authorization header formatting to support Ollama Turbo (ollama.com), which requires a different authentication format than self-hosted Ollama instances.

- When api_base starts with 'https://ollama.com', use raw API key without 'Bearer' prefix
- For all other URLs, use standard 'Bearer {api_key}' format
- Updated auth headers in chat completion, text completion, and embeddings
- Added tests to verify the authorization header behavior

This change enables LiteLLM to work with Ollama Turbo (ollama.com) which requires a different authentication format than self-hosted Ollama instances.

## Title

feat: Add conditional auth header for Ollama Turbo

## Relevant issues

N/A - Feature addition

## Pre-Submission checklist

**Please complete all items before asking a LiteLLM maintainer to review your PR**

- [x] I have Added testing in the [`tests/`](https://github.com/BerriAI/litellm/tree/main/tests/) directory, **Adding at least 1 test is a hard requirement** - [see details](https://docs.litellm.ai/docs/extras/contributing_code)
- [x] I have added a screenshot of my new test passing locally 
- [x] My PR passes all unit tests on [`make test-unit`](https://docs.litellm.ai/docs/extras/contributing_code)
- [x] My PR's scope is as isolated as possible, it only solves 1 specific problem

## Type

🆕 New Feature

## Changes

- Added conditional logic to check if `api_base` starts with `'https://ollama.com'`
- Updated 4 locations in `ollama_chat.py` to use conditional auth headers
- Updated `chat/transformation.py` for chat completions
- Updated `completion/transformation.py` for text completions
- Updated `completion/handler.py` for embeddings
- Added `api_key` parameter to embedding calls in `main.py`
- Created comprehensive test suite in `test_ollama_turbo_integration.py`

## Test Results

<img width="1176" height="762" alt="image" src="https://github.com/user-attachments/assets/e4516648-4ad6-4a12-8b0b-ab0fe03ee01f" />

All 4 auth header tests pass successfully, verifying:
- ✅ Ollama.com URLs receive auth header without "Bearer" prefix
- ✅ Environment variable OLLAMA_API_KEY is correctly handled
- ✅ All endpoints (chat, embeddings, vision) use correct auth format
- ✅ No regressions - existing Ollama functionality preserved