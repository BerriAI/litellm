# LiteLLM Responses API Testing

This repository contains fixes for the LiteLLM Responses API, specifically addressing tool format transformation issues.

## Quick Start

### Prerequisites
1. Python 3.x
2. Redis (for session management)

### Setup

```bash
# 1. Install Redis
# macOS:
brew install redis
brew services start redis

# Linux:
sudo apt-get install redis-server
sudo systemctl start redis

# 2. Install LiteLLM with proxy support
pip install -e ".[proxy]"

# 3. Verify Redis is running
redis-cli ping  # Should return PONG
```

### Configuration

1. Edit `responses_api_config.yaml` and add your API keys:
   - `YOUR_ANTHROPIC_API_KEY` - Get from https://console.anthropic.com/
   - `YOUR_DEEPSEEK_API_KEY` - Get from https://platform.deepseek.com/
   - `YOUR_GOOGLE_API_KEY` - Get from https://aistudio.google.com/apikey

### Running the Test

```bash
# Terminal 1: Start the proxy
litellm --config responses_api_config.yaml --port 4000

# Terminal 2: Run the test
python test_responses_api.py
```

## What This Tests

The test suite validates:

1. **Basic Responses** - Verifies each provider can return responses
2. **Session Management** - Tests context retention across multiple requests using Redis
3. **Streaming** - Validates streaming responses work correctly

## Expected Results

✅ **Working Features:**
- Basic request/response for all providers
- Session management with context retention (Claude, DeepSeek, Gemini)
- Response ID generation and session linking

⚠️ **Known Limitations:**
- Some providers may have varying context retention capabilities
- Streaming support varies by provider

## Fixes Included

This repository includes a fix for the Responses API tool format transformation issue in:
- `litellm/responses/litellm_completion_transformation/transformation.py`

The fix ensures tools are properly transformed from the nested Responses API format to the format expected by the Chat Completions API.

## Troubleshooting

### Redis Issues
```bash
# Check Redis is running
redis-cli ping

# Monitor Redis activity
redis-cli MONITOR

# Check stored sessions
redis-cli keys "litellm_patch:session:*"
```

### Proxy Issues
```bash
# Run with verbose logging
litellm --config responses_api_config.yaml --port 4000 --debug

# Check proxy health
curl http://localhost:4000/health
```

## Files

- `test_responses_api.py` - Comprehensive test suite
- `responses_api_config.yaml` - Proxy configuration
- This README