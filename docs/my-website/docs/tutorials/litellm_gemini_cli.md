# Use LiteLLM with Gemini CLI

Quickstart


### 1. Install Gemini CLI

```bash
git clone https://github.com/ishaan-jaff/gemini-cli.git
cd gemini-cli
```

### 2. Point Gemini CLI to LiteLLM Proxy

Set `BASE_URL` to the LiteLLM Proxy URL and set `GEMINI_API_KEY` to your LiteLLM Proxy API key.

```bash
export BASE_URL=http://localhost:4000
export GEMINI_API_KEY=sk-1234567890
```

### 3. Run Gemini CLI

```bash
npm run build && npm start
```

### 4. Test it 

Send a test request on Gemini CLI, this will get routed to LiteLLM Proxy and then to Gemini.
