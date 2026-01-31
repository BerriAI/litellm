# Contributing - UI

Here's how to run the LiteLLM UI locally for making changes: 

## 1. Clone the repo 
```bash
git clone https://github.com/BerriAI/litellm.git
```

## 2. Start the UI + Proxy 

**2.1 Start the proxy on port 4000** 

Tell the proxy where the UI is located
```bash
export PROXY_BASE_URL="http://localhost:3000/"

### ALSO ###  - set the basic env variables
DATABASE_URL = "postgresql://<user>:<password>@<host>:<port>/<dbname>"
LITELLM_MASTER_KEY = "sk-1234"
STORE_MODEL_IN_DB = "True"
```

```bash
cd litellm/litellm/proxy
python3 proxy_cli.py --config /path/to/config.yaml --port 4000
```

**2.2 Start the UI**

Set the mode as development (this will assume the proxy is running on localhost:4000)
```bash
export NODE_ENV="development" 
```

```bash
cd litellm/ui/litellm-dashboard

npm run dev

# starts on http://0.0.0.0:3000
```

## 3. Go to local UI 

```bash
http://0.0.0.0:3000
```