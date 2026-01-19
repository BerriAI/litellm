# Contributing - UI

Thanks for contributing to the LiteLLM UI! This guide will help you set up your local development environment.


## 1. Clone the repo

```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm
```

## 2. Start the Proxy

Create a config file (e.g., `config.yaml`):

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o

general_settings:
  master_key: sk-1234
  database_url: postgresql://<user>:<password>@<host>:<port>/<dbname>
  store_model_in_db: true
```

Start the proxy on port 4000:

```bash
poetry run litellm --config config.yaml --port 4000
```

The UI comes pre-built in the repo. Access it at `http://localhost:4000/ui`

## 3. UI Development

There are two options for UI development:

### Option A: Build Mode (Recommended)

This builds the UI and copies it to the proxy. Changes require rebuilding.

1. Make your code changes in `ui/litellm-dashboard/src/`

2. Build the UI
```bash
cd ui/litellm-dashboard
npm install
npm run build
```

After building, copy the output to the proxy:

```bash
cp -r out/* ../../litellm/proxy/_experimental/out/
```

Then restart the proxy and access the UI at `http://localhost:4000/ui`

### Option B: Development Mode (Hot Reload)

This runs the UI on port 3000 with hot reload. The proxy runs on port 4000.

```bash
cd ui/litellm-dashboard
npm install
npm run dev
```

Access the UI at `http://localhost:3000`

:::note
Development mode may have redirect issues between ports 3000 and 4000, if so use Build Mode instead.
:::