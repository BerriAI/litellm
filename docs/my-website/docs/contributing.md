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

### Option A: Development Mode (Hot Reload)

This runs the UI on port 3000 with hot reload. The proxy runs on port 4000.

```bash
cd ui/litellm-dashboard
npm install
npm run dev
```

**Login flow:**
1. Go to `http://localhost:3000`
2. You'll be redirected to `http://localhost:4000/ui` for login
3. After logging in, manually navigate back to `http://localhost:3000/`
4. You're now authenticated and can develop with hot reload

:::note
If you experience redirect loops or authentication issues, clear your browser cookies for localhost or use Build Mode instead.
:::

### Option B: Build Mode

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

## 4. Submitting a PR

1. Create a new branch for your changes:
```bash
git checkout -b feat/your-feature-name
```

2. Stage and commit your changes:
```bash
git add .
git commit -m "feat: description of your changes"
```

3. Push to your fork:
```bash
git push origin feat/your-feature-name
```

4. Create a Pull Request on GitHub following the [PR template](https://github.com/BerriAI/litellm/blob/main/.github/pull_request_template.md)
