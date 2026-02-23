# Contributing Code

## Checklist before submitting a PR

Here are the core requirements for any PR submitted to LiteLLM:

- [ ] Sign the [Contributor License Agreement (CLA)](#contributor-license-agreement-cla)
- [ ] Keep scope as isolated as possible — your changes should address **one specific problem** at a time

### Proxy (Backend) PRs

- [ ] Add testing — **at least 1 test is a hard requirement** ([details](#2-adding-tests))
- [ ] Ensure your PR passes:
  - [ ] [Unit Tests](#3-running-unit-tests) — `make test-unit`
  - [ ] [Formatting / Linting Tests](#4-running-linting-tests) — `make lint`

### UI PRs

- [ ] Ensure the UI builds successfully — `npm run build`
- [ ] Ensure all UI unit tests pass — `npm run test`
- [ ] If you are adding a **new component** or **new logic**, add corresponding tests

## Contributor License Agreement (CLA)

Before contributing code to LiteLLM, you must sign our [Contributor License Agreement (CLA)](https://cla-assistant.io/BerriAI/litellm). This is a legal requirement for all contributions to be merged into the main repository. The CLA helps protect both you and the project by clearly defining the terms under which your contributions are made.

**Important:** We strongly recommend signing the CLA **before** starting work on your contribution to avoid delays in the review process. You can find and sign the CLA [here](https://cla-assistant.io/BerriAI/litellm).

---

## Proxy (Backend)

### 1. Setting up your local dev environment

Step 1: Clone the repo

```shell
git clone https://github.com/BerriAI/litellm.git
```

Step 2: Install dev dependencies

```shell
poetry install --with dev --extras proxy
```

### 2. Adding tests

- Add your tests to the [`tests/test_litellm/` directory](https://github.com/BerriAI/litellm/tree/main/tests/litellm).
- This directory mirrors the `litellm/` directory 1:1 and should **only** contain mocked tests.
- **Do not** add real LLM API calls to this directory.

#### File naming convention for `tests/test_litellm/`

The test directory follows the same structure as `litellm/`:

- `test_{filename}.py` maps to `litellm/{filename}.py`
- `litellm/proxy/test_caching_routes.py` maps to `litellm/proxy/caching_routes.py`

### 3. Running unit tests

Run the following command from the root of the `litellm` directory:

```shell
make test-unit
```

### 4. Running linting tests

Run the following command from the root of the `litellm` directory:

```shell
make lint
```

LiteLLM uses `mypy` for type checking. CI/CD also runs `black` for formatting.

### 5. Submit a PR

- Push your changes to your fork on GitHub
- Open a Pull Request from your fork

---

## UI

### 1. Setting up your local dev environment

Step 1: Clone the repo

```shell
git clone https://github.com/BerriAI/litellm.git
```

Step 2: Navigate to the UI dashboard directory

```shell
cd ui/litellm-dashboard
```

Step 3: Install dependencies

```shell
npm install
```

Step 4: Start the development server

```shell
npm run dev
```

### 2. Adding tests

If you are adding a **new component** or **new logic**, you must add corresponding tests.

### 3. Running UI unit tests

```shell
npm run test
```

### 4. Building the UI

Ensure the UI builds successfully before submitting your PR:

```shell
npm run build
```

### 5. Submit a PR

- Push your changes to your fork on GitHub
- Open a Pull Request from your fork

---

## Advanced

### Building the LiteLLM Docker Image

Follow these instructions if you want to build and run the LiteLLM Docker image yourself.

Step 1: Clone the repo

```shell
git clone https://github.com/BerriAI/litellm.git
```

Step 2: Build the Docker image

Build using `Dockerfile.non_root`:

```shell
docker build -f docker/Dockerfile.non_root -t litellm_test_image .
```

Step 3: Run the Docker image

Make sure `config.yaml` is present in the root directory. This is your LiteLLM proxy config file.

```shell
docker run \
    -v $(pwd)/proxy_config.yaml:/app/config.yaml \
    -e DATABASE_URL="postgresql://xxxxxxxx" \
    -e LITELLM_MASTER_KEY="sk-1234" \
    -p 4000:4000 \
    litellm_test_image \
    --config /app/config.yaml --detailed_debug
```

### Running the LiteLLM Proxy Locally

1. Navigate to the `proxy/` directory:

```shell
cd litellm/litellm/proxy
```

2. Run the proxy:

```shell
python3 proxy_cli.py --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```
