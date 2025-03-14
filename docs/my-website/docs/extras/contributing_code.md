# Contributing Code

## **Checklist before submitting a PR**

Here are the core requirements for any PR submitted to LiteLLM


- [ ] Add testing, **Adding at least 1 test is a hard requirement** - [see details](#2-adding-testing-to-your-pr)
- [ ] Ensure your PR passes the following tests:
    - [ ] [Unit Tests](#3-running-unit-tests)
    - [ ] [Formatting / Linting Tests](#35-running-linting-tests)
- [ ] Keep scope as isolated as possible. As a general rule, your changes should address 1 specific problem at a time



## Quick start

## 1. Setup your local dev environment


Here's how to modify the repo locally:

Step 1: Clone the repo

```shell
git clone https://github.com/BerriAI/litellm.git
```

Step 2: Install dev dependencies:

```shell
poetry install --with dev --extras proxy
```

That's it, your local dev environment is ready!

## 2. Adding Testing to your PR

- Add your test to the [`tests/litellm/` directory](https://github.com/BerriAI/litellm/tree/main/tests/litellm)

- This directory 1:1 maps the the `litellm/` directory, and can only contain mocked tests.
- Do not add real llm api calls to this directory.

### 2.1 File Naming Convention for `tests/litellm/`

The `tests/litellm/` directory follows the same directory structure as `litellm/`.

- `litellm/proxy/test_caching_routes.py` maps to `litellm/proxy/caching_routes.py`
- `test_{filename}.py` maps to `litellm/{filename}.py`

## 3. Running Unit Tests

run the following command on the root of the litellm directory

```shell
make test-unit
```

## 3.5 Running Linting Tests

run the following command on the root of the litellm directory

```shell
make lint
```

LiteLLM uses mypy for linting. On ci/cd we also run `black` for formatting.

## 4. Submit a PR with your changes!

- push your fork to your GitHub repo
- submit a PR from there


## Advanced
### Building LiteLLM Docker Image 

Some people might want to build the LiteLLM docker image themselves. Follow these instructions if you want to build / run the LiteLLM Docker Image yourself.

Step 1: Clone the repo

```shell
git clone https://github.com/BerriAI/litellm.git
```

Step 2: Build the Docker Image

Build using Dockerfile.non_root

```shell
docker build -f docker/Dockerfile.non_root -t litellm_test_image .
```

Step 3: Run the Docker Image

Make sure config.yaml is present in the root directory. This is your litellm proxy config file.

```shell
docker run \
    -v $(pwd)/proxy_config.yaml:/app/config.yaml \
    -e DATABASE_URL="postgresql://xxxxxxxx" \
    -e LITELLM_MASTER_KEY="sk-1234" \
    -p 4000:4000 \
    litellm_test_image \
    --config /app/config.yaml --detailed_debug
```
