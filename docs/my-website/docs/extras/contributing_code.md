# Contributing Code

Here are the core requirements for any PR submitted to LiteLLM

- Follow the [fork and pull request workflow](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)
- Fill out the relevant issue(s) your PR solves
- Add testing, **Adding atleast 1 test is a hard requirement**
- Ensure your PR passes the following tests 
    - Unit Tests
    - Formatting / Linting Tests
- Keep scope as isolated as possible. As a general rule, your changes should address 1 specific problem at a time




## Quick start

## 1. Setup your local dev environment


Here's how to modify the repo locally:

Step 1: Clone the repo

```shell
git clone https://github.com/BerriAI/litellm.git
```

Step 2: Install dependencies:

```shell
pip install -r requirements.txt
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




### Checklist for PRs



Step 3: Test your change:

a. Add a pytest test within `tests/litellm/`

This folder follows the same directory structure as `litellm/`.

If a corresponding test file does not exist, create one.

b. Run the test

```shell
cd tests/litellm # pwd: Documents/litellm/litellm/tests/litellm
pytest /path/to/test_file.py
```

Step 4: Submit a PR with your changes! 🚀

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
