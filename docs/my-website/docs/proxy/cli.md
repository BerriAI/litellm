# CLI Arguments

This page documents all command-line interface (CLI) arguments available for the LiteLLM proxy server.

## Server Configuration

### --host
   - **Default:** `'0.0.0.0'`
   - The host for the server to listen on.
   - **Usage:** 
     ```shell
     litellm --host 127.0.0.1
     ```
   - **Usage - set Environment Variable:** `HOST`
    ```shell
    export HOST=127.0.0.1
    litellm
    ```

### --port
   - **Default:** `4000`
   - The port to bind the server to.
   - **Usage:** 
     ```shell
     litellm --port 8080
     ```
  - **Usage - set Environment Variable:** `PORT`
    ```shell
    export PORT=8080
    litellm
    ```

### --num_workers
   - **Default:** Number of logical CPUs in the system, or `4` if that cannot be determined
   - The number of uvicorn / gunicorn workers to spin up.
   - **Usage:** 
     ```shell
     litellm --num_workers 4
     ```
  - **Usage - set Environment Variable:** `NUM_WORKERS`
    ```shell
    export NUM_WORKERS=4
    litellm
    ```

### --config
   - **Short form:** `-c`
   - **Default:** `None`
   - Path to the proxy configuration file (e.g., config.yaml).
   - **Usage:** 
     ```shell
     litellm --config path/to/config.yaml
     ```

### --log_config
   - **Default:** `None`
   - **Type:** `str`
   - Path to the logging configuration file for uvicorn.
   - **Usage:** 
     ```shell
     litellm --log_config path/to/log_config.conf
     ```

### --keepalive_timeout
   - **Default:** `None`
   - **Type:** `int`
   - Set the uvicorn keepalive timeout in seconds (uvicorn timeout_keep_alive parameter).
   - **Usage:** 
     ```shell
     litellm --keepalive_timeout 30
     ```
  - **Usage - set Environment Variable:** `KEEPALIVE_TIMEOUT`
    ```shell
    export KEEPALIVE_TIMEOUT=30
    litellm
    ```

### --max_requests_before_restart
   - **Default:** `None`
   - **Type:** `int`
   - Restart worker after this many requests. This is useful for mitigating memory growth over time.
   - For uvicorn: maps to `limit_max_requests`
   - For gunicorn: maps to `max_requests`
   - **Usage:** 
     ```shell
     litellm --max_requests_before_restart 10000
     ```
  - **Usage - set Environment Variable:** `MAX_REQUESTS_BEFORE_RESTART`
    ```shell
    export MAX_REQUESTS_BEFORE_RESTART=10000
    litellm
    ```

## Server Backend Options

### --run_gunicorn
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Starts proxy via gunicorn instead of uvicorn. Better for managing multiple workers in production.
   - **Usage:** 
     ```shell
     litellm --run_gunicorn
     ```

### --run_hypercorn
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Starts proxy via hypercorn instead of uvicorn. Supports HTTP/2.
   - **Usage:** 
     ```shell
     litellm --run_hypercorn
     ```

### --skip_server_startup
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Skip starting the server after setup (useful for database migrations only).
   - **Usage:** 
     ```shell
     litellm --skip_server_startup
     ```

## SSL/TLS Configuration

### --ssl_keyfile_path
   - **Default:** `None`
   - **Type:** `str`
   - Path to the SSL keyfile. Use this when you want to provide SSL certificate when starting proxy.
   - **Usage:** 
     ```shell
     litellm --ssl_keyfile_path /path/to/key.pem --ssl_certfile_path /path/to/cert.pem
     ```
  - **Usage - set Environment Variable:** `SSL_KEYFILE_PATH`
    ```shell
    export SSL_KEYFILE_PATH=/path/to/key.pem
    litellm
    ```

### --ssl_certfile_path
   - **Default:** `None`
   - **Type:** `str`
   - Path to the SSL certfile. Use this when you want to provide SSL certificate when starting proxy.
   - **Usage:** 
     ```shell
     litellm --ssl_certfile_path /path/to/cert.pem --ssl_keyfile_path /path/to/key.pem
     ```
  - **Usage - set Environment Variable:** `SSL_CERTFILE_PATH`
    ```shell
    export SSL_CERTFILE_PATH=/path/to/cert.pem
    litellm
    ```

### --ciphers
   - **Default:** `None`
   - **Type:** `str`
   - Ciphers to use for the SSL setup. Only used with `--run_hypercorn`.
   - **Usage:** 
     ```shell
     litellm --run_hypercorn --ssl_keyfile_path /path/to/key.pem --ssl_certfile_path /path/to/cert.pem --ciphers "ECDHE+AESGCM"
     ```

## Model Configuration

### --model or -m
   - **Default:** `None`
   - The model name to pass to LiteLLM.
   - **Usage:** 
     ```shell
     litellm --model gpt-3.5-turbo
     ```

### --alias
   - **Default:** `None`
   - An alias for the model, for user-friendly reference. Use this to give a litellm model name (e.g., "huggingface/codellama/CodeLlama-7b-Instruct-hf") a more user-friendly name ("codellama").
   - **Usage:** 
     ```shell
     litellm --alias my-gpt-model
     ```

### --api_base
   - **Default:** `None`
   - The API base for the model LiteLLM should call.
   - **Usage:** 
     ```shell
     litellm --model huggingface/tinyllama --api_base https://k58ory32yinf1ly0.us-east-1.aws.endpoints.huggingface.cloud
     ```

### --api_version
   - **Default:** `2024-07-01-preview`
   - For Azure services, specify the API version.
   - **Usage:** 
     ```shell
     litellm --model azure/gpt-deployment --api_version 2023-08-01 --api_base https://<your api base>"
     ```

### --headers
   - **Default:** `None`
   - Headers for the API call (as JSON string).
   - **Usage:** 
     ```shell
     litellm --model my-model --headers '{"Authorization": "Bearer token"}'
     ```

### --add_key
   - **Default:** `None`
   - Add a key to the model configuration.
   - **Usage:** 
     ```shell
     litellm --add_key my-api-key
     ```

### --save
   - **Type:** `bool` (Flag)
   - Save the model-specific config.
   - **Usage:** 
     ```shell
     litellm --model gpt-3.5-turbo --save
     ```

## Model Parameters

### --temperature
   - **Default:** `None`
   - **Type:** `float`
   - Set the temperature for the model.
   - **Usage:** 
     ```shell
     litellm --temperature 0.7
     ```

### --max_tokens
   - **Default:** `None`
   - **Type:** `int`
   - Set the maximum number of tokens for the model output.
   - **Usage:** 
     ```shell
     litellm --max_tokens 50
     ```

### --request_timeout
   - **Default:** `None`
   - **Type:** `int`
   - Set the timeout in seconds for completion calls.
   - **Usage:** 
     ```shell
     litellm --request_timeout 300
     ```

### --max_budget
   - **Default:** `None`
   - **Type:** `float`
   - Set max budget for API calls. Works for hosted models like OpenAI, TogetherAI, Anthropic, etc.
   - **Usage:** 
     ```shell
     litellm --max_budget 100.0
     ```

### --drop_params
   - **Type:** `bool` (Flag)
   - Drop any unmapped params.
   - **Usage:** 
     ```shell
     litellm --drop_params
     ```

### --add_function_to_prompt
   - **Type:** `bool` (Flag)
   - If a function passed but unsupported, pass it as a part of the prompt.
   - **Usage:** 
     ```shell
     litellm --add_function_to_prompt
     ```

## Database Configuration

### --iam_token_db_auth
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Connects to an RDS database using IAM token authentication instead of a password. This is useful for AWS RDS instances that are configured to use IAM database authentication.
   - When enabled, LiteLLM will generate an IAM authentication token to connect to the database.
   - **Required Environment Variables:**
     - `DATABASE_HOST` - The RDS database host
     - `DATABASE_PORT` - The database port
     - `DATABASE_USER` - The database user
     - `DATABASE_NAME` - The database name
     - `DATABASE_SCHEMA` (optional) - The database schema
   - **Usage:** 
     ```shell
     litellm --iam_token_db_auth
     ```
   - **Usage - set Environment Variable:** `IAM_TOKEN_DB_AUTH`
     ```shell
     export IAM_TOKEN_DB_AUTH=True
     export DATABASE_HOST=mydb.us-east-1.rds.amazonaws.com
     export DATABASE_PORT=5432
     export DATABASE_USER=mydbuser
     export DATABASE_NAME=mydb
     litellm
     ```

### --use_prisma_db_push
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Use `prisma db push` instead of `prisma migrate` for database schema updates. This is useful when you want to quickly sync your database schema without creating migration files.
   - **Usage:** 
     ```shell
     litellm --use_prisma_db_push
     ```

## Debugging

### --debug
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Enable debugging mode for the input.
   - **Usage:** 
     ```shell
     litellm --debug
     ```
  - **Usage - set Environment Variable:** `DEBUG`
    ```shell
    export DEBUG=True
    litellm
    ```

### --detailed_debug
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Enable detailed debugging mode to view verbose debug logs.
   - **Usage:** 
     ```shell
     litellm --detailed_debug
     ```
  - **Usage - set Environment Variable:** `DETAILED_DEBUG`
    ```shell
    export DETAILED_DEBUG=True
    litellm
    ```

### --local
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - For local debugging purposes.
   - **Usage:** 
     ```shell
     litellm --local
     ```

## Testing & Health Checks

### --test
   - **Type:** `bool` (Flag)
   - Proxy chat completions URL to make a test request to.
   - **Usage:** 
     ```shell
     litellm --test
     ```

### --test_async
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Calls async endpoints `/queue/requests` and `/queue/response`.
   - **Usage:** 
     ```shell
     litellm --test_async
     ```

### --num_requests
   - **Default:** `10`
   - **Type:** `int`
   - Number of requests to hit async endpoint with (used with `--test_async`).
   - **Usage:** 
     ```shell
     litellm --test_async --num_requests 100
     ```

### --health
   - **Type:** `bool` (Flag)
   - Runs a health check on all models in config.yaml.
   - **Usage:** 
     ```shell
     litellm --health
     ```

## Other Options

### --version
   - **Short form:** `-v`
   - **Type:** `bool` (Flag)
   - Print LiteLLM version and exit.
   - **Usage:** 
     ```shell
     litellm --version
     ```

### --telemetry
   - **Default:** `True`
   - **Type:** `bool`
   - Help track usage of this feature. Turn off for privacy.
   - **Usage:** 
     ```shell
     litellm --telemetry False
     ```

### --use_queue
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - To use celery workers for async endpoints.
   - **Usage:** 
     ```shell
     litellm --use_queue
     ```
