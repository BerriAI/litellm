# CLI Arguments
Cli arguments,  --host, --port, --num_workers

## --host
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

## --port
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

## --num_workers
   - **Default:** `1`
   - The number of uvicorn workers to spin up.
   - **Usage:** 
     ```shell
     litellm --num_workers 4
     ```
  - **Usage - set Environment Variable:** `NUM_WORKERS`
    ```shell
    export NUM_WORKERS=4
    litellm
    ```

## --api_base
   - **Default:** `None`
   - The API base for the model litellm should call.
   - **Usage:** 
     ```shell
     litellm --model huggingface/tinyllama --api_base https://k58ory32yinf1ly0.us-east-1.aws.endpoints.huggingface.cloud
     ```

## --api_version
   - **Default:** `None`
   - For Azure services, specify the API version.
   - **Usage:** 
     ```shell
     litellm --model azure/gpt-deployment --api_version 2023-08-01 --api_base https://<your api base>"
     ```

## --model or -m
   - **Default:** `None`
   - The model name to pass to Litellm.
   - **Usage:** 
     ```shell
     litellm --model gpt-3.5-turbo
     ```

## --test
   - **Type:** `bool` (Flag)
   - Proxy chat completions URL to make a test request.
   - **Usage:** 
     ```shell
     litellm --test
     ```

## --health
   - **Type:** `bool` (Flag)
   - Runs a health check on all models in config.yaml
   - **Usage:** 
     ```shell
     litellm --health
     ```

## --alias
   - **Default:** `None`
   - An alias for the model, for user-friendly reference.
   - **Usage:** 
     ```shell
     litellm --alias my-gpt-model
     ```

## --debug
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

## --detailed_debug
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Enable debugging mode for the input.
   - **Usage:** 
     ```shell
     litellm --detailed_debug
     ```
  - **Usage - set Environment Variable:** `DETAILED_DEBUG`
    ```shell
    export DETAILED_DEBUG=True
    litellm
    ```

#### --temperature
   - **Default:** `None`
   - **Type:** `float`
   - Set the temperature for the model.
   - **Usage:** 
     ```shell
     litellm --temperature 0.7
     ```

## --max_tokens
   - **Default:** `None`
   - **Type:** `int`
   - Set the maximum number of tokens for the model output.
   - **Usage:** 
     ```shell
     litellm --max_tokens 50
     ```

## --request_timeout
   - **Default:** `6000`
   - **Type:** `int`
   - Set the timeout in seconds for completion calls.
   - **Usage:** 
     ```shell
     litellm --request_timeout 300
     ```

## --drop_params
   - **Type:** `bool` (Flag)
   - Drop any unmapped params.
   - **Usage:** 
     ```shell
     litellm --drop_params
     ```

## --add_function_to_prompt
   - **Type:** `bool` (Flag)
   - If a function passed but unsupported, pass it as a part of the prompt.
   - **Usage:** 
     ```shell
     litellm --add_function_to_prompt
     ```

## --config
   - Configure Litellm by providing a configuration file path.
   - **Usage:** 
     ```shell
     litellm --config path/to/config.yaml
     ```

## --telemetry
   - **Default:** `True`
   - **Type:** `bool`
   - Help track usage of this feature.
   - **Usage:** 
     ```shell
     litellm --telemetry False
     ```


## --log_config
   - **Default:** `None`
   - **Type:** `str`
   - Specify a log configuration file for uvicorn.
   - **Usage:** 
     ```shell
     litellm --log_config path/to/log_config.conf
     ```

## --skip_server_startup
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Skip starting the server after setup (useful for DB migrations only).
   - **Usage:** 
     ```shell
     litellm --skip_server_startup
     ```