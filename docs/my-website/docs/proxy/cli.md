# CLI Arguments
Cli arguments,  --host, --port, --num_workers

#### --host
   - **Default:** `'0.0.0.0'`
   - The host for the server to listen on.
   - **Usage:** 
     ```shell
     litellm --host 127.0.0.1
     ```

#### --port
   - **Default:** `8000`
   - The port to bind the server to.
   - **Usage:** 
     ```shell
     litellm --port 8080
     ```

#### --num_workers
   - **Default:** `1`
   - The number of uvicorn workers to spin up.
   - **Usage:** 
     ```shell
     litellm --num_workers 4
     ```

#### --api_base
   - **Default:** `None`
   - The API base for the model litellm should call.
   - **Usage:** 
     ```shell
     litellm --model huggingface/tinyllama --api_base https://k58ory32yinf1ly0.us-east-1.aws.endpoints.huggingface.cloud
     ```

#### --api_version
   - **Default:** `None`
   - For Azure services, specify the API version.
   - **Usage:** 
     ```shell
     litellm --model azure/gpt-deployment --api_version 2023-08-01 --api_base https://<your api base>"
     ```

#### --model or -m
   - **Default:** `None`
   - The model name to pass to Litellm.
   - **Usage:** 
     ```shell
     litellm --model gpt-3.5-turbo
     ```

#### --test
   - **Type:** `bool` (Flag)
   - Proxy chat completions URL to make a test request.
   - **Usage:** 
     ```shell
     litellm --test
     ```

#### --health
   - **Type:** `bool` (Flag)
   - Runs a health check on all models in config.yaml
   - **Usage:** 
     ```shell
     litellm --health
     ```

#### --alias
   - **Default:** `None`
   - An alias for the model, for user-friendly reference.
   - **Usage:** 
     ```shell
     litellm --alias my-gpt-model
     ```

#### --debug
   - **Default:** `False`
   - **Type:** `bool` (Flag)
   - Enable debugging mode for the input.
   - **Usage:** 
     ```shell
     litellm --debug
     ```

#### --temperature
   - **Default:** `None`
   - **Type:** `float`
   - Set the temperature for the model.
   - **Usage:** 
     ```shell
     litellm --temperature 0.7
     ```

#### --max_tokens
   - **Default:** `None`
   - **Type:** `int`
   - Set the maximum number of tokens for the model output.
   - **Usage:** 
     ```shell
     litellm --max_tokens 50
     ```

#### --request_timeout
   - **Default:** `600`
   - **Type:** `int`
   - Set the timeout in seconds for completion calls.
   - **Usage:** 
     ```shell
     litellm --request_timeout 300
     ```

#### --drop_params
   - **Type:** `bool` (Flag)
   - Drop any unmapped params.
   - **Usage:** 
     ```shell
     litellm --drop_params
     ```

#### --add_function_to_prompt
   - **Type:** `bool` (Flag)
   - If a function passed but unsupported, pass it as a part of the prompt.
   - **Usage:** 
     ```shell
     litellm --add_function_to_prompt
     ```

#### --config
   - Configure Litellm by providing a configuration file path.
   - **Usage:** 
     ```shell
     litellm --config path/to/config.yaml
     ```

#### --telemetry
   - **Default:** `True`
   - **Type:** `bool`
   - Help track usage of this feature.
   - **Usage:** 
     ```shell
     litellm --telemetry False
     ```


