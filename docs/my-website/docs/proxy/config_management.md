# File Management

## `include` external YAML files in a config.yaml 

You can use `include` to include external YAML files in a config.yaml. 

**Quick Start Usage:**

To include a config file, use `include` with either a single file or a list of files. 

Contents of `parent_config.yaml`:
```yaml
include:
  - model_config.yaml # 👈 Key change, will include the contents of model_config.yaml

litellm_settings:
  callbacks: ["prometheus"] 
```


Contents of `model_config.yaml`:
```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_base: https://exampleopenaiendpoint-production.up.railway.app/
  - model_name: fake-anthropic-endpoint
    litellm_params:
      model: anthropic/fake
      api_base: https://exampleanthropicendpoint-production.up.railway.app/

```

Start proxy server 

This will start the proxy server with config `parent_config.yaml`. Since the `include` directive is used, the server will also include the contents of `model_config.yaml`.
```
litellm --config parent_config.yaml --detailed_debug
```





## Examples using `include`

Include a single file:
```yaml
include:
  - model_config.yaml
```

Include multiple files:
```yaml
include:
  - model_config.yaml
  - another_config.yaml
```