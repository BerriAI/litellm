# rules

- Send the OpenAI Responses payload to `{api_base}/responses`, where the base is the workspace `/serving-endpoints` URL from `api_base` or `DATABRICKS_API_BASE`
- Authenticate with the personal access token from `api_key` or `DATABRICKS_API_KEY`. OAuth M2M and Databricks SDK auth stay on Python
- Only OpenAI GPT endpoints serve Responses; other models take Python's Chat Completions bridge
- No native WebSocket mode

# references

- https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference
