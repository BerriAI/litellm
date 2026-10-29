# Bedrock batch endpoints

Set `aws_bedrock_control_plane_endpoint` on a deployment when batch create, retrieve, and cancel requests must use a custom Bedrock control-plane endpoint. The proxy signs these requests for the configured URL. Without this setting, batch requests use the standard regional Bedrock endpoint

```yaml
model_list:
  - model_name: bedrock-batch
    litellm_params:
      model: bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
      aws_region_name: us-east-1
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_batch_role_arn: os.environ/AWS_BATCH_ROLE_ARN
      s3_output_bucket_name: os.environ/AWS_S3_OUTPUT_BUCKET_NAME
      aws_bedrock_control_plane_endpoint: os.environ/BEDROCK_CONTROL_PLANE_ENDPOINT
```

Set `BEDROCK_CONTROL_PLANE_ENDPOINT` to your Bedrock control-plane VPC endpoint or compatible gateway base URL, without `/model-invocation-job`. The proxy adds the operation path. The endpoint must be reachable from the proxy, and the configured AWS region must match the endpoint's signing region

`api_base` and `aws_bedrock_runtime_endpoint` configure runtime inference separately. They do not override the batch control-plane endpoint. `s3_endpoint_url` configures S3 file access separately

The control-plane setting belongs in deployment configuration. Sending `aws_bedrock_control_plane_endpoint` in an API request body is rejected. There is no automatic `AWS_BEDROCK_CONTROL_PLANE_ENDPOINT` environment fallback, so reference the desired environment variable explicitly as shown above

Start the proxy with this configuration and set `LITELLM_API_BASE` to its URL. Set `BEDROCK_BATCH_INPUT_URI` to a valid input JSONL file in S3 that the batch service role can read

```bash
curl -sS "$LITELLM_API_BASE/v1/batches" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"bedrock-batch\",\"input_file_id\":\"$BEDROCK_BATCH_INPUT_URI\",\"endpoint\":\"/v1/chat/completions\",\"completion_window\":\"24h\"}"
```

Use the returned `id` as `BATCH_ID` to retrieve or cancel the same job through the deployment's configured endpoint

```bash
curl -sS "$LITELLM_API_BASE/v1/batches/$BATCH_ID?model=bedrock-batch" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"

curl -sS -X POST "$LITELLM_API_BASE/v1/batches/$BATCH_ID/cancel?model=bedrock-batch" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```
