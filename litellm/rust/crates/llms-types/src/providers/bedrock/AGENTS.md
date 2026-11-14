# rules

- Shared across Bedrock's chat, Messages and audio-transcription adapters: Converse and InvokeModel paths, the invocation-metrics key and the Converse response body
- The Converse response is a partial projection. Unread blocks decode as `ConverseContentBlock::Other` so adapters can decline them

# references

- https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
- https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html
- https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html
