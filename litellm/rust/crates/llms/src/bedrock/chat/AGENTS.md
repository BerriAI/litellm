# rules

- Chat Completions over Converse. Converse wire docs live in `llms-types/src/providers/bedrock/AGENTS.md`
- `topK` is not forwarded, because Python's placement depends on the model catalog this crate cannot see
- `invoke_handler.rs` ports Python's `AWSEventStreamDecoder`: it decodes InvokeModel event streams the way botocore's `EventStreamBuffer` does
- `invoke_transformations/base_invoke_transformation.rs` ports the InvokeModel helpers of Python's `base_invoke_transformation.py`
