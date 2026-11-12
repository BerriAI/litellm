# rules

- Chat Completions over Converse. Converse wire docs live in `llms-types/src/providers/bedrock/AGENTS.md`
- `topK` is not forwarded, because Python's placement depends on the model catalog this crate cannot see
- `invoke_handler.rs` decodes InvokeModel event streams for Claude, reusing `anthropic/chat` response handling
