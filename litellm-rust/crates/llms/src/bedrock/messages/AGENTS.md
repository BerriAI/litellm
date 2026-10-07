# structure

- Bedrock's Messages routes, one subfolder per route: `invoke_transformations` (InvokeModel on bedrock-runtime), `claude_platform` (Claude Platform on AWS) and `mantle` (bedrock-mantle)

# invariants

- Only Claude models reach these adapters. Every other Bedrock model, and every `converse/` route, goes through the chat-completion adapter

# gotchas

- A route keeps its folder here even when it calls a different AWS endpoint, because the LiteLLM model name (`bedrock/<route>/<model>`) is what selects it
- `claude_platform` is only a route and has no provider of its own. `mantle` is also reachable through the separate `bedrock_mantle` provider, which wraps this adapter from `bedrock_mantle/messages`

# references

- https://docs.aws.amazon.com/bedrock/latest/userguide/apis.md
