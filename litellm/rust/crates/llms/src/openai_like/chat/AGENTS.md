# rules

- The shared config for every OpenAI-compatible host. Format spec lives in `llms-types/src/formats/chat_completions/AGENTS.md`
- Keeps Python's two deviations: `max_completion_tokens` becomes `max_tokens`, and null usage counts become zero
- Declines tool requests before the call, since tool-call responses are not normalized yet
