This directory owns shared Messages API data contracts and serialization: request and response bodies, messages, content blocks, usage, and stream-event payloads. Messages is a format independent of the provider that originated it. Keep one canonical definition of each shared contract here

Adapter contracts and execution inputs such as `MessagesTransformContext` belong in `llms/src/base_llm/messages`. Provider rewriting and interpretation belong in `llms/src/<provider>/messages`. Call envelopes, live streams, and call orchestration belong in `core/src/messages`

Represent web-search results, encrypted-content fields, and thinking configuration as data here. Decisions to flatten results, remove encrypted content, select thinking budgets, or require beta headers belong to provider transformations. Data-shape validation belongs here, while model capability checks and request adaptation do not
