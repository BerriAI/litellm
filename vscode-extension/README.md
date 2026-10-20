# LiteLLM for VS Code

Chat in VS Code with every model your [LiteLLM AI Gateway](https://docs.litellm.ai) exposes. The extension registers LiteLLM as a language model provider, so the gateway's models show up in the chat model picker next to the built-in ones, with the price and reasoning effort controls the gateway reports for each of them

## What you get

The model list comes from the gateway's `GET /model_group/info` endpoint, scoped to the virtual key you configure, so the picker shows exactly the chat models that key can use. Each model carries its input and output price per 1M tokens in the picker and in the Language Models editor, and its context limits come from the gateway too, so VS Code sizes prompts correctly. A model whose gateway entry lists `supported_reasoning_efforts` gets a Reasoning Effort submenu in the picker's Configure Model menu, and the chosen effort is sent as `reasoning_effort` on every request to that model. Requests go to `POST /v1/chat/completions` on the gateway as streaming chat completions with tools and images passed through, so routing, fallbacks, guardrails, and spend tracking all apply as usual

## Token usage and context tracking

The extension requests usage statistics on every streaming call (`stream_options.include_usage`) and forwards the gateway's final usage chunk to VS Code as a `usage` data part. VS Code's Copilot harness reads that part, which is what makes the token counter appear in the chat view and what lets it track how much of the model's context window the conversation has consumed. That same consumption tracking drives automatic context compaction: when the rendered prompt approaches the model's `maxInputTokens` (from the gateway's `max_input_tokens`, or a 128k assumption when the gateway doesn't report one), the harness summarizes the older turns before the request goes out

## Session grouping

Every request the extension sends carries an `x-litellm-session-id` header that is stable for the lifetime of one VS Code chat conversation and unique across conversations. The LiteLLM proxy groups requests by this header, so all turns of a chat session show up as a single conversation in the dashboard's spend logs instead of one entry per request. No configuration is needed

## Setup

1. Install the extension
2. Run `Chat: Manage Language Models` from the Command Palette and pick `LiteLLM`
3. Enter a name for the connection, the gateway URL (for example `https://litellm.example.com`), and a LiteLLM virtual key. The key is stored in VS Code's secret storage
4. Open the chat model picker. The gateway's chat models are listed under the name you chose, each with its price

Add the same provider again with another name to reach a second gateway or a second key. Run `LiteLLM: Refresh Models` after the gateway's model list changes. To change the key of an existing connection or to drop it, use the gear on its row in the Language Models editor (`Update API Key`, `Delete`); to change the URL, open its entry with `Open in Language Models (JSON)` from the same menu. If the stored key is ever lost the editor shows a `missing its API key` row for that connection until you update the key

## Requirements

VS Code 1.115 or newer and a LiteLLM AI Gateway the key can reach. The key needs access to at least one model group whose mode is `chat`

## Development

```
npm ci
npm run typecheck
npm test
npm run package
```

`npm run package` writes a `.vsix` you can install with `code --install-extension litellm-vscode-<version>.vsix`
