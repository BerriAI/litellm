import { createOpenAI } from "@ai-sdk/openai";
import { stepCountIs, streamText, tool } from "ai";
import { z } from "zod";

const required = (name) => {
  const value = process.env[name];
  if (!value) {
    throw new Error(`${name} must be set`);
  }
  return value;
};

const proxyUrl = required("LITELLM_PROXY_URL");
const apiKey = required("LITELLM_API_KEY");
const modelId = required("E2E_MODEL");
const api = required("E2E_API");
const secretWord = required("E2E_SECRET_WORD");

const openai = createOpenAI({ baseURL: `${proxyUrl}/v1`, apiKey });
const modelByApi = {
  responses: () => openai.responses(modelId),
  chat: () => openai.chat(modelId),
};
if (!(api in modelByApi)) {
  throw new Error(`E2E_API must be one of ${Object.keys(modelByApi).join(", ")}, got ${api}`);
}

const result = streamText({
  model: modelByApi[api](),
  prompt:
    "Call the reveal_secret_word tool once, then reply with exactly the word it returns and nothing else.",
  tools: {
    reveal_secret_word: tool({
      description: "Reveals the secret word the user is waiting for.",
      inputSchema: z.object({
        reason: z.string().describe("Why the word is needed"),
      }),
      execute: async () => ({ word: secretWord }),
    }),
  },
  stopWhen: stepCountIs(2),
});

const summary = {
  api,
  model: modelId,
  text_deltas: 0,
  tool_calls: [],
  tool_results: [],
  errors: [],
};
for await (const part of result.fullStream) {
  switch (part.type) {
    case "text-delta":
      summary.text_deltas += 1;
      break;
    case "tool-call":
      summary.tool_calls.push({ tool_name: part.toolName, input: part.input });
      break;
    case "tool-result":
      summary.tool_results.push({ tool_name: part.toolName, output: part.output });
      break;
    case "error":
      summary.errors.push(String(part.error?.message ?? part.error));
      break;
    default:
      break;
  }
}

process.stdout.write(
  JSON.stringify({
    ...summary,
    text: await result.text,
    steps: (await result.steps).length,
    finish_reason: await result.finishReason,
  }),
);
