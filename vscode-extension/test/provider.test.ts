import type { ChatCompletionChunk, ChatCompletionCreateParamsStreaming } from "openai/resources/chat/completions";
import { APIError } from "openai";
import { describe, expect, it } from "vitest";
import type * as vscode from "vscode";
import type { GatewayClient, GatewayConfig, ModelGroupsResult } from "../src/gateway";
import { ESTIMATED_TOKENS_PER_IMAGE, MODE_RESPONSES } from "../src/messages";
import { GATEWAY_TIMEOUT_MESSAGE, LiteLLMChatProvider, TRUNCATED_MESSAGE, type LiteLLMModel } from "../src/provider";
import { CancellationTokenSource, LanguageModelTextPart, LanguageModelToolCallPart } from "./vscode-mock";

interface StreamRequest {
  readonly config: GatewayConfig;
  readonly params: ChatCompletionCreateParamsStreaming;
}

interface FakeGateway extends GatewayClient {
  readonly listCalls: readonly GatewayConfig[];
  readonly streamRequests: readonly StreamRequest[];
  readonly responsesRequests: readonly any[];
}

const chunk = (delta: ChatCompletionChunk.Choice.Delta, finishReason: ChatCompletionChunk.Choice["finish_reason"] = null): ChatCompletionChunk => ({
  id: "chatcmpl-1",
  object: "chat.completion.chunk",
  created: 0,
  model: "gpt-5.6",
  choices: [{ index: 0, delta, finish_reason: finishReason }],
});

const responsesEvent = (type: string, overrides?: Record<string, unknown>): any => ({
  type,
  object: "response",
  id: "resp_test",
  created_at: 1234567890,
  ...overrides,
});

const modelGroups: ModelGroupsResult = {
  kind: "ok",
  groups: [
    {
      modelGroup: "gpt-5.6",
      providers: ["openai"],
      mode: "chat",
      maxInputTokens: 922000,
      maxOutputTokens: 128000,
      inputCostPerToken: 4e-6,
      outputCostPerToken: 2e-5,
      supportsVision: true,
      supportsFunctionCalling: true,
      supportedReasoningEfforts: ["low", "high"],
    },
  ],
};

const fakeGateway = (
  listResult: ModelGroupsResult = modelGroups,
  stream: (signal: AbortSignal) => AsyncIterable<ChatCompletionChunk> = () => (async function* () {})(),
  responsesStream?: (signal: AbortSignal) => AsyncIterable<any>,
): FakeGateway => {
  const listCalls: GatewayConfig[] = [];
  const streamRequests: StreamRequest[] = [];
  const responsesRequests: any[] = [];
  return {
    listCalls,
    streamRequests,
    responsesRequests,
    async listModelGroups(config) {
      listCalls.push(config);
      return listResult;
    },
    async streamChatCompletion(config, params, signal) {
      streamRequests.push({ config, params });
      return stream(signal);
    },
    async streamResponsesCompletion(config, params, signal) {
      responsesRequests.push({ config, params });
      return responsesStream ? responsesStream(signal) : (async function* () {})();
    },
  };
};

const token = (): vscode.CancellationToken => new CancellationTokenSource().token as unknown as vscode.CancellationToken;

const prepare = (configuration: Record<string, unknown> | undefined): vscode.PrepareLanguageModelChatModelOptions =>
  ({ silent: true, configuration }) as vscode.PrepareLanguageModelChatModelOptions;

const gateway: GatewayConfig = { baseUrl: "http://127.0.0.1:4000", apiKey: "sk-test" };

const model = (overrides: Partial<LiteLLMModel> = {}): LiteLLMModel => ({
  id: "gpt-5.6",
  name: "gpt-5.6",
  family: "gpt-5.6",
  version: "1.0",
  maxInputTokens: 922000,
  maxOutputTokens: 128000,
  capabilities: { imageInput: true, toolCalling: true },
  gateway,
  mode: "chat",
  ...overrides,
});

const userMessage = (parts: readonly unknown[]): vscode.LanguageModelChatRequestMessage =>
  ({ role: 1, content: parts, name: undefined }) as vscode.LanguageModelChatRequestMessage;

const responseOptions = (overrides: Partial<vscode.ProvideLanguageModelChatResponseOptions> = {}): vscode.ProvideLanguageModelChatResponseOptions =>
  ({ toolMode: 1, ...overrides }) as vscode.ProvideLanguageModelChatResponseOptions;

const collect = (
  provider: LiteLLMChatProvider,
  cancellation: CancellationTokenSource = new CancellationTokenSource(),
): { readonly parts: readonly vscode.LanguageModelResponsePart[]; readonly run: Promise<void> } => {
  const parts: vscode.LanguageModelResponsePart[] = [];
  const run = provider.provideLanguageModelChatResponse(
    model(),
    [userMessage([{ value: "hi" }])],
    responseOptions(),
    { report: (part) => parts.push(part) },
    cancellation.token as unknown as vscode.CancellationToken,
  );
  return { parts, run };
};

describe("provideLanguageModelChatInformation", () => {
  it("returns nothing for the unconfigured probe without touching the gateway", async () => {
    const client = fakeGateway();
    expect(await new LiteLLMChatProvider(client).provideLanguageModelChatInformation(prepare(undefined), token())).toEqual([]);
    expect(client.listCalls).toEqual([]);
  });

  it("names the API key when the stored secret is gone instead of listing nothing", async () => {
    const client = fakeGateway();
    await expect(
      new LiteLLMChatProvider(client).provideLanguageModelChatInformation(prepare({ baseUrl: "http://127.0.0.1:4000" }), token()),
    ).rejects.toThrow(/missing its API key/);
    expect(client.listCalls).toEqual([]);
  });

  it("rejects a gateway URL that is not http or https", async () => {
    await expect(
      new LiteLLMChatProvider(fakeGateway()).provideLanguageModelChatInformation(prepare({ baseUrl: "litellm.example.com", apiKey: "sk" }), token()),
    ).rejects.toThrow(/"litellm.example.com" is not an http or https URL/);
  });

  it("lists the gateway's chat models with pricing, effort choices, and the gateway attached", async () => {
    const client = fakeGateway();
    const models = await new LiteLLMChatProvider(client).provideLanguageModelChatInformation(
      prepare({ baseUrl: "http://127.0.0.1:4000/v1/", apiKey: "sk-test" }),
      token(),
    );
    expect(client.listCalls).toEqual([gateway]);
    expect(models).toEqual([
      expect.objectContaining({
        id: "gpt-5.6",
        detail: "$4.00 in / $20.00 out per 1M tokens",
        inputCost: 4,
        outputCost: 20,
        pricing: "$4.00 in / $20.00 out per 1M tokens",
        priceCategory: "high",
        maxInputTokens: 922000,
        capabilities: { imageInput: true, toolCalling: true },
        configurationSchema: expect.objectContaining({ properties: expect.objectContaining({ reasoningEffort: expect.anything() }) }),
        gateway,
        mode: "chat",
      }),
    ]);
  });

  it("shows the gateway's error message, not its whole JSON body, when discovery fails", async () => {
    const body = JSON.stringify({
      error: { message: "Authentication Error, Invalid proxy server token passed", type: "auth_error", param: "sk-...abcd", code: "401" },
    });
    await expect(
      new LiteLLMChatProvider(fakeGateway({ kind: "http_error", status: 401, body })).provideLanguageModelChatInformation(
        prepare({ baseUrl: "http://127.0.0.1:4000", apiKey: "sk-bad" }),
        token(),
      ),
    ).rejects.toThrow("LiteLLM gateway at http://127.0.0.1:4000 answered 401 for /model_group/info: Authentication Error, Invalid proxy server token passed");
  });

  it("lists responses mode models preserved from gateway data", async () => {
    const responsesGroups: ModelGroupsResult = {
      kind: "ok",
      groups: [
        {
          modelGroup: "responses-model",
          providers: ["openai"],
          mode: MODE_RESPONSES,
          maxInputTokens: 128000,
          maxOutputTokens: 16000,
          inputCostPerToken: 5e-6,
          outputCostPerToken: 2.5e-5,
          supportsVision: false,
          supportsFunctionCalling: false,
          supportedReasoningEfforts: ["default", "low", "high"],
        },
      ],
    };
    const client = fakeGateway(responsesGroups);
    const models = await new LiteLLMChatProvider(client).provideLanguageModelChatInformation(
      prepare({ baseUrl: "http://127.0.0.1:4000/v1/", apiKey: "sk-test" }),
      token(),
    );
    expect(models.length).toBe(1);
    expect(models[0]).toMatchObject({ id: "responses-model", mode: MODE_RESPONSES });
  });
});

describe("provideLanguageModelChatResponse", () => {
  it("streams text and tool calls with the picked reasoning effort and a required tool choice", async () => {
    const client = fakeGateway(modelGroups, () =>
      (async function* () {
        yield chunk({ content: "Reading" });
        yield chunk({ tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "read_file", arguments: '{"path":"a"}' } }] });
        yield chunk({}, "tool_calls");
      })(),
    );
    const parts: vscode.LanguageModelResponsePart[] = [];
    await new LiteLLMChatProvider(client).provideLanguageModelChatResponse(
      model(),
      [userMessage([{ value: "read a" }])],
      responseOptions({ toolMode: 2, tools: [{ name: "read_file", description: "Read" }], modelConfiguration: { reasoningEffort: "high" } }),
      { report: (part) => parts.push(part) },
      token(),
    );
    expect(parts).toEqual([new LanguageModelTextPart("Reading"), new LanguageModelToolCallPart("call_1", "read_file", { path: "a" })]);
    expect(client.streamRequests).toEqual([
      {
        config: gateway,
        params: expect.objectContaining({ model: "gpt-5.6", reasoning_effort: "high", tool_choice: "required", tools: [expect.anything()] }),
      },
    ]);
  });

  it("finishes quietly when the user cancels mid-stream and drops its cancellation listener", async () => {
    const cancellation = new CancellationTokenSource();
    const client = fakeGateway(modelGroups, (signal) =>
      (async function* () {
        yield chunk({ content: "partial" });
        await new Promise<void>((resolve) => signal.addEventListener("abort", () => resolve(), { once: true }));
        throw new Error("Request was aborted.");
      })(),
    );
    const { parts, run } = collect(new LiteLLMChatProvider(client), cancellation);
    await new Promise((resolve) => setTimeout(resolve, 0));
    cancellation.cancel();
    await expect(run).resolves.toBeUndefined();
    expect(parts).toEqual([new LanguageModelTextPart("partial")]);
    expect(cancellation.disposedListeners).toBe(1);
  });

  it("surfaces a gateway failure as an error and still drops its cancellation listener", async () => {
    const cancellation = new CancellationTokenSource();
    const client = fakeGateway(modelGroups, () =>
      (async function* () {
        throw new Error("502 Bad Gateway");
      })(),
    );
    const { run } = collect(new LiteLLMChatProvider(client), cancellation);
    await expect(run).rejects.toThrow("502 Bad Gateway");
    expect(cancellation.disposedListeners).toBe(1);
  });

  it("explains a gateway 504 and suggests checking upstream health", async () => {
    const timeout = new APIError(504, undefined, "Gateway Time-out", new Headers());
    const client = fakeGateway(modelGroups, () => {
      throw timeout;
    });
    const { run } = collect(new LiteLLMChatProvider(client));

    await expect(run).rejects.toThrow(
      GATEWAY_TIMEOUT_MESSAGE,
    );
  });

  it("reports the text it got and then fails when the model hits its output limit", async () => {
    const client = fakeGateway(modelGroups, () =>
      (async function* () {
        yield chunk({ content: "half an ans" });
        yield chunk({}, "length");
      })(),
    );
    const { parts, run } = collect(new LiteLLMChatProvider(client));
    await expect(run).rejects.toThrow(TRUNCATED_MESSAGE);
    expect(parts).toEqual([new LanguageModelTextPart("half an ans")]);
  });

  it("fails on tool arguments that are not JSON", async () => {
    const client = fakeGateway(modelGroups, () =>
      (async function* () {
        yield chunk({ tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "grep", arguments: "{oops" } }] });
      })(),
    );
    const { run } = collect(new LiteLLMChatProvider(client));
    await expect(run).rejects.toThrow("invalid JSON arguments for tool grep");
  });

  it("calls responses API for mode='responses' models and preserves mode", async () => {
    const responsesEvents = (signal: AbortSignal): AsyncIterable<any> => ({
      [Symbol.asyncIterator]: async function* () {
        yield responsesEvent("response.created", { output: [] });
        yield responsesEvent("response.output_text.delta", { content_index: 0, delta: "Hello" });
        yield responsesEvent("response.output_text.delta", { content_index: 0, delta: " world" });
        yield responsesEvent("response.output_text.done", { content_index: 0, text: "Hello world" });
        yield responsesEvent("response.completed", { status: "completed" });
      },
    });
    const client = fakeGateway(modelGroups, () => (async function* () {})(), responsesEvents);
    const parts: vscode.LanguageModelResponsePart[] = [];
    await new LiteLLMChatProvider(client).provideLanguageModelChatResponse(
      model({ mode: MODE_RESPONSES }),
      [userMessage([{ value: "hi" }])],
      responseOptions(),
      { report: (part) => parts.push(part) },
      token(),
    );
    expect(client.responsesRequests).toHaveLength(1);
    expect(client.responsesRequests[0].params).toMatchObject({
      model: "gpt-5.6",
      stream: true,
      input: [{ type: "message", role: "user", content: [{ type: "input_text", text: "hi" }] }],
    });
    expect(client.responsesRequests[0].params).not.toHaveProperty("messages");
    expect(parts).toEqual([new LanguageModelTextPart("Hello"), new LanguageModelTextPart(" world")]);
  });
});

describe("provideTokenCount", () => {
  const provider = new LiteLLMChatProvider(fakeGateway());

  it("estimates plain text at four characters per token", async () => {
    expect(await provider.provideTokenCount(model(), "abcdefgh")).toBe(2);
  });

  it("counts tool results and tool calls, not only text parts", async () => {
    const textOnly = await provider.provideTokenCount(model(), userMessage([{ value: "ok" }]));
    const withToolResult = await provider.provideTokenCount(
      model(),
      userMessage([{ callId: "call_1", content: [{ value: "x".repeat(400) }] }, { value: "ok" }]),
    );
    expect(withToolResult).toBeGreaterThan(textOnly + 100);
  });

  it("charges a flat estimate per image instead of counting its bytes", async () => {
    const withImage = await provider.provideTokenCount(model(), userMessage([{ value: "see" }, { mimeType: "image/png", data: new Uint8Array(50000) }]));
    const withoutImage = await provider.provideTokenCount(model(), userMessage([{ value: "see" }]));
    expect(withImage - withoutImage).toBeGreaterThanOrEqual(ESTIMATED_TOKENS_PER_IMAGE);
    expect(withImage - withoutImage).toBeLessThan(ESTIMATED_TOKENS_PER_IMAGE + 20);
  });
});
