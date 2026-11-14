import { describe, expect, it } from "vitest";
import type * as vscode from "vscode";
import { ESTIMATED_TOKENS_PER_IMAGE, buildChatCompletionParams, estimateMessageTokens, toChatCompletionMessages, type ChatRequestInput } from "../src/messages";

const USER = 1 as vscode.LanguageModelChatMessageRole;
const ASSISTANT = 2 as vscode.LanguageModelChatMessageRole;
const SYSTEM = 3 as vscode.LanguageModelChatMessageRole;

const message = (role: vscode.LanguageModelChatMessageRole, content: readonly unknown[]): vscode.LanguageModelChatRequestMessage => ({
  role,
  content,
  name: undefined,
});

const text = (value: string): unknown => ({ value });
const image = (bytes: readonly number[], mimeType = "image/png"): unknown => ({ mimeType, data: Uint8Array.from(bytes) });
const toolCall = (callId: string, name: string, input: object): unknown => ({ callId, name, input });
const toolResult = (callId: string, content: readonly unknown[]): unknown => ({ callId, content });

const request = (overrides: Partial<ChatRequestInput> = {}): ChatRequestInput => ({
  model: "gpt-5.6",
  messages: [message(USER, [text("hi")])],
  tools: [],
  requireToolCall: false,
  reasoningEffort: undefined,
  modelOptions: {},
  ...overrides,
});

describe("toChatCompletionMessages", () => {
  it("maps system, user, and assistant text", () => {
    expect(
      toChatCompletionMessages([
        message(SYSTEM, [text("be terse")]),
        message(USER, [text("hello "), text("there")]),
        message(ASSISTANT, [text("hi")]),
      ]),
    ).toEqual([
      { role: "system", content: "be terse" },
      { role: "user", content: [{ type: "text", text: "hello " }, { type: "text", text: "there" }] },
      { role: "assistant", content: "hi" },
    ]);
  });

  it("sends user images as data URLs", () => {
    expect(toChatCompletionMessages([message(USER, [text("what is this"), image([1, 2, 3])])])).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "what is this" },
          { type: "image_url", image_url: { url: "data:image/png;base64,AQID" } },
        ],
      },
    ]);
  });

  it("round-trips tool calls and puts tool results before the user's follow-up text", () => {
    expect(
      toChatCompletionMessages([
        message(ASSISTANT, [text("checking"), toolCall("call_1", "read_file", { path: "a.ts" })]),
        message(USER, [toolResult("call_1", [text("export const a = 1;")]), text("thanks")]),
      ]),
    ).toEqual([
      {
        role: "assistant",
        content: "checking",
        tool_calls: [{ id: "call_1", type: "function", function: { name: "read_file", arguments: '{"path":"a.ts"}' } }],
      },
      { role: "tool", tool_call_id: "call_1", content: "export const a = 1;" },
      { role: "user", content: [{ type: "text", text: "thanks" }] },
    ]);
  });

  it("emits a content-less assistant turn that only called tools", () => {
    expect(toChatCompletionMessages([message(ASSISTANT, [toolCall("c", "t", {})])])).toEqual([
      { role: "assistant", content: null, tool_calls: [{ id: "c", type: "function", function: { name: "t", arguments: "{}" } }] },
    ]);
  });

  it("drops an assistant turn with neither text nor tool calls", () => {
    expect(toChatCompletionMessages([message(USER, [text("hi")]), message(ASSISTANT, [text("")]), message(USER, [text("again")])])).toEqual([
      { role: "user", content: [{ type: "text", text: "hi" }] },
      { role: "user", content: [{ type: "text", text: "again" }] },
    ]);
  });

  it("hoists images out of tool results into a user message and serializes prompt-tsx values", () => {
    expect(
      toChatCompletionMessages([
        message(USER, [toolResult("call_2", [text("screenshot:"), image([9], "image/jpeg"), { value: { node: 1 } }])]),
      ]),
    ).toEqual([
      { role: "tool", tool_call_id: "call_2", content: 'screenshot:{"node":1}' },
      { role: "user", content: [{ type: "image_url", image_url: { url: "data:image/jpeg;base64,CQ==" } }] },
    ]);
  });

  it("decodes text data parts and ignores unknown parts", () => {
    expect(toChatCompletionMessages([message(USER, [{ mimeType: "text/plain", data: Uint8Array.from([104, 105]) }, 42])])).toEqual([
      { role: "user", content: [{ type: "text", text: "hi" }] },
    ]);
  });
});

describe("estimateMessageTokens", () => {
  it("counts what the gateway will receive, tool results and tool calls included", () => {
    const plain = estimateMessageTokens(message(USER, [text("ok")]));
    const withToolResult = estimateMessageTokens(message(USER, [toolResult("call_1", [text("y".repeat(800))]), text("ok")]));
    const withToolCall = estimateMessageTokens(message(ASSISTANT, [toolCall("call_1", "read_file", { path: "z".repeat(800) })]));
    expect(plain).toBeGreaterThan(0);
    expect(withToolResult).toBeGreaterThanOrEqual(plain + 200);
    expect(withToolCall).toBeGreaterThanOrEqual(200);
  });

  it("charges each image a flat estimate rather than its base64 length", () => {
    const withoutImage = estimateMessageTokens(message(USER, [text("see")]));
    const withImages = estimateMessageTokens(message(USER, [text("see"), image(new Array(30000).fill(0)), image([1])]));
    expect(withImages - withoutImage).toBeGreaterThanOrEqual(2 * ESTIMATED_TOKENS_PER_IMAGE);
    expect(withImages - withoutImage).toBeLessThan(2 * ESTIMATED_TOKENS_PER_IMAGE + 20);
  });

  it("counts nothing for a turn the gateway will never see", () => {
    expect(estimateMessageTokens(message(ASSISTANT, []))).toBe(0);
  });
});

describe("buildChatCompletionParams", () => {
  it("streams with usage and forwards only the chosen extras", () => {
    expect(buildChatCompletionParams(request())).toEqual({
      model: "gpt-5.6",
      messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
      stream: true,
      stream_options: { include_usage: true },
    });
  });

  it("declares tools as functions and requires a call only when VS Code does", () => {
    const tools: readonly vscode.LanguageModelChatTool[] = [
      { name: "read_file", description: "Read a file", inputSchema: { type: "object", properties: { path: { type: "string" } } } },
      { name: "noop", description: "No input" },
    ];
    const auto = buildChatCompletionParams(request({ tools }));
    expect(auto.tools).toEqual([
      {
        type: "function",
        function: { name: "read_file", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" } } } },
      },
      { type: "function", function: { name: "noop", description: "No input" } },
    ]);
    expect(auto.tool_choice).toBeUndefined();
    expect(buildChatCompletionParams(request({ tools, requireToolCall: true })).tool_choice).toBe("required");
    expect(buildChatCompletionParams(request({ requireToolCall: true })).tool_choice).toBeUndefined();
  });

  it("sends reasoning_effort only when the user picked one", () => {
    expect(buildChatCompletionParams(request({ reasoningEffort: "xhigh" })).reasoning_effort).toBe("xhigh");
    expect(buildChatCompletionParams(request()).reasoning_effort).toBeUndefined();
  });

  it("forwards numeric sampling options and drops everything else", () => {
    const params = buildChatCompletionParams(
      request({ modelOptions: { temperature: 0.2, max_tokens: 500, seed: "7", foo: "bar", top_p: 0.9 } }),
    );
    expect(params).toMatchObject({ temperature: 0.2, max_tokens: 500, top_p: 0.9 });
    expect(params).not.toHaveProperty("seed");
    expect(params).not.toHaveProperty("foo");
  });
});
