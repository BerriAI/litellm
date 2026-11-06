import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useConversation } from "./useConversation";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "https://example.com"),
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
}));

vi.mock("@/lib/toast", () => ({
  toast: { fromError: vi.fn(), info: vi.fn(), success: vi.fn() },
}));

vi.mock("../utils", () => ({
  convertToDotPrompt: vi.fn(() => "---\nmodel: gpt-5.4\n---\nhello"),
  extractVariables: vi.fn(() => []),
}));

const sseBody = (text: string, splitAt: number | undefined) => {
  const bytes = new TextEncoder().encode(text);
  const parts = splitAt === undefined ? [bytes] : [bytes.slice(0, splitAt), bytes.slice(splitAt)];
  return new ReadableStream<Uint8Array>({
    start(controller) {
      parts.forEach((part) => controller.enqueue(part));
      controller.close();
    },
  });
};

const sendMessage = async (sse: string, splitAt?: number) => {
  vi.mocked(fetch).mockResolvedValue({ ok: true, body: sseBody(sse, splitAt) } as Response);
  const { result } = renderHook(() => useConversation({}, "token"));
  act(() => {
    result.current.setInputMessage("hi");
  });
  await act(async () => {
    await result.current.handleSendMessage();
  });
  return result.current.messages;
};

describe("useConversation", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("streams the reply into the last assistant message", async () => {
    const messages = await sendMessage(
      [
        'data: {"model":"gpt-5.4","choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"model":"gpt-5.4","choices":[{"delta":{"content":"lo"}}]}',
        "data: [DONE]",
        "",
      ].join("\n\n"),
    );

    expect(messages.map(({ role, content }) => ({ role, content }))).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "Hello" },
    ]);
  });

  it("shows the error event message after the partial reply when the stream ends with an error", async () => {
    const messages = await sendMessage(
      [
        'data: {"model":"gpt-5.4","choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"error":{"message":"blocked by guardrail","type":"None","code":"400"}}',
        "data: [DONE]",
        "",
      ].join("\n\n"),
    );

    expect(messages.map(({ role, content }) => ({ role, content }))).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "Hel" },
      { role: "assistant", content: "Error: blocked by guardrail" },
    ]);
  });

  it("shows an error event that arrives split across reads", async () => {
    const sse = [
      'data: {"model":"gpt-5.4","choices":[{"delta":{"content":"Hel"}}]}',
      'data: {"error":{"message":"blocked by the café guardrail"}}',
      "data: [DONE]",
      "",
    ].join("\n\n");
    const insideTheAccent = new TextEncoder().encode(sse.slice(0, sse.indexOf("é"))).length + 1;

    const messages = await sendMessage(sse, insideTheAccent);

    expect(messages.map(({ role, content }) => ({ role, content }))).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "Hel" },
      { role: "assistant", content: "Error: blocked by the café guardrail" },
    ]);
  });

  it("skips a null event and keeps streaming", async () => {
    const messages = await sendMessage(
      ["data: null", 'data: {"model":"gpt-5.4","choices":[{"delta":{"content":"Hello"}}]}', "data: [DONE]", ""].join(
        "\n\n",
      ),
    );

    expect(messages.map(({ role, content }) => ({ role, content }))).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "Hello" },
    ]);
  });

  it("replaces the empty assistant message with the error when the stream fails before any content", async () => {
    const messages = await sendMessage(
      ['data: {"error":{"message":"blocked by guardrail"}}', "data: [DONE]", ""].join("\n\n"),
    );

    expect(messages).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "Error: blocked by guardrail" },
    ]);
  });
});
