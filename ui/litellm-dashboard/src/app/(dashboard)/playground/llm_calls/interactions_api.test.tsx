import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { toast } from "@/lib/toast";
import { makeInteractionsRequest } from "./interactions_api";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "https://example.com"),
  getGatewayBaseUrl: vi.fn(() => "https://gateway.example.com"),
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
}));

vi.mock("@/lib/toast", () => ({
  toast: { fromError: vi.fn() },
}));

const sseBody = (text: string) =>
  new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });

describe("interactions_api", () => {
  const mockUpdateUI = vi.fn();
  const mockFetch = vi.fn();

  beforeEach(() => {
    vi.stubGlobal("fetch", mockFetch);
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("parses native Gemini step events and captures the nested interaction model", async () => {
    const sse = [
      'data: {"event_type":"interaction.created","interaction":{"model":"gemini-3-flash"}}',
      'data: {"event_type":"step.start","index":0,"step":{"type":"text"}}',
      'data: {"event_type":"step.delta","index":0,"delta":{"type":"text","text":"Hello"}}',
      'data: {"event_type":"step.delta","index":0,"delta":{"type":"text","text":" world"}}',
      'data: {"event_type":"step.stop","index":0}',
      'data: {"event_type":"interaction.completed","interaction":{"model":"gemini-3-flash"}}',
      "data: [DONE]",
      "",
      "",
    ].join("\n\n");

    mockFetch.mockResolvedValue({ ok: true, body: sseBody(sse) } as Response);

    await makeInteractionsRequest("hi", mockUpdateUI, "gemini-flash", "token");

    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe("https://gateway.example.com/v1beta/interactions");
    expect(mockUpdateUI).toHaveBeenCalledTimes(2);
    expect(mockUpdateUI).toHaveBeenNthCalledWith(1, "Hello", "gemini-3-flash");
    expect(mockUpdateUI).toHaveBeenNthCalledWith(2, " world", "gemini-3-flash");
  });

  it("parses bridge step events and captures the top-level model", async () => {
    const sse = [
      'data: {"event_type":"interaction.created","id":"x","model":"gpt-5.4","status":"in_progress"}',
      'data: {"event_type":"step.delta","index":0,"delta":{"type":"text","text":"Hi"}}',
      'data: {"event_type":"interaction.completed","id":"x","model":"gpt-5.4"}',
      "data: [DONE]",
      "",
      "",
    ].join("\n\n");

    mockFetch.mockResolvedValue({ ok: true, body: sseBody(sse) } as Response);

    await makeInteractionsRequest("hi", mockUpdateUI, "gpt-mini", "token");

    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(mockFetch.mock.calls[0][0]).toBe("https://gateway.example.com/v1beta/interactions");
    expect(mockUpdateUI).toHaveBeenCalledTimes(1);
    expect(mockUpdateUI).toHaveBeenCalledWith("Hi", "gpt-5.4");
  });

  it("rejects and toasts the message when the stream ends with an error event", async () => {
    const sse = [
      'data: {"event_type":"step.delta","index":0,"delta":{"type":"text","text":"Hi"}}',
      'data: {"error":{"message":"blocked by guardrail","type":"None","code":"400"}}',
      "data: [DONE]",
      "",
      "",
    ].join("\n\n");

    mockFetch.mockResolvedValue({ ok: true, body: sseBody(sse) } as Response);

    await expect(makeInteractionsRequest("hi", mockUpdateUI, "gpt-mini", "token")).rejects.toThrow(
      "blocked by guardrail",
    );

    expect(mockUpdateUI).toHaveBeenCalledTimes(1);
    expect(mockUpdateUI).toHaveBeenCalledWith("Hi", "gpt-mini");
    expect(toast.fromError).toHaveBeenCalledWith(expect.stringContaining("blocked by guardrail"));
  });
});
