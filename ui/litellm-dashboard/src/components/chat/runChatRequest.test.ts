import { beforeEach, describe, expect, it, vi } from "vitest";
import { makeOpenAIResponsesRequest } from "@/components/llm_calls/responses_api";
import { generateChatImages, toResponsesMessages } from "./imageHelpers";
import { runChatRequest } from "./runChatRequest";
import type { MCPEvent } from "./types";

vi.mock("@/components/llm_calls/responses_api", () => ({ makeOpenAIResponsesRequest: vi.fn() }));
vi.mock("./imageHelpers", () => ({
  generateChatImages: vi.fn(),
  toResponsesMessages: vi.fn((messages) => messages),
}));

function requestOptions(overrides: Partial<Parameters<typeof runChatRequest>[0]> = {}) {
  return {
    model: "chat-model",
    mode: "chat" as const,
    prompt: "Hello",
    accessToken: "test-token",
    signal: new AbortController().signal,
    messages: [{ role: "user" as const, content: "Hello" }],
    previousResponseId: "previous-response",
    mcpServers: ["search"],
    onResponseId: vi.fn(),
    onUpdate: vi.fn(),
    ...overrides,
  };
}

const event: MCPEvent = { type: "response.mcp_call.completed", item_id: "tool-call-1" };

describe("runChatRequest", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(toResponsesMessages).mockImplementation((messages) => messages);
  });

  it("should accumulate chat content, reasoning, metrics, and completed tool events", async () => {
    const options = requestOptions();
    const usage = { promptTokens: 4, completionTokens: 7 };
    vi.mocked(makeOpenAIResponsesRequest).mockImplementation(async (...args) => {
      args[1]("assistant", "Hello");
      args[1]("assistant", " there");
      args[6]?.("First");
      args[6]?.(" thought");
      args[7]?.(120);
      args[8]?.(usage);
      args[15]?.("new-response");
      args[16]?.(event);
      args[24]?.(350);
      expect(options.onUpdate).not.toHaveBeenCalledWith({ mcpEvents: [event] });
    });

    await runChatRequest(options);
    expect(options.onUpdate).toHaveBeenCalledWith({ content: "Hello there" });
    expect(options.onUpdate).toHaveBeenCalledWith({ reasoningContent: "First thought" });
    expect(options.onUpdate).toHaveBeenCalledWith({ timeToFirstToken: 120 });
    expect(options.onUpdate).toHaveBeenCalledWith({ totalLatency: 350 });
    expect(options.onUpdate).toHaveBeenCalledWith({ usage });
    expect(options.onUpdate).toHaveBeenLastCalledWith({ mcpEvents: [event] });
    expect(options.onResponseId).toHaveBeenCalledWith("new-response");
    const args = vi.mocked(makeOpenAIResponsesRequest).mock.calls[0];
    expect(args[5]).toBe(options.signal);
    expect(args[13]).toEqual(["search"]);
    expect(args[14]).toBe("previous-response");
  });

  it("should preserve partial text on abort without persisting partial tools or late response ids", async () => {
    const controller = new AbortController();
    const options = requestOptions({ signal: controller.signal });
    vi.mocked(makeOpenAIResponsesRequest).mockImplementation(async (...args) => {
      args[1]("assistant", "Partial answer");
      args[16]?.(event);
      controller.abort();
      args[15]?.("late-response");
      throw new DOMException("Stopped", "AbortError");
    });

    await runChatRequest(options);
    expect(options.onUpdate).toHaveBeenLastCalledWith({ content: "Partial answer [stopped]" });
    expect(options.onUpdate).not.toHaveBeenCalledWith({ mcpEvents: [event] });
    expect(options.onResponseId).toHaveBeenCalledExactlyOnceWith(null);
  });

  it("should preserve partial output and reset the session after a request error", async () => {
    const options = requestOptions();
    vi.mocked(makeOpenAIResponsesRequest).mockImplementation(async (...args) => {
      args[1]("assistant", "Partial answer");
      args[16]?.(event);
      throw new Error("Provider unavailable");
    });
    await runChatRequest(options);
    expect(options.onUpdate).toHaveBeenLastCalledWith({ content: "Partial answer" });
    expect(options.onUpdate).not.toHaveBeenCalledWith({ mcpEvents: [event] });
    expect(options.onResponseId).toHaveBeenCalledWith(null);
  });

  it("should show an error when no response text has arrived", async () => {
    const options = requestOptions();
    vi.mocked(makeOpenAIResponsesRequest).mockRejectedValue(new Error("Provider unavailable"));
    await runChatRequest(options);
    expect(options.onUpdate).toHaveBeenLastCalledWith({ content: "Provider unavailable" });
    expect(options.onResponseId).toHaveBeenCalledWith(null);
  });

  it("should return generated images through the same message update callback", async () => {
    const options = requestOptions({ mode: "image", model: "image-model" });
    const images = [{ id: "image-1", name: "Generated image 1", url: "https://images.example/image.png" }];
    vi.mocked(generateChatImages).mockResolvedValue(images);
    await runChatRequest(options);
    expect(options.onUpdate).toHaveBeenCalledExactlyOnceWith({ generatedImages: images });
    expect(makeOpenAIResponsesRequest).not.toHaveBeenCalled();
    const expectedRequest = {
      prompt: "Hello",
      model: "image-model",
      accessToken: "test-token",
      signal: options.signal,
    };
    expect(generateChatImages).toHaveBeenCalledWith(expectedRequest);
  });

  it("should discard image results that arrive after the request was stopped", async () => {
    const controller = new AbortController();
    const options = requestOptions({ mode: "image", signal: controller.signal });
    vi.mocked(generateChatImages).mockImplementation(async () => {
      controller.abort();
      return [{ id: "image-1", name: "Generated image 1", url: "https://images.example/image.png" }];
    });
    await runChatRequest(options);
    expect(options.onUpdate).toHaveBeenCalledExactlyOnceWith({ content: " [stopped]" });
    expect(options.onResponseId).toHaveBeenCalledWith(null);
  });
});
