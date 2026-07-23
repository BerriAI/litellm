import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useConversation } from "./useConversation";

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    fromBackend: vi.fn(),
    info: vi.fn(),
    success: vi.fn(),
  },
}));

vi.mock("../utils", () => ({
  convertToDotPrompt: vi.fn().mockReturnValue(""),
  extractVariables: vi.fn().mockReturnValue([]),
}));

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "http://localhost:4000",
  getGlobalLitellmHeaderName: () => "Authorization",
}));

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useConversation", () => {
  it("aborts the in-flight request when the hook unmounts", () => {
    let capturedSignal: AbortSignal | null | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn((_input: RequestInfo, init?: RequestInit) => {
        capturedSignal = init?.signal;
        return new Promise<Response>(() => {});
      }),
    );

    const { result, unmount } = renderHook(() => useConversation({}, "sk-test"));

    act(() => {
      result.current.setInputMessage("hello");
    });
    act(() => {
      void result.current.handleSendMessage();
    });

    expect(capturedSignal).toBeDefined();
    expect(capturedSignal?.aborted).toBe(false);

    unmount();

    expect(capturedSignal?.aborted).toBe(true);
  });
});
