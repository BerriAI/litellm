import { renderHook, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { useChatHistory } from "./useChatHistory";

describe("useChatHistory", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("creates unique conversation and message IDs without crypto.randomUUID", () => {
    vi.stubGlobal("crypto", { getRandomValues: crypto.getRandomValues.bind(crypto) });
    const { result } = renderHook(() => useChatHistory(null, "http-user"));

    act(() => {
      result.current.createConversation("test-model");
    });
    const conversationId = result.current.currentActiveId!;
    act(() => {
      result.current.appendMessage(conversationId, { role: "user", content: "Hello" });
      result.current.appendMessage(conversationId, { role: "assistant", content: "Hi" });
    });

    const conversation = result.current.activeConversation!;
    const ids = [conversation.id, ...conversation.messages.map((message) => message.id)];
    expect(new Set(ids).size).toBe(3);
    for (const id of ids) {
      expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
    }
    expect(conversation.messages.map((message) => message.content)).toEqual(["Hello", "Hi"]);
  });

  it("does not leak conversations between different users on the same browser", () => {
    const { result: alice } = renderHook(({ userId }) => useChatHistory(null, userId), {
      initialProps: { userId: "alice" },
    });
    act(() => {
      alice.current.createConversation("gpt-4");
    });
    expect(alice.current.conversations).toHaveLength(1);

    const { result: bob } = renderHook(({ userId }) => useChatHistory(null, userId), {
      initialProps: { userId: "bob" },
    });
    expect(bob.current.conversations).toHaveLength(0);
  });

  it("reloads conversations scoped to the new user when userId changes", () => {
    const { result, rerender } = renderHook(({ userId }) => useChatHistory(null, userId), {
      initialProps: { userId: "alice" },
    });
    act(() => {
      result.current.createConversation("gpt-4");
    });
    expect(result.current.conversations).toHaveLength(1);

    rerender({ userId: "bob" });
    expect(result.current.conversations).toHaveLength(0);

    rerender({ userId: "alice" });
    expect(result.current.conversations).toHaveLength(1);
  });
});
