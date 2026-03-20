import { renderHook, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { useChatHistory } from "./useChatHistory";

describe("useChatHistory", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  describe("updateTextUI", () => {
    it("should create a new assistant message when chat is empty", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Hello", "gpt-4");
      });

      expect(result.current.chatHistory).toEqual([
        { role: "assistant", content: "Hello", model: "gpt-4" },
      ]);
    });

    it("should append to the last assistant message", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Hello", "gpt-4");
      });
      act(() => {
        result.current.updateTextUI("assistant", " world");
      });

      expect(result.current.chatHistory).toEqual([
        { role: "assistant", content: "Hello world", model: "gpt-4" },
      ]);
    });

    it("should not overwrite model on subsequent chunks", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Hello", "gpt-4");
      });
      act(() => {
        result.current.updateTextUI("assistant", " world", "gpt-3.5");
      });

      expect(result.current.chatHistory[0].model).toBe("gpt-4");
    });

    it("should create a new message when role changes", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("user", "Hi");
      });
      act(() => {
        result.current.updateTextUI("assistant", "Hello", "gpt-4");
      });

      expect(result.current.chatHistory).toHaveLength(2);
      expect(result.current.chatHistory[0].role).toBe("user");
      expect(result.current.chatHistory[1].role).toBe("assistant");
    });

    it("should not append to image messages", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateImageUI("http://img.png", "dall-e");
      });
      act(() => {
        result.current.updateTextUI("assistant", "description", "gpt-4");
      });

      expect(result.current.chatHistory).toHaveLength(2);
    });

    it("should not append to audio messages", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateAudioUI("http://audio.mp3", "tts-1");
      });
      act(() => {
        result.current.updateTextUI("assistant", "text", "gpt-4");
      });

      expect(result.current.chatHistory).toHaveLength(2);
    });
  });

  describe("updateReasoningContent", () => {
    it("should add reasoning content to existing assistant message", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Answer", "gpt-4");
      });
      act(() => {
        result.current.updateReasoningContent("thinking...");
      });

      expect(result.current.chatHistory[0].reasoningContent).toBe("thinking...");
    });

    it("should append reasoning content across chunks", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "", "gpt-4");
      });
      act(() => {
        result.current.updateReasoningContent("step 1");
      });
      act(() => {
        result.current.updateReasoningContent(" step 2");
      });

      expect(result.current.chatHistory[0].reasoningContent).toBe("step 1 step 2");
    });

    it("should create assistant message with reasoning when last message is user", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.setChatHistory([{ role: "user", content: "question" }]);
      });
      act(() => {
        result.current.updateReasoningContent("thinking...");
      });

      expect(result.current.chatHistory).toHaveLength(2);
      expect(result.current.chatHistory[1]).toEqual({
        role: "assistant",
        content: "",
        reasoningContent: "thinking...",
      });
    });

    it("should not update when chat is empty", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateReasoningContent("thinking...");
      });

      expect(result.current.chatHistory).toHaveLength(0);
    });
  });

  describe("updateTimingData", () => {
    it("should add timeToFirstToken to existing assistant message", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Hello", "gpt-4");
      });
      act(() => {
        result.current.updateTimingData(150);
      });

      expect(result.current.chatHistory[0].timeToFirstToken).toBe(150);
    });

    it("should create assistant message when last is user", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.setChatHistory([{ role: "user", content: "hi" }]);
      });
      act(() => {
        result.current.updateTimingData(200);
      });

      expect(result.current.chatHistory).toHaveLength(2);
      expect(result.current.chatHistory[1].timeToFirstToken).toBe(200);
    });
  });

  describe("updateUsageData", () => {
    it("should add usage data to assistant message", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Hello", "gpt-4");
      });

      const usage = { completionTokens: 10, promptTokens: 5, totalTokens: 15 };
      act(() => {
        result.current.updateUsageData(usage);
      });

      expect(result.current.chatHistory[0].usage).toEqual(usage);
    });

    it("should add toolName when provided", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Hello", "gpt-4");
      });

      const usage = { completionTokens: 10, promptTokens: 5, totalTokens: 15 };
      act(() => {
        result.current.updateUsageData(usage, "search_tool");
      });

      expect(result.current.chatHistory[0].toolName).toBe("search_tool");
    });
  });

  describe("updateTotalLatency", () => {
    it("should add totalLatency to assistant message", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Hello", "gpt-4");
      });
      act(() => {
        result.current.updateTotalLatency(500);
      });

      expect(result.current.chatHistory[0].totalLatency).toBe(500);
    });
  });

  describe("updateA2AMetadata", () => {
    it("should add A2A metadata to assistant message", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Hello", "gpt-4");
      });

      const metadata = { taskId: "task-1", contextId: "ctx-1" };
      act(() => {
        result.current.updateA2AMetadata(metadata);
      });

      expect(result.current.chatHistory[0].a2aMetadata).toEqual(metadata);
    });
  });

  describe("updateSearchResults", () => {
    it("should add search results to assistant message", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Hello", "gpt-4");
      });

      const searchResults = [{ object: "search", search_query: "test", data: [] }];
      act(() => {
        result.current.updateSearchResults(searchResults);
      });

      expect(result.current.chatHistory[0].searchResults).toEqual(searchResults);
    });
  });

  describe("updateImageUI", () => {
    it("should add image message to history", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateImageUI("http://img.png", "dall-e-3");
      });

      expect(result.current.chatHistory).toEqual([
        { role: "assistant", content: "http://img.png", model: "dall-e-3", isImage: true },
      ]);
    });
  });

  describe("updateEmbeddingsUI", () => {
    it("should add truncated embeddings message", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateEmbeddingsUI("[0.1, 0.2, 0.3]", "text-embedding-ada");
      });

      expect(result.current.chatHistory[0].isEmbeddings).toBe(true);
      expect(result.current.chatHistory[0].model).toBe("text-embedding-ada");
    });
  });

  describe("updateAudioUI", () => {
    it("should add audio message to history", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateAudioUI("http://audio.mp3", "tts-1");
      });

      expect(result.current.chatHistory).toEqual([
        { role: "assistant", content: "http://audio.mp3", model: "tts-1", isAudio: true },
      ]);
    });
  });

  describe("updateChatImageUI", () => {
    it("should add image to existing assistant message", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Here is the image", "gpt-4");
      });
      act(() => {
        result.current.updateChatImageUI("http://img.png", "gpt-4");
      });

      expect(result.current.chatHistory[0].image).toEqual({
        url: "http://img.png",
        detail: "auto",
      });
    });

    it("should create new assistant message with image when no assistant message exists", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateChatImageUI("http://img.png", "gpt-4");
      });

      expect(result.current.chatHistory[0]).toEqual({
        role: "assistant",
        content: "",
        model: "gpt-4",
        image: { url: "http://img.png", detail: "auto" },
      });
    });
  });

  describe("handleMCPEvent", () => {
    it("should add MCP event", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.handleMCPEvent({ type: "tool_call", item_id: "1" });
      });

      expect(result.current.mcpEvents).toHaveLength(1);
    });

    it("should deduplicate events by item_id and type", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      const event = { type: "tool_call", item_id: "1" };
      act(() => {
        result.current.handleMCPEvent(event);
      });
      act(() => {
        result.current.handleMCPEvent(event);
      });

      expect(result.current.mcpEvents).toHaveLength(1);
    });

    it("should allow events without item_id (no dedup)", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.handleMCPEvent({ type: "tool_call" });
      });
      act(() => {
        result.current.handleMCPEvent({ type: "tool_call" });
      });

      expect(result.current.mcpEvents).toHaveLength(2);
    });

    it("should allow events with same item_id/type but different sequence_number", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.handleMCPEvent({ type: "tool_call", item_id: "1", sequence_number: 1 });
      });
      act(() => {
        result.current.handleMCPEvent({ type: "tool_call", item_id: "1", sequence_number: 2 });
      });

      expect(result.current.mcpEvents).toHaveLength(2);
    });
  });

  describe("clearMCPEvents", () => {
    it("should clear MCP events without affecting chat history", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Hello", "gpt-4");
        result.current.handleMCPEvent({ type: "tool_call", item_id: "1" });
      });
      act(() => {
        result.current.clearMCPEvents();
      });

      expect(result.current.mcpEvents).toEqual([]);
      expect(result.current.chatHistory).toHaveLength(1);
    });
  });

  describe("clearChatHistory", () => {
    it("should clear all state", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateTextUI("assistant", "Hello", "gpt-4");
        result.current.handleMCPEvent({ type: "tool_call", item_id: "1" });
      });
      act(() => {
        result.current.clearChatHistory();
      });

      expect(result.current.chatHistory).toEqual([]);
      expect(result.current.mcpEvents).toEqual([]);
      expect(result.current.messageTraceId).toBeNull();
      expect(result.current.responsesSessionId).toBeNull();
    });

    it("should revoke audio object URLs when clearing", () => {
      const revokeSpy = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.updateAudioUI("blob:http://localhost/audio-1", "tts-1");
      });
      act(() => {
        result.current.clearChatHistory();
      });

      expect(revokeSpy).toHaveBeenCalledWith("blob:http://localhost/audio-1");
      revokeSpy.mockRestore();
    });

    it("should clear sessionStorage when not simplified", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      sessionStorage.setItem("chatHistory", "[]");
      sessionStorage.setItem("messageTraceId", "trace-1");
      sessionStorage.setItem("responsesSessionId", "resp-1");

      act(() => {
        result.current.clearChatHistory();
      });

      expect(sessionStorage.getItem("chatHistory")).toBeNull();
      expect(sessionStorage.getItem("messageTraceId")).toBeNull();
      expect(sessionStorage.getItem("responsesSessionId")).toBeNull();
    });

    it("should NOT clear sessionStorage when simplified", () => {
      sessionStorage.setItem("chatHistory", '[{"role":"user","content":"hi"}]');

      const { result } = renderHook(() => useChatHistory({ simplified: true }));

      act(() => {
        result.current.clearChatHistory();
      });

      // simplified mode should not touch sessionStorage
      expect(sessionStorage.getItem("chatHistory")).toBe('[{"role":"user","content":"hi"}]');
    });
  });

  describe("session management", () => {
    it("handleResponseId should set responsesSessionId when useApiSessionManagement is true", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.handleResponseId("resp-123");
      });

      expect(result.current.responsesSessionId).toBe("resp-123");
    });

    it("handleResponseId should NOT set responsesSessionId when useApiSessionManagement is false", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.handleToggleSessionManagement(false);
      });
      act(() => {
        result.current.handleResponseId("resp-123");
      });

      expect(result.current.responsesSessionId).toBeNull();
    });

    it("handleToggleSessionManagement should clear session when switching to UI mode", () => {
      const { result } = renderHook(() => useChatHistory({ simplified: false }));

      act(() => {
        result.current.handleResponseId("resp-123");
      });
      act(() => {
        result.current.handleToggleSessionManagement(false);
      });

      expect(result.current.useApiSessionManagement).toBe(false);
      expect(result.current.responsesSessionId).toBeNull();
    });
  });
});
