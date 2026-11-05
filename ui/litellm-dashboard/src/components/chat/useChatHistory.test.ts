import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { useChatHistory } from "./useChatHistory";
import { deleteChatImages, loadChatImages, persistChatImages } from "./chatImageStorage";
import type { Conversation } from "./types";

vi.mock("./chatImageStorage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./chatImageStorage")>()),
  persistChatImages: vi.fn(),
  loadChatImages: vi.fn(),
  deleteChatImages: vi.fn(),
}));

const aliceStorageKey = "litellm_chat_history_v1:alice";
const image = { id: "image-1", name: "design.png", url: "data:image/png;base64,aW1hZ2U=", mimeType: "image/png" };
const savedConversation: Conversation = {
  id: "conversation-1",
  title: "Design",
  model: "vision-model",
  mcpServerNames: [],
  createdAt: 1,
  updatedAt: 1,
  messages: [
    { id: "message-1", role: "user", content: "Describe this", timestamp: 1, attachments: [{ ...image, url: "" }] },
  ],
};

describe("useChatHistory", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    vi.mocked(persistChatImages).mockReset().mockResolvedValue(undefined);
    vi.mocked(loadChatImages)
      .mockReset()
      .mockResolvedValue(new Map([[image.id, image.url]]));
    vi.mocked(deleteChatImages).mockReset().mockResolvedValue(undefined);
  });

  it("should render without leaking conversations between different users on the same browser", () => {
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

  it("should reload conversations scoped to the new user when userId changes", () => {
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

  it("should save image bytes before committing lightweight history, keeping the latest streamed text", async () => {
    let finishSaving!: () => void;
    vi.mocked(persistChatImages).mockReturnValue(
      new Promise<void>((resolve) => {
        finishSaving = resolve;
      }),
    );
    const { result } = renderHook(() => useChatHistory(null, "alice"));
    let id!: string;
    act(() => {
      id = result.current.createConversation("vision-model");
    });
    act(() => {
      result.current.appendMessage(id, { role: "user", content: "", attachments: [image] });
    });
    act(() => {
      result.current.appendMessage(id, { role: "assistant", content: "A blue design" });
    });
    expect(localStorage.getItem(aliceStorageKey)).not.toContain(image.id);

    await act(async () => {
      finishSaving();
    });
    await waitFor(() => expect(localStorage.getItem(aliceStorageKey)).toContain("A blue design"));
    const stored = JSON.parse(localStorage.getItem(aliceStorageKey)!) as Conversation[];
    expect(stored[0].messages[0].attachments).toEqual([{ ...image, url: "" }]);
    expect(stored[0].title).toBe("design.png");
    expect(result.current.conversations[0].messages[0].attachments?.[0].url).toBe(image.url);
  });

  it("should restore images from the current user's media store", async () => {
    localStorage.setItem(aliceStorageKey, JSON.stringify([savedConversation]));
    const { result } = renderHook(() => useChatHistory(savedConversation.id, "alice"));
    await waitFor(() => expect(result.current.activeConversation?.messages[0].attachments?.[0].url).toBe(image.url));
    expect(loadChatImages).toHaveBeenCalledWith("alice", [image.id]);
    const { result: bob } = renderHook(() => useChatHistory(null, "bob"));
    expect(bob.current.conversations).toEqual([]);
    expect(loadChatImages).not.toHaveBeenCalledWith("bob", expect.anything());
  });

  it("should retain the previous saved history if storing an image fails", async () => {
    vi.mocked(persistChatImages).mockRejectedValue(new Error("Quota exceeded"));
    const { result } = renderHook(() => useChatHistory(null, "alice"));
    let id!: string;
    act(() => {
      id = result.current.createConversation("vision-model");
    });
    const previousHistory = localStorage.getItem(aliceStorageKey);
    act(() => {
      result.current.appendMessage(id, { role: "user", content: "Describe", attachments: [image] });
    });
    await waitFor(() => expect(result.current.storageUnavailable).toBe(true));
    expect(localStorage.getItem(aliceStorageKey)).toBe(previousHistory);
    expect(result.current.conversations[0].messages[0].attachments?.[0].url).toBe(image.url);
  });

  it("should report missing saved images without discarding their history", async () => {
    localStorage.setItem(aliceStorageKey, JSON.stringify([savedConversation]));
    vi.mocked(loadChatImages).mockResolvedValue(new Map());
    const { result } = renderHook(() => useChatHistory(savedConversation.id, "alice"));
    await waitFor(() => expect(result.current.storageUnavailable).toBe(true));
    expect(result.current.activeConversation?.messages[0].content).toBe("Describe this");
    expect(result.current.activeConversation?.messages[0].attachments?.[0].id).toBe(image.id);
  });

  it("should remove media only after the conversation deletion has been saved", async () => {
    localStorage.setItem(aliceStorageKey, JSON.stringify([savedConversation]));
    const { result } = renderHook(() => useChatHistory(savedConversation.id, "alice"));
    await waitFor(() => expect(result.current.activeConversation?.messages[0].attachments?.[0].url).toBe(image.url));
    vi.mocked(deleteChatImages).mockImplementation(async () => {
      expect(JSON.parse(localStorage.getItem(aliceStorageKey)!)).toEqual([]);
    });
    act(() => {
      result.current.deleteConversation(savedConversation.id);
    });
    await waitFor(() => expect(deleteChatImages).toHaveBeenCalledWith("alice", [image.id]));
  });

  it("should keep saved media when the localStorage deletion cannot be committed", async () => {
    localStorage.setItem(aliceStorageKey, JSON.stringify([savedConversation]));
    const { result } = renderHook(() => useChatHistory(savedConversation.id, "alice"));
    await waitFor(() => expect(result.current.activeConversation?.messages[0].attachments?.[0].url).toBe(image.url));
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("Storage blocked");
    });
    act(() => {
      result.current.deleteConversation(savedConversation.id);
    });
    await waitFor(() => expect(result.current.storageUnavailable).toBe(true));
    expect(deleteChatImages).not.toHaveBeenCalled();
    expect(localStorage.getItem(aliceStorageKey)).toContain(image.id);
  });

  it("should clean up an image deleted before its first media write completes", async () => {
    let finishSaving!: () => void;
    vi.mocked(persistChatImages).mockReturnValue(
      new Promise<void>((resolve) => {
        finishSaving = resolve;
      }),
    );
    const { result } = renderHook(() => useChatHistory(null, "alice"));
    let id!: string;
    act(() => {
      id = result.current.createConversation("vision-model");
    });
    act(() => {
      result.current.appendMessage(id, { role: "user", content: "Describe", attachments: [image] });
    });
    act(() => {
      result.current.deleteConversation(id);
    });
    expect(deleteChatImages).not.toHaveBeenCalled();
    await act(async () => {
      finishSaving();
    });
    await waitFor(() => expect(deleteChatImages).toHaveBeenCalledWith("alice", [image.id]));
    expect(JSON.parse(localStorage.getItem(aliceStorageKey)!)).toEqual([]);
  });

  it("should allow deleting a conversation whose image can no longer be restored", async () => {
    localStorage.setItem(aliceStorageKey, JSON.stringify([savedConversation]));
    vi.mocked(loadChatImages).mockResolvedValue(new Map());
    const { result } = renderHook(() => useChatHistory(savedConversation.id, "alice"));
    await waitFor(() => expect(result.current.storageUnavailable).toBe(true));
    act(() => {
      result.current.deleteConversation(savedConversation.id);
    });
    await waitFor(() => expect(result.current.storageUnavailable).toBe(false));
    expect(JSON.parse(localStorage.getItem(aliceStorageKey)!)).toEqual([]);
    act(() => {
      result.current.createConversation("text-model");
    });
    expect(localStorage.getItem(aliceStorageKey)).toContain("text-model");
  });

  it("should resume saving when an image that exceeded storage quota is removed", async () => {
    vi.mocked(persistChatImages).mockRejectedValue(new Error("Quota exceeded"));
    const { result } = renderHook(() => useChatHistory(null, "alice"));
    let id!: string;
    act(() => {
      id = result.current.createConversation("vision-model");
    });
    act(() => {
      result.current.appendMessage(id, { role: "user", content: "Describe", attachments: [image] });
    });
    await waitFor(() => expect(result.current.storageUnavailable).toBe(true));
    act(() => {
      result.current.deleteConversation(id);
    });
    await waitFor(() => expect(result.current.storageUnavailable).toBe(false));
    expect(JSON.parse(localStorage.getItem(aliceStorageKey)!)).toEqual([]);
  });

  it("should not resurrect a deleted conversation when an older hook mount finishes saving", async () => {
    let finishSaving!: () => void;
    vi.mocked(persistChatImages).mockReturnValue(
      new Promise<void>((resolve) => {
        finishSaving = resolve;
      }),
    );
    const first = renderHook(() => useChatHistory(null, "alice"));
    let id!: string;
    act(() => {
      id = first.result.current.createConversation("vision-model");
    });
    act(() => {
      first.result.current.appendMessage(id, { role: "user", content: "Describe", attachments: [image] });
    });
    first.unmount();
    const second = renderHook(() => useChatHistory(id, "alice"));
    act(() => {
      second.result.current.deleteConversation(id);
    });
    await act(async () => {
      finishSaving();
    });
    expect(JSON.parse(localStorage.getItem(aliceStorageKey)!)).toEqual([]);
    expect(second.result.current.conversations).toEqual([]);
    expect(deleteChatImages).toHaveBeenCalledWith("alice", [image.id]);
  });
  it("should save, restore, and delete document bytes with the existing user-scoped media store", async () => {
    const file = {
      id: "file-1",
      kind: "file" as const,
      name: "brief.docx",
      mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      url: "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,ZmlsZQ==",
    };
    const first = renderHook(() => useChatHistory(null, "alice"));
    let id!: string;
    act(() => {
      id = first.result.current.createConversation("file-model");
    });
    act(() => {
      first.result.current.appendMessage(id, { role: "user", content: "Read this", attachments: [file] });
    });
    await waitFor(() => expect(persistChatImages).toHaveBeenCalledWith("alice", [file]));
    await waitFor(() => expect(localStorage.getItem(aliceStorageKey)).toContain("brief.docx"));
    expect(localStorage.getItem(aliceStorageKey)).not.toContain(file.url);
    first.unmount();
    vi.mocked(loadChatImages).mockResolvedValue(new Map([[file.id, file.url]]));
    const restored = renderHook(() => useChatHistory(id, "alice"));
    await waitFor(() => expect(restored.result.current.activeConversation?.messages[0].attachments).toEqual([file]));
    act(() => {
      restored.result.current.deleteConversation(id);
    });
    await waitFor(() => expect(deleteChatImages).toHaveBeenCalledWith("alice", [file.id]));
    expect(JSON.parse(localStorage.getItem(aliceStorageKey)!)).toEqual([]);
  });
});
