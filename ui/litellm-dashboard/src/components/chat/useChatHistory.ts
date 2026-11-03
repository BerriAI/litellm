import { v4 as uuid } from "uuid";
import { useCallback, useEffect, useState } from "react";
import { AssistantMessageUpdate, ChatMessage, Conversation } from "./types";
import {
  deleteChatImages,
  getConversationImages,
  loadChatImages,
  persistChatImages,
  withoutImageData,
  withRestoredImages,
} from "./chatImageStorage";

const STORAGE_KEY_PREFIX = "litellm_chat_history_v1";
const MAX_CONVERSATIONS = 100;
const TITLE_MAX_LENGTH = 40;
const saveVersions = new Map<string, number>();
const latestImageIds = new Map<string, Set<string>>();

function generateTitle(firstUserMessage: string): string {
  const trimmed = firstUserMessage.trim();
  if (trimmed.length <= TITLE_MAX_LENGTH) {
    return trimmed;
  }
  return trimmed.slice(0, TITLE_MAX_LENGTH) + "…";
}

function storageKeyFor(userId: string): string {
  return `${STORAGE_KEY_PREFIX}:${encodeURIComponent(userId)}`;
}

function loadFromStorage(storageKey: string): { conversations: Conversation[]; storageUnavailable: boolean } {
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) {
      return { conversations: [], storageUnavailable: false };
    }
    const parsed = JSON.parse(raw) as Conversation[];
    return { conversations: parsed, storageUnavailable: false };
  } catch {
    return { conversations: [], storageUnavailable: true };
  }
}

function saveToStorage(storageKey: string, conversations: Conversation[]): boolean {
  try {
    localStorage.setItem(storageKey, JSON.stringify(conversations));
    return true;
  } catch {
    return false;
  }
}

function trimConversations(conversations: Conversation[]): Conversation[] {
  if (conversations.length <= MAX_CONVERSATIONS) {
    return conversations;
  }
  return [...conversations].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_CONVERSATIONS);
}

export function useChatHistory(
  activeConversationId: string | null,
  userId: string,
): {
  conversations: Conversation[];
  activeConversation: Conversation | null;
  currentActiveId: string | null;
  storageUnavailable: boolean;
  staleId: boolean;
  createConversation: (model: string) => string;
  appendMessage: (conversationId: string, message: Omit<ChatMessage, "id" | "timestamp">) => void;
  updateLastAssistantMessage: (conversationId: string, updates: AssistantMessageUpdate) => void;
  /** Remove the message with `messageId` and all subsequent messages from the conversation. */
  truncateFromMessage: (conversationId: string, messageId: string) => void;
  deleteConversation: (id: string) => void;
  renameConversation: (id: string, newTitle: string) => void;
  setActiveConversationId: (id: string | null) => void;
} {
  const [conversations, setConversations] = useState<Conversation[]>(
    () => loadFromStorage(storageKeyFor(userId)).conversations,
  );
  const [storageUnavailable, setStorageUnavailable] = useState<boolean>(
    () => loadFromStorage(storageKeyFor(userId)).storageUnavailable,
  );
  const [historyLoadFailed, setHistoryLoadFailed] = useState<boolean>(
    () => loadFromStorage(storageKeyFor(userId)).storageUnavailable,
  );
  const [staleId, setStaleId] = useState(false);
  const [currentActiveId, setCurrentActiveId] = useState<string | null>(activeConversationId);

  // Sync internal active id whenever the URL-derived prop changes (e.g. "New chat" → null)
  const [prevActiveConversationId, setPrevActiveConversationId] = useState(activeConversationId);
  if (activeConversationId !== prevActiveConversationId) {
    setPrevActiveConversationId(activeConversationId);
    setCurrentActiveId(activeConversationId);
    setStaleId(false);
  }

  // Reload conversations from storage whenever userId changes (e.g. auth resolving after mount)
  const [prevUserId, setPrevUserId] = useState(userId);
  if (userId !== prevUserId) {
    setPrevUserId(userId);
    const { conversations: loaded, storageUnavailable: unavailable } = loadFromStorage(storageKeyFor(userId));
    setConversations(loaded);
    setStorageUnavailable(unavailable);
    setHistoryLoadFailed(unavailable);

    if (activeConversationId !== null && !loaded.some((c) => c.id === activeConversationId)) {
      setStaleId(true);
    }
  }

  useEffect(() => {
    let canceled = false;
    const stored = loadFromStorage(storageKeyFor(userId)).conversations;
    const missingIds = getConversationImages(stored)
      .filter((image) => !image.url)
      .map((image) => image.id);
    if (missingIds.length) {
      loadChatImages(userId, missingIds)
        .then((images) => {
          if (canceled) return;
          setConversations((current) => withRestoredImages(current, images));
          if (new Set(missingIds).size !== images.size) setStorageUnavailable(true);
        })
        .catch(() => {
          if (!canceled) setStorageUnavailable(true);
        });
    }
    return () => {
      canceled = true;
    };
  }, [userId]);

  // Commit image bytes before replacing the lightweight conversation history.
  useEffect(() => {
    const images = getConversationImages(conversations);
    latestImageIds.set(userId, new Set(images.map((image) => image.id)));
    if (historyLoadFailed) return;
    let canceled = false;
    const version = (saveVersions.get(userId) ?? 0) + 1;
    saveVersions.set(userId, version);
    const reportFailure = () => {
      if (!canceled) setStorageUnavailable(true);
    };
    const removeUnreferencedImages = (ids: string[]) => {
      const savedImages = getConversationImages(loadFromStorage(storageKeyFor(userId)).conversations);
      const retainedIds = new Set([...savedImages.map((image) => image.id), ...(latestImageIds.get(userId) ?? [])]);
      const removedIds = [...new Set(ids)].filter((id) => !retainedIds.has(id));
      if (removedIds.length) void deleteChatImages(userId, removedIds).catch(reportFailure);
    };
    const save = () => {
      if (saveVersions.get(userId) !== version) {
        removeUnreferencedImages(images.map((image) => image.id));
        return;
      }
      const storageKey = storageKeyFor(userId);
      const previous = loadFromStorage(storageKey).conversations;
      if (!saveToStorage(storageKey, withoutImageData(conversations))) {
        queueMicrotask(reportFailure);
        return;
      }
      if (!images.some((image) => !image.url)) {
        queueMicrotask(() => {
          if (!canceled) setStorageUnavailable(false);
        });
      }
      removeUnreferencedImages(getConversationImages(previous).map((image) => image.id));
    };
    if (images.some((image) => image.url.startsWith("data:"))) {
      void persistChatImages(userId, images).then(save).catch(reportFailure);
    } else save();
    return () => {
      canceled = true;
    };
  }, [conversations, userId, historyLoadFailed]);

  const createConversation = useCallback((model: string): string => {
    const id = uuid();
    const now = Date.now();
    const newConversation: Conversation = {
      id,
      title: "New conversation",
      model,
      messages: [],
      mcpServerNames: [],
      createdAt: now,
      updatedAt: now,
    };
    setConversations((prev) => trimConversations([newConversation, ...prev]));
    setCurrentActiveId(id);
    return id;
  }, []);

  const appendMessage = useCallback((conversationId: string, message: Omit<ChatMessage, "id" | "timestamp">) => {
    const newMessage: ChatMessage = {
      ...message,
      id: uuid(),
      timestamp: Date.now(),
    };

    setConversations((prev) => {
      const updated = prev.map((conv) => {
        if (conv.id !== conversationId) return conv;
        const updatedMessages = [...conv.messages, newMessage];
        let title = conv.title;
        if (
          title === "New conversation" &&
          newMessage.role === "user" &&
          conv.messages.filter((m) => m.role === "user").length === 0
        ) {
          title = generateTitle(newMessage.content || newMessage.attachments?.[0]?.name || "Image conversation");
        }
        return { ...conv, title, messages: updatedMessages, updatedAt: Date.now() };
      });
      return trimConversations(updated);
    });
  }, []);

  const updateLastAssistantMessage = useCallback((conversationId: string, updates: AssistantMessageUpdate) => {
    setConversations((prev) => {
      const updated = prev.map((conv) => {
        if (conv.id !== conversationId) return conv;
        const messages = [...conv.messages];
        const lastAssistantIndex = messages.reduceRight((found, msg, idx) => {
          if (found !== -1) return found;
          return msg.role === "assistant" ? idx : -1;
        }, -1);
        if (lastAssistantIndex === -1) return conv;
        messages[lastAssistantIndex] = { ...messages[lastAssistantIndex], ...updates };
        return { ...conv, messages, updatedAt: Date.now() };
      });
      return trimConversations(updated);
    });
  }, []);

  const truncateFromMessage = useCallback((conversationId: string, messageId: string) => {
    setConversations((prev) => {
      const updated = prev.map((conv) => {
        if (conv.id !== conversationId) return conv;
        const idx = conv.messages.findIndex((m) => m.id === messageId);
        if (idx === -1) return conv;
        return { ...conv, messages: conv.messages.slice(0, idx), updatedAt: Date.now() };
      });
      return trimConversations(updated);
    });
  }, []);

  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => trimConversations(prev.filter((c) => c.id !== id)));
      if (currentActiveId === id) setCurrentActiveId(null);
    },
    [currentActiveId],
  );

  const renameConversation = useCallback((id: string, newTitle: string) => {
    setConversations((prev) =>
      trimConversations(
        prev.map((conv) => (conv.id === id ? { ...conv, title: newTitle, updatedAt: Date.now() } : conv)),
      ),
    );
  }, []);

  const setActiveConversationId = useCallback((id: string | null) => {
    setCurrentActiveId(id);
    setStaleId(false);
  }, []);

  const activeConversation =
    currentActiveId !== null ? conversations.find((c) => c.id === currentActiveId) ?? null : null;

  return {
    conversations,
    activeConversation,
    currentActiveId,
    storageUnavailable,
    staleId,
    createConversation,
    appendMessage,
    updateLastAssistantMessage,
    truncateFromMessage,
    deleteConversation,
    renameConversation,
    setActiveConversationId,
  };
}
