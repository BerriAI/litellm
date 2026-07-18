import { useCallback, useEffect, useState } from "react";
import { ChatMessage, Conversation } from "./types";

const STORAGE_KEY_PREFIX = "litellm_chat_history_v1";
const MAX_CONVERSATIONS = 100;
const TITLE_MAX_LENGTH = 40;

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
  updateLastAssistantMessage: (
    conversationId: string,
    updates: Partial<Pick<ChatMessage, "content" | "reasoningContent" | "mcpEvents">>,
  ) => void;
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

    if (activeConversationId !== null && !loaded.some((c) => c.id === activeConversationId)) {
      setStaleId(true);
    }
  }

  // Persist to localStorage after every conversations change
  useEffect(() => {
    if (storageUnavailable) return;
    const success = saveToStorage(storageKeyFor(userId), conversations);
    if (!success) {
      // Defer: a setState call directly in an effect body causes a synchronous
      // cascading render; queuing it as a microtask callback avoids that.
      queueMicrotask(() => setStorageUnavailable(true));
    }
  }, [conversations, userId, storageUnavailable]);

  const createConversation = useCallback((model: string): string => {
    const id = crypto.randomUUID();
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
      id: crypto.randomUUID(),
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
          title = generateTitle(newMessage.content);
        }
        return { ...conv, title, messages: updatedMessages, updatedAt: Date.now() };
      });
      return trimConversations(updated);
    });
  }, []);

  const updateLastAssistantMessage = useCallback(
    (conversationId: string, updates: Partial<Pick<ChatMessage, "content" | "reasoningContent" | "mcpEvents">>) => {
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
    },
    [],
  );

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
