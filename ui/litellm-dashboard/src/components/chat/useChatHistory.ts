import { useCallback, useEffect, useState } from "react";
import { ChatMessage, Conversation } from "./types";

const STORAGE_KEY = "litellm_chat_history_v1";
const MAX_CONVERSATIONS = 100;
const TITLE_MAX_LENGTH = 40;

function generateTitle(firstUserMessage: string): string {
  const trimmed = firstUserMessage.trim();
  if (trimmed.length <= TITLE_MAX_LENGTH) {
    return trimmed;
  }
  return trimmed.slice(0, TITLE_MAX_LENGTH);
}

function loadFromStorage(): { conversations: Conversation[]; storageUnavailable: boolean } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return { conversations: [], storageUnavailable: false };
    }
    const parsed = JSON.parse(raw) as Conversation[];
    return { conversations: parsed, storageUnavailable: false };
  } catch {
    return { conversations: [], storageUnavailable: true };
  }
}

function saveToStorage(conversations: Conversation[]): boolean {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    return true;
  } catch {
    return false;
  }
}

function trimConversations(conversations: Conversation[]): Conversation[] {
  if (conversations.length <= MAX_CONVERSATIONS) {
    return conversations;
  }
  return [...conversations]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, MAX_CONVERSATIONS);
}

export function useChatHistory(activeConversationId: string | null): {
  conversations: Conversation[];
  activeConversation: Conversation | null;
  storageUnavailable: boolean;
  staleId: boolean;
  createConversation: (model: string) => string;
  appendMessage: (conversationId: string, message: Omit<ChatMessage, "id" | "timestamp">) => void;
  updateLastAssistantMessage: (conversationId: string, updates: Partial<Pick<ChatMessage, "content" | "reasoningContent">>) => void;
  deleteConversation: (id: string) => void;
  renameConversation: (id: string, newTitle: string) => void;
  setActiveConversationId: (id: string | null) => void;
} {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [storageUnavailable, setStorageUnavailable] = useState(false);
  const [staleId, setStaleId] = useState(false);
  const [currentActiveId, setCurrentActiveId] = useState<string | null>(activeConversationId);

  useEffect(() => {
    const { conversations: loaded, storageUnavailable: unavailable } = loadFromStorage();
    setConversations(loaded);
    setStorageUnavailable(unavailable);

    if (activeConversationId !== null) {
      const found = loaded.some((c) => c.id === activeConversationId);
      if (!found) {
        setStaleId(true);
      }
    }
  }, []);

  const persistConversations = useCallback(
    (updated: Conversation[]) => {
      const trimmed = trimConversations(updated);
      setConversations(trimmed);
      if (!storageUnavailable) {
        const success = saveToStorage(trimmed);
        if (!success) {
          setStorageUnavailable(true);
        }
      }
    },
    [storageUnavailable],
  );

  const createConversation = useCallback(
    (model: string): string => {
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
      persistConversations([newConversation, ...conversations]);
      setCurrentActiveId(id);
      return id;
    },
    [conversations, persistConversations],
  );

  const appendMessage = useCallback(
    (conversationId: string, message: Omit<ChatMessage, "id" | "timestamp">) => {
      const newMessage: ChatMessage = {
        ...message,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
      };

      setConversations((prev) => {
        const updated = prev.map((conv) => {
          if (conv.id !== conversationId) {
            return conv;
          }

          const updatedMessages = [...conv.messages, newMessage];

          let title = conv.title;
          if (
            title === "New conversation" &&
            newMessage.role === "user" &&
            conv.messages.filter((m) => m.role === "user").length === 0
          ) {
            title = generateTitle(newMessage.content);
          }

          return {
            ...conv,
            title,
            messages: updatedMessages,
            updatedAt: Date.now(),
          };
        });

        const trimmed = trimConversations(updated);
        if (!storageUnavailable) {
          const success = saveToStorage(trimmed);
          if (!success) {
            setStorageUnavailable(true);
          }
        }
        return trimmed;
      });
    },
    [storageUnavailable],
  );

  const updateLastAssistantMessage = useCallback(
    (
      conversationId: string,
      updates: Partial<Pick<ChatMessage, "content" | "reasoningContent">>,
    ) => {
      setConversations((prev) => {
        const updated = prev.map((conv) => {
          if (conv.id !== conversationId) {
            return conv;
          }

          const messages = [...conv.messages];
          const lastAssistantIndex = messages.reduceRight((found, msg, idx) => {
            if (found !== -1) return found;
            return msg.role === "assistant" ? idx : -1;
          }, -1);

          if (lastAssistantIndex === -1) {
            return conv;
          }

          messages[lastAssistantIndex] = {
            ...messages[lastAssistantIndex],
            ...updates,
          };

          return {
            ...conv,
            messages,
            updatedAt: Date.now(),
          };
        });

        const trimmed = trimConversations(updated);
        if (!storageUnavailable) {
          const success = saveToStorage(trimmed);
          if (!success) {
            setStorageUnavailable(true);
          }
        }
        return trimmed;
      });
    },
    [storageUnavailable],
  );

  const deleteConversation = useCallback(
    (id: string) => {
      const updated = conversations.filter((c) => c.id !== id);
      persistConversations(updated);
      if (currentActiveId === id) {
        setCurrentActiveId(null);
      }
    },
    [conversations, currentActiveId, persistConversations],
  );

  const renameConversation = useCallback(
    (id: string, newTitle: string) => {
      const updated = conversations.map((conv) =>
        conv.id === id ? { ...conv, title: newTitle, updatedAt: Date.now() } : conv,
      );
      persistConversations(updated);
    },
    [conversations, persistConversations],
  );

  const setActiveConversationId = useCallback((id: string | null) => {
    setCurrentActiveId(id);
    setStaleId(false);
  }, []);

  const activeConversation =
    currentActiveId !== null
      ? (conversations.find((c) => c.id === currentActiveId) ?? null)
      : null;

  return {
    conversations,
    activeConversation,
    storageUnavailable,
    staleId,
    createConversation,
    appendMessage,
    updateLastAssistantMessage,
    deleteConversation,
    renameConversation,
    setActiveConversationId,
  };
}
