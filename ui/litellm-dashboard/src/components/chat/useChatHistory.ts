import { useCallback, useEffect, useRef, useState } from "react";
import { ChatMessage, Conversation } from "./types";

const STORAGE_KEY = "litellm_chat_history_v1";
const MAX_CONVERSATIONS = 100;
const TITLE_MAX_LENGTH = 40;

function generateTitle(firstUserMessage: string): string {
  const trimmed = firstUserMessage.trim();
  if (trimmed.length <= TITLE_MAX_LENGTH) {
    return trimmed;
  }
  return trimmed.slice(0, TITLE_MAX_LENGTH) + "…";
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
  truncateAfterMessage: (conversationId: string, messageId: string) => void;
  deleteConversation: (id: string) => void;
  renameConversation: (id: string, newTitle: string) => void;
  setActiveConversationId: (id: string | null) => void;
} {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [storageUnavailable, setStorageUnavailable] = useState(false);
  const [staleId, setStaleId] = useState(false);
  const [currentActiveId, setCurrentActiveId] = useState<string | null>(activeConversationId);
  // Ref so updater functions stay pure (no state setter calls inside setConversations)
  const storageUnavailableRef = useRef(false);
  const initializedRef = useRef(false);

  // Sync internal active id whenever the URL-derived prop changes (e.g. "New chat" → null)
  useEffect(() => {
    setCurrentActiveId(activeConversationId);
    setStaleId(false);
  }, [activeConversationId]);

  useEffect(() => {
    const { conversations: loaded, storageUnavailable: unavailable } = loadFromStorage();
    storageUnavailableRef.current = unavailable;
    setConversations(loaded);
    setStorageUnavailable(unavailable);
    initializedRef.current = true;

    if (activeConversationId !== null) {
      const found = loaded.some((c) => c.id === activeConversationId);
      if (!found) {
        setStaleId(true);
      }
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist to localStorage after every conversations change (pure effect, no setState inside updaters)
  useEffect(() => {
    if (!initializedRef.current) return;
    if (storageUnavailableRef.current) return;
    const success = saveToStorage(conversations);
    if (!success) {
      storageUnavailableRef.current = true;
      setStorageUnavailable(true);
    }
  }, [conversations]);

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
      setConversations((prev) => trimConversations([newConversation, ...prev]));
      setCurrentActiveId(id);
      return id;
    },
    [],
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
    },
    [],
  );

  const updateLastAssistantMessage = useCallback(
    (
      conversationId: string,
      updates: Partial<Pick<ChatMessage, "content" | "reasoningContent">>,
    ) => {
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

  const truncateAfterMessage = useCallback(
    (conversationId: string, messageId: string) => {
      setConversations((prev) => {
        const updated = prev.map((conv) => {
          if (conv.id !== conversationId) return conv;
          const idx = conv.messages.findIndex((m) => m.id === messageId);
          if (idx === -1) return conv;
          return { ...conv, messages: conv.messages.slice(0, idx), updatedAt: Date.now() };
        });
        return trimConversations(updated);
      });
    },
    [],
  );

  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => trimConversations(prev.filter((c) => c.id !== id)));
      if (currentActiveId === id) setCurrentActiveId(null);
    },
    [currentActiveId],
  );

  const renameConversation = useCallback(
    (id: string, newTitle: string) => {
      setConversations((prev) =>
        trimConversations(
          prev.map((conv) =>
            conv.id === id ? { ...conv, title: newTitle, updatedAt: Date.now() } : conv,
          ),
        ),
      );
    },
    [],
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
    truncateAfterMessage,
    deleteConversation,
    renameConversation,
    setActiveConversationId,
  };
}
