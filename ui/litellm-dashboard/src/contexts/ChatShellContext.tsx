"use client";

import React, { createContext, useContext, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useChatHistory } from "@/components/chat/useChatHistory";
import type { ChatMessage, Conversation } from "@/components/chat/types";

interface ChatShellContextValue {
  accessToken: string;
  userId: string;
  userEmail: string;
  userRole: string;
  premiumUser: boolean;
  selectedMCPServers: string[];
  setSelectedMCPServers: (servers: string[]) => void;
  conversations: Conversation[];
  activeConversation: Conversation | null;
  activeConversationId: string | null;
  storageUnavailable: boolean;
  staleId: boolean;
  createConversation: (model: string) => string;
  appendMessage: (conversationId: string, message: Omit<ChatMessage, "id" | "timestamp">) => void;
  updateLastAssistantMessage: (
    conversationId: string,
    updates: Partial<Pick<ChatMessage, "content" | "reasoningContent" | "mcpEvents">>,
  ) => void;
  truncateFromMessage: (conversationId: string, messageId: string) => void;
  deleteConversation: (id: string) => void;
  renameConversation: (id: string, newTitle: string) => void;
}

const ChatShellContext = createContext<ChatShellContextValue | null>(null);

export function useChatShell(): ChatShellContextValue {
  const ctx = useContext(ChatShellContext);
  if (!ctx) {
    throw new Error("useChatShell must be used within a ChatShellProvider");
  }
  return ctx;
}

interface ChatShellProviderProps {
  accessToken: string;
  userId: string;
  userEmail: string;
  userRole: string;
  premiumUser: boolean;
  children: React.ReactNode;
}

export function ChatShellProvider({
  accessToken,
  userId,
  userEmail,
  userRole,
  premiumUser,
  children,
}: ChatShellProviderProps) {
  const searchParams = useSearchParams();
  const activeConversationId = searchParams.get("id");
  const [selectedMCPServers, setSelectedMCPServers] = useState<string[]>([]);

  const {
    conversations,
    activeConversation,
    storageUnavailable,
    staleId,
    createConversation,
    appendMessage,
    updateLastAssistantMessage,
    truncateFromMessage,
    deleteConversation,
    renameConversation,
  } = useChatHistory(activeConversationId, userId);

  return (
    <ChatShellContext.Provider
      value={{
        accessToken,
        userId,
        userEmail,
        userRole,
        premiumUser,
        selectedMCPServers,
        setSelectedMCPServers,
        conversations,
        activeConversation,
        activeConversationId,
        storageUnavailable,
        staleId,
        createConversation,
        appendMessage,
        updateLastAssistantMessage,
        truncateFromMessage,
        deleteConversation,
        renameConversation,
      }}
    >
      {children}
    </ChatShellContext.Provider>
  );
}
