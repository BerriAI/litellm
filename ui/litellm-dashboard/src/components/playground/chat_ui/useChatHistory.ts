import React, { useState, useEffect } from "react";
import { MessageType, A2ATaskMetadata } from "./types";
import { TokenUsage } from "./ResponseMetrics";
import { MCPEvent } from "../../mcp_tools/types";
import { truncateString } from "../../../utils/textUtils";

export interface UseChatHistoryReturn {
  // State
  chatHistory: MessageType[];
  setChatHistory: React.Dispatch<React.SetStateAction<MessageType[]>>;
  mcpEvents: MCPEvent[];
  setMCPEvents: React.Dispatch<React.SetStateAction<MCPEvent[]>>;
  messageTraceId: string | null;
  setMessageTraceId: React.Dispatch<React.SetStateAction<string | null>>;
  responsesSessionId: string | null;
  setResponsesSessionId: React.Dispatch<React.SetStateAction<string | null>>;
  useApiSessionManagement: boolean;
  setUseApiSessionManagement: React.Dispatch<React.SetStateAction<boolean>>;

  // Actions
  updateTextUI: (role: string, chunk: string, model?: string) => void;
  updateReasoningContent: (chunk: string) => void;
  updateTimingData: (timeToFirstToken: number) => void;
  updateUsageData: (usage: TokenUsage, toolName?: string) => void;
  updateA2AMetadata: (a2aMetadata: A2ATaskMetadata) => void;
  updateTotalLatency: (totalLatency: number) => void;
  updateSearchResults: (searchResults: any[]) => void;
  handleResponseId: (responseId: string) => void;
  handleToggleSessionManagement: (useApi: boolean) => void;
  handleMCPEvent: (event: MCPEvent) => void;
  updateImageUI: (imageUrl: string, model: string) => void;
  updateEmbeddingsUI: (embeddings: string, model?: string) => void;
  updateAudioUI: (audioUrl: string, model: string) => void;
  updateChatImageUI: (imageUrl: string, model?: string) => void;
  clearChatHistory: () => void;
  clearMCPEvents: () => void;
}

export function useChatHistory({ simplified }: { simplified: boolean }): UseChatHistoryReturn {
  const [chatHistory, setChatHistory] = useState<MessageType[]>(() => {
    if (simplified) return [];
    try {
      const saved = sessionStorage.getItem("chatHistory");
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error("Error parsing chatHistory from sessionStorage", error);
      return [];
    }
  });

  const [mcpEvents, setMCPEvents] = useState<MCPEvent[]>([]);

  const [messageTraceId, setMessageTraceId] = useState<string | null>(
    () => sessionStorage.getItem("messageTraceId") || null,
  );

  const [responsesSessionId, setResponsesSessionId] = useState<string | null>(
    () => sessionStorage.getItem("responsesSessionId") || null,
  );

  const [useApiSessionManagement, setUseApiSessionManagement] = useState<boolean>(() => {
    const saved = sessionStorage.getItem("useApiSessionManagement");
    return saved ? JSON.parse(saved) : true; // Default to API session management
  });

  // Debounced chatHistory persistence
  useEffect(() => {
    if (simplified) return; // Do not persist chat history in simplified (embedded) mode
    const handler = setTimeout(() => {
      sessionStorage.setItem("chatHistory", JSON.stringify(chatHistory));
    }, 500); // Debounce by 500ms

    return () => {
      clearTimeout(handler);
    };
  }, [chatHistory, simplified]);

  // messageTraceId/responsesSessionId/useApiSessionManagement persistence
  useEffect(() => {
    if (messageTraceId) {
      sessionStorage.setItem("messageTraceId", messageTraceId);
    } else {
      sessionStorage.removeItem("messageTraceId");
    }
    if (responsesSessionId) {
      sessionStorage.setItem("responsesSessionId", responsesSessionId);
    } else {
      sessionStorage.removeItem("responsesSessionId");
    }
    sessionStorage.setItem("useApiSessionManagement", JSON.stringify(useApiSessionManagement));
  }, [messageTraceId, responsesSessionId, useApiSessionManagement]);

  const updateTextUI = (role: string, chunk: string, model?: string) => {
    setChatHistory((prev) => {
      const last = prev[prev.length - 1];
      // if the last message is already from this same role, append
      if (last && last.role === role && !last.isImage && !last.isAudio) {
        // build a new object, but only set `model` if it wasn't there already
        const updated: MessageType = {
          ...last,
          content: last.content + chunk,
          model: last.model ?? model, // ← only use the passed‐in model on the first chunk
        };
        return [...prev.slice(0, -1), updated];
      } else {
        // otherwise start a brand new assistant bubble
        return [
          ...prev,
          {
            role,
            content: chunk,
            model, // model set exactly once here
          },
        ];
      }
    });
  };

  const updateReasoningContent = (chunk: string) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];

      if (lastMessage && lastMessage.role === "assistant" && !lastMessage.isImage && !lastMessage.isAudio) {
        return [
          ...prevHistory.slice(0, prevHistory.length - 1),
          {
            ...lastMessage,
            reasoningContent: (lastMessage.reasoningContent || "") + chunk,
          },
        ];
      } else {
        // If there's no assistant message yet, we'll create one with empty content
        // but with reasoning content
        if (prevHistory.length > 0 && prevHistory[prevHistory.length - 1].role === "user") {
          return [
            ...prevHistory,
            {
              role: "assistant",
              content: "",
              reasoningContent: chunk,
            },
          ];
        }

        return prevHistory;
      }
    });
  };

  const updateTimingData = (timeToFirstToken: number) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];

      if (lastMessage && lastMessage.role === "assistant") {
        return [
          ...prevHistory.slice(0, prevHistory.length - 1),
          {
            ...lastMessage,
            timeToFirstToken,
          },
        ];
      }
      // If the last message is a user message and no assistant message exists yet,
      // create a new assistant message with empty content
      else if (lastMessage && lastMessage.role === "user") {
        return [
          ...prevHistory,
          {
            role: "assistant",
            content: "",
            timeToFirstToken,
          },
        ];
      }

      return prevHistory;
    });
  };

  const updateUsageData = (usage: TokenUsage, toolName?: string) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];

      if (lastMessage && lastMessage.role === "assistant") {
        const updatedMessage = {
          ...lastMessage,
          usage,
          toolName,
        };

        return [...prevHistory.slice(0, prevHistory.length - 1), updatedMessage];
      }

      return prevHistory;
    });
  };

  const updateA2AMetadata = (a2aMetadata: A2ATaskMetadata) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];

      if (lastMessage && lastMessage.role === "assistant") {
        const updatedMessage = {
          ...lastMessage,
          a2aMetadata,
        };
        return [...prevHistory.slice(0, prevHistory.length - 1), updatedMessage];
      }

      return prevHistory;
    });
  };

  const updateTotalLatency = (totalLatency: number) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];

      if (lastMessage && lastMessage.role === "assistant") {
        return [
          ...prevHistory.slice(0, prevHistory.length - 1),
          {
            ...lastMessage,
            totalLatency,
          },
        ];
      }

      return prevHistory;
    });
  };

  const updateSearchResults = (searchResults: any[]) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];

      if (lastMessage && lastMessage.role === "assistant") {
        const updatedMessage = {
          ...lastMessage,
          searchResults,
        };

        return [...prevHistory.slice(0, prevHistory.length - 1), updatedMessage];
      }

      return prevHistory;
    });
  };

  const handleResponseId = (responseId: string) => {
    if (useApiSessionManagement) {
      setResponsesSessionId(responseId);
    }
  };

  const handleToggleSessionManagement = (useApi: boolean) => {
    setUseApiSessionManagement(useApi);
    if (!useApi) {
      // Clear API session when switching to UI mode
      setResponsesSessionId(null);
    }
  };

  const handleMCPEvent = (event: MCPEvent) => {
    setMCPEvents((prev) => {
      // Check if this is a duplicate event (same item_id and type)
      // Only check for duplicates if item_id is defined (for mcp_list_tools, item_id is "mcp_list_tools")
      const isDuplicate = event.item_id
        ? prev.some(
            (existingEvent) =>
              existingEvent.item_id === event.item_id &&
              existingEvent.type === event.type &&
              (existingEvent.sequence_number === event.sequence_number ||
                (existingEvent.sequence_number === undefined && event.sequence_number === undefined)),
          )
        : false;

      if (isDuplicate) {
        return prev;
      }

      return [...prev, event];
    });
  };

  const updateImageUI = (imageUrl: string, model: string) => {
    setChatHistory((prevHistory) => [...prevHistory, { role: "assistant", content: imageUrl, model, isImage: true }]);
  };

  const updateEmbeddingsUI = (embeddings: string, model?: string) => {
    setChatHistory((prevHistory) => [
      ...prevHistory,
      { role: "assistant", content: truncateString(embeddings, 100), model, isEmbeddings: true },
    ]);
  };

  const updateAudioUI = (audioUrl: string, model: string) => {
    setChatHistory((prevHistory) => [...prevHistory, { role: "assistant", content: audioUrl, model, isAudio: true }]);
  };

  const updateChatImageUI = (imageUrl: string, model?: string) => {
    setChatHistory((prev) => {
      const last = prev[prev.length - 1];
      // If the last message is from assistant and has content, add image to it
      if (last && last.role === "assistant" && !last.isImage && !last.isAudio) {
        const updated = {
          ...last,
          image: {
            url: imageUrl,
            detail: "auto",
          },
          model: last.model ?? model,
        };
        return [...prev.slice(0, -1), updated];
      } else {
        // Otherwise create a new assistant message with just the image
        return [
          ...prev,
          {
            role: "assistant",
            content: "",
            model,
            image: {
              url: imageUrl,
              detail: "auto",
            },
          },
        ];
      }
    });
  };

  const clearChatHistory = () => {
    // Clean up audio object URLs before clearing history
    chatHistory.forEach((message) => {
      if (message.isAudio && typeof message.content === "string") {
        URL.revokeObjectURL(message.content);
      }
    });

    setChatHistory([]);
    setMessageTraceId(null);
    setResponsesSessionId(null); // Clear responses session ID
    setMCPEvents([]); // Clear MCP events
    if (!simplified) {
      sessionStorage.removeItem("chatHistory");
      sessionStorage.removeItem("messageTraceId");
      sessionStorage.removeItem("responsesSessionId");
    }
  };

  const clearMCPEvents = () => {
    setMCPEvents([]);
  };

  return {
    chatHistory,
    setChatHistory,
    mcpEvents,
    setMCPEvents,
    messageTraceId,
    setMessageTraceId,
    responsesSessionId,
    setResponsesSessionId,
    useApiSessionManagement,
    setUseApiSessionManagement,
    updateTextUI,
    updateReasoningContent,
    updateTimingData,
    updateUsageData,
    updateA2AMetadata,
    updateTotalLatency,
    updateSearchResults,
    handleResponseId,
    handleToggleSessionManagement,
    handleMCPEvent,
    updateImageUI,
    updateEmbeddingsUI,
    updateAudioUI,
    updateChatImageUI,
    clearChatHistory,
    clearMCPEvents,
  };
}
