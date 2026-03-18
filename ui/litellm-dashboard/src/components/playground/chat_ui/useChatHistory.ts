import { useEffect, useRef, useState } from "react";
import { truncateString } from "../../../utils/textUtils";
import type { MCPEvent } from "../../mcp_tools/types";
import NotificationsManager from "../../molecules/notifications_manager";
import { TokenUsage } from "./ResponseMetrics";
import { A2ATaskMetadata, MessageType } from "./types";

interface UseChatHistoryParams {
  simplified: boolean;
  /** Called during clearChatHistory to also clear file upload state */
  onClearUploads?: () => void;
}

interface UseChatHistoryReturn {
  chatHistory: MessageType[];
  setChatHistory: React.Dispatch<React.SetStateAction<MessageType[]>>;
  inputMessage: string;
  setInputMessage: (msg: string) => void;
  messageTraceId: string | null;
  setMessageTraceId: (id: string | null) => void;
  responsesSessionId: string | null;
  useApiSessionManagement: boolean;
  mcpEvents: MCPEvent[];
  chatEndRef: React.RefObject<HTMLDivElement | null>;
  // Updaters
  updateTextUI: (role: string, chunk: string, model?: string) => void;
  updateReasoningContent: (chunk: string) => void;
  updateTimingData: (timeToFirstToken: number) => void;
  updateUsageData: (usage: TokenUsage, toolName?: string) => void;
  updateA2AMetadata: (metadata: A2ATaskMetadata) => void;
  updateTotalLatency: (latency: number) => void;
  updateSearchResults: (searchResults: unknown[]) => void;
  updateImageUI: (url: string, model: string) => void;
  updateEmbeddingsUI: (embeddings: string, model?: string) => void;
  updateAudioUI: (url: string, model: string) => void;
  updateChatImageUI: (url: string, model?: string) => void;
  handleResponseId: (responseId: string) => void;
  handleToggleSessionManagement: (useApi: boolean) => void;
  handleMCPEvent: (event: MCPEvent) => void;
  clearMCPEvents: () => void;
  clearChatHistory: () => void;
}

function useChatHistory({ simplified, onClearUploads }: UseChatHistoryParams): UseChatHistoryReturn {
  const [inputMessage, setInputMessage] = useState("");
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
  const [messageTraceId, setMessageTraceId] = useState<string | null>(
    () => sessionStorage.getItem("messageTraceId") || null,
  );
  const [responsesSessionId, setResponsesSessionId] = useState<string | null>(
    () => sessionStorage.getItem("responsesSessionId") || null,
  );
  const [useApiSessionManagement, setUseApiSessionManagement] = useState<boolean>(() => {
    const saved = sessionStorage.getItem("useApiSessionManagement");
    return saved ? JSON.parse(saved) : true;
  });
  const [mcpEvents, setMCPEvents] = useState<MCPEvent[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Debounced persistence of chatHistory to sessionStorage
  useEffect(() => {
    if (simplified) return;
    const handler = setTimeout(() => {
      sessionStorage.setItem("chatHistory", JSON.stringify(chatHistory));
    }, 500);
    return () => clearTimeout(handler);
  }, [chatHistory, simplified]);

  // Persist session management state
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

  // Auto-scroll on chat update
  useEffect(() => {
    if (chatEndRef.current) {
      setTimeout(() => {
        chatEndRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "end",
        });
      }, 100);
    }
  }, [chatHistory]);

  // --- Updater functions ---

  const updateTextUI = (role: string, chunk: string, model?: string) => {
    console.log("updateTextUI called with:", role, chunk, model);
    setChatHistory((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === role && !last.isImage && !last.isAudio) {
        const updated: MessageType = {
          ...last,
          content: last.content + chunk,
          model: last.model ?? model,
        };
        return [...prev.slice(0, -1), updated];
      } else {
        return [...prev, { role, content: chunk, model }];
      }
    });
  };

  const updateReasoningContent = (chunk: string) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];
      if (lastMessage && lastMessage.role === "assistant" && !lastMessage.isImage && !lastMessage.isAudio) {
        return [
          ...prevHistory.slice(0, -1),
          { ...lastMessage, reasoningContent: (lastMessage.reasoningContent || "") + chunk },
        ];
      } else if (prevHistory.length > 0 && prevHistory[prevHistory.length - 1].role === "user") {
        return [...prevHistory, { role: "assistant", content: "", reasoningContent: chunk }];
      }
      return prevHistory;
    });
  };

  const updateTimingData = (timeToFirstToken: number) => {
    console.log("updateTimingData called with:", timeToFirstToken);
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];
      if (lastMessage && lastMessage.role === "assistant") {
        return [...prevHistory.slice(0, -1), { ...lastMessage, timeToFirstToken }];
      } else if (lastMessage && lastMessage.role === "user") {
        return [...prevHistory, { role: "assistant", content: "", timeToFirstToken }];
      }
      return prevHistory;
    });
  };

  const updateUsageData = (usage: TokenUsage, toolName?: string) => {
    console.log("Received usage data:", usage);
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];
      if (lastMessage && lastMessage.role === "assistant") {
        return [...prevHistory.slice(0, -1), { ...lastMessage, usage, toolName }];
      }
      return prevHistory;
    });
  };

  const updateA2AMetadata = (a2aMetadata: A2ATaskMetadata) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];
      if (lastMessage && lastMessage.role === "assistant") {
        return [...prevHistory.slice(0, -1), { ...lastMessage, a2aMetadata }];
      }
      return prevHistory;
    });
  };

  const updateTotalLatency = (totalLatency: number) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];
      if (lastMessage && lastMessage.role === "assistant") {
        return [...prevHistory.slice(0, -1), { ...lastMessage, totalLatency }];
      }
      return prevHistory;
    });
  };

  const updateSearchResults = (searchResults: unknown[]) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];
      if (lastMessage && lastMessage.role === "assistant") {
        return [...prevHistory.slice(0, -1), { ...lastMessage, searchResults }];
      }
      return prevHistory;
    });
  };

  const updateImageUI = (imageUrl: string, model: string) => {
    setChatHistory((prev) => [...prev, { role: "assistant", content: imageUrl, model, isImage: true }]);
  };

  const updateEmbeddingsUI = (embeddings: string, model?: string) => {
    setChatHistory((prev) => [
      ...prev,
      { role: "assistant", content: truncateString(embeddings, 100), model, isEmbeddings: true },
    ]);
  };

  const updateAudioUI = (audioUrl: string, model: string) => {
    setChatHistory((prev) => [...prev, { role: "assistant", content: audioUrl, model, isAudio: true }]);
  };

  const updateChatImageUI = (imageUrl: string, model?: string) => {
    setChatHistory((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "assistant" && !last.isImage && !last.isAudio) {
        return [
          ...prev.slice(0, -1),
          { ...last, image: { url: imageUrl, detail: "auto" }, model: last.model ?? model },
        ];
      } else {
        return [
          ...prev,
          { role: "assistant", content: "", model, image: { url: imageUrl, detail: "auto" } },
        ];
      }
    });
  };

  const handleResponseId = (responseId: string) => {
    console.log("Received response ID for session management:", responseId);
    if (useApiSessionManagement) {
      setResponsesSessionId(responseId);
    }
  };

  const handleToggleSessionManagement = (useApi: boolean) => {
    setUseApiSessionManagement(useApi);
    if (!useApi) {
      setResponsesSessionId(null);
    }
  };

  const handleMCPEvent = (event: MCPEvent) => {
    console.log("ChatUI: Received MCP event:", event);
    setMCPEvents((prev) => {
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
        console.log("ChatUI: Duplicate MCP event, skipping");
        return prev;
      }

      return [...prev, event];
    });
  };

  const clearMCPEvents = () => setMCPEvents([]);

  const clearChatHistory = () => {
    // Revoke URLs to prevent memory leaks
    chatHistory.forEach((message) => {
      if (message.isImage && typeof message.content === "string" && message.content.startsWith("blob:")) {
        URL.revokeObjectURL(message.content);
      }
      if (message.isAudio && typeof message.content === "string") {
        URL.revokeObjectURL(message.content);
      }
    });

    setChatHistory([]);
    setMessageTraceId(null);
    setResponsesSessionId(null);
    setMCPEvents([]);
    onClearUploads?.();
    if (!simplified) {
      sessionStorage.removeItem("chatHistory");
      sessionStorage.removeItem("messageTraceId");
      sessionStorage.removeItem("responsesSessionId");
    }
    NotificationsManager.success("Chat history cleared.");
  };

  return {
    chatHistory,
    setChatHistory,
    inputMessage,
    setInputMessage,
    messageTraceId,
    setMessageTraceId,
    responsesSessionId,
    useApiSessionManagement,
    mcpEvents,
    chatEndRef,
    updateTextUI,
    updateReasoningContent,
    updateTimingData,
    updateUsageData,
    updateA2AMetadata,
    updateTotalLatency,
    updateSearchResults,
    updateImageUI,
    updateEmbeddingsUI,
    updateAudioUI,
    updateChatImageUI,
    handleResponseId,
    handleToggleSessionManagement,
    handleMCPEvent,
    clearMCPEvents,
    clearChatHistory,
  };
}

export default useChatHistory;
