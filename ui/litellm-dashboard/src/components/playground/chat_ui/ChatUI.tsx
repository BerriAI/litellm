"use client";

import {
  ApiOutlined,
  ArrowUpOutlined,
  ClearOutlined,
  CodeOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  FilePdfOutlined,
  InfoCircleOutlined,
  KeyOutlined,
  LinkOutlined,
  LoadingOutlined,
  PictureOutlined,
  RobotOutlined,
  SafetyOutlined,
  SettingOutlined,
  SoundOutlined,
  TagsOutlined,
  ToolOutlined,
  UserOutlined,
} from "@ant-design/icons";
import { Card, Text, TextInput, Title, Button as TremorButton } from "@tremor/react";
import { Button, Input, Modal, Popover, Select, Spin, Tooltip, Typography, Upload } from "antd";
import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import { v4 as uuidv4 } from "uuid";
import { truncateString } from "../../../utils/textUtils";
import GuardrailSelector from "../../guardrails/GuardrailSelector";
import PolicySelector from "../../policies/PolicySelector";
import MCPToolArgumentsForm, { MCPToolArgumentsFormRef } from "../../mcp_tools/MCPToolArgumentsForm";
import { MCPServer } from "../../mcp_tools/types";
import NotificationsManager from "../../molecules/notifications_manager";
import { callMCPTool, fetchMCPServers, listMCPTools } from "../../networking";
import TagSelector from "../../tag_management/TagSelector";
import VectorStoreSelector from "../../vector_store_management/VectorStoreSelector";
import { makeA2ASendMessageRequest } from "../llm_calls/a2a_send_message";
import { makeAnthropicMessagesRequest } from "../llm_calls/anthropic_messages";
import { makeOpenAIAudioSpeechRequest } from "../llm_calls/audio_speech";
import { makeOpenAIAudioTranscriptionRequest } from "../llm_calls/audio_transcriptions";
import { makeOpenAIChatCompletionRequest } from "../llm_calls/chat_completion";
import { makeOpenAIEmbeddingsRequest } from "../llm_calls/embeddings_api";
import { Agent, fetchAvailableAgents } from "../llm_calls/fetch_agents";
import { fetchAvailableModels, ModelGroup } from "../llm_calls/fetch_models";
import { makeOpenAIImageEditsRequest } from "../llm_calls/image_edits";
import { makeOpenAIImageGenerationRequest } from "../llm_calls/image_generation";
import { makeOpenAIResponsesRequest } from "../llm_calls/responses_api";
import A2AMetrics from "./A2AMetrics";
import AdditionalModelSettings from "./AdditionalModelSettings";
import AudioRenderer from "./AudioRenderer";
import { OPEN_AI_VOICE_SELECT_OPTIONS, OpenAIVoice } from "./chatConstants";
import ChatImageRenderer from "./ChatImageRenderer";
import ChatImageUpload from "./ChatImageUpload";
import { createChatDisplayMessage, createChatMultimodalMessage } from "./ChatImageUtils";
import CodeInterpreterOutput from "./CodeInterpreterOutput";
import CodeInterpreterTool from "./CodeInterpreterTool";
import { generateCodeSnippet } from "./CodeSnippets";
import EndpointSelector from "./EndpointSelector";
import MCPEventsDisplay, { MCPEvent } from "./MCPEventsDisplay";
import { EndpointType, getEndpointType } from "./mode_endpoint_mapping";
import ReasoningContent from "./ReasoningContent";
import ResponseMetrics, { TokenUsage } from "./ResponseMetrics";
import ResponsesImageRenderer from "./ResponsesImageRenderer";
import ResponsesImageUpload from "./ResponsesImageUpload";
import { createDisplayMessage, createMultimodalMessage } from "./ResponsesImageUtils";
import { SearchResultsDisplay } from "./SearchResultsDisplay";
import SessionManagement from "./SessionManagement";
import { A2ATaskMetadata, MessageType } from "./types";
import { useCodeInterpreter } from "./useCodeInterpreter";

const { TextArea } = Input;
const { Dragger } = Upload;

interface ChatUIProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  disabledPersonalKeyCreation: boolean;
  proxySettings?: {
    PROXY_BASE_URL?: string;
    LITELLM_UI_API_DOC_BASE_URL?: string | null;
  };
}

const MCP_SUPPORTED_ENDPOINTS = new Set<EndpointType>([
  EndpointType.CHAT,
  EndpointType.RESPONSES,
  EndpointType.MCP,
]);

const ChatUI: React.FC<ChatUIProps> = ({
  accessToken,
  token,
  userRole,
  userID,
  disabledPersonalKeyCreation,
  proxySettings,
}) => {
  const [mcpServers, setMCPServers] = useState<MCPServer[]>([]);
  const [selectedMCPServers, setSelectedMCPServers] = useState<string[]>(() => {
    const saved = sessionStorage.getItem("selectedMCPServers");
    try {
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error("Error parsing selectedMCPServers from sessionStorage", error);
      return [];
    }
  });
  const [isLoadingMCPServers, setIsLoadingMCPServers] = useState(false);
  const [serverToolsMap, setServerToolsMap] = useState<Record<string, any[]>>({});
  const [selectedMCPDirectTool, setSelectedMCPDirectTool] = useState<string | undefined>(undefined);
  const mcpToolArgsFormRef = useRef<MCPToolArgumentsFormRef>(null);
  const [mcpServerToolRestrictions, setMCPServerToolRestrictions] = useState<Record<string, string[]>>(() => {
    const saved = sessionStorage.getItem("mcpServerToolRestrictions");
    try {
      return saved ? JSON.parse(saved) : {};
    } catch (error) {
      console.error("Error parsing mcpServerToolRestrictions from sessionStorage", error);
      return {};
    }
  });
  const [apiKeySource, setApiKeySource] = useState<"session" | "custom">(() => {
    const saved = sessionStorage.getItem("apiKeySource");
    if (saved) {
      try {
        return JSON.parse(saved) as "session" | "custom";
      } catch (error) {
        console.error("Error parsing apiKeySource from sessionStorage", error);
      }
    }
    return disabledPersonalKeyCreation ? "custom" : "session";
  });
  const [apiKey, setApiKey] = useState<string>(() => sessionStorage.getItem("apiKey") || "");
  const [customProxyBaseUrl, setCustomProxyBaseUrl] = useState<string>(
    () => sessionStorage.getItem("customProxyBaseUrl") || "",
  );
  const [inputMessage, setInputMessage] = useState("");
  const [chatHistory, setChatHistory] = useState<MessageType[]>(() => {
    try {
      const saved = sessionStorage.getItem("chatHistory");
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error("Error parsing chatHistory from sessionStorage", error);
      return [];
    }
  });
  const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined);
  const [showCustomModelInput, setShowCustomModelInput] = useState<boolean>(false);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [agentInfo, setAgentInfo] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string | undefined>(undefined);
  const customModelTimeout = useRef<NodeJS.Timeout | null>(null);
  const [endpointType, setEndpointType] = useState<string>(
    () => sessionStorage.getItem("endpointType") || EndpointType.CHAT,
  );
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [selectedTags, setSelectedTags] = useState<string[]>(() => {
    const saved = sessionStorage.getItem("selectedTags");
    try {
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error("Error parsing selectedTags from sessionStorage", error);
      return [];
    }
  });
  const [selectedVoice, setSelectedVoice] = useState<OpenAIVoice>(() => {
    const saved = sessionStorage.getItem("selectedVoice");
    if (!saved) return "alloy";
    try {
      return JSON.parse(saved) as OpenAIVoice;
    } catch {
      // If stored value is not valid JSON, treat it as a plain string
      return saved as OpenAIVoice;
    }
  });
  const [selectedVectorStores, setSelectedVectorStores] = useState<string[]>(() => {
    const saved = sessionStorage.getItem("selectedVectorStores");
    try {
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error("Error parsing selectedVectorStores from sessionStorage", error);
      return [];
    }
  });
  const [selectedGuardrails, setSelectedGuardrails] = useState<string[]>(() => {
    const saved = sessionStorage.getItem("selectedGuardrails");
    try {
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error("Error parsing selectedGuardrails from sessionStorage", error);
      return [];
    }
  });
  const [selectedPolicies, setSelectedPolicies] = useState<string[]>(() => {
    const saved = sessionStorage.getItem("selectedPolicies");
    try {
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error("Error parsing selectedPolicies from sessionStorage", error);
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
    return saved ? JSON.parse(saved) : true; // Default to API session management
  });
  const [uploadedImages, setUploadedImages] = useState<File[]>([]);
  const [imagePreviewUrls, setImagePreviewUrls] = useState<string[]>([]);
  const [responsesUploadedImage, setResponsesUploadedImage] = useState<File | null>(null);
  const [responsesImagePreviewUrl, setResponsesImagePreviewUrl] = useState<string | null>(null);
  const [chatUploadedImage, setChatUploadedImage] = useState<File | null>(null);
  const [chatImagePreviewUrl, setChatImagePreviewUrl] = useState<string | null>(null);
  const [uploadedAudio, setUploadedAudio] = useState<File | null>(null);
  const [isGetCodeModalVisible, setIsGetCodeModalVisible] = useState(false);
  const [generatedCode, setGeneratedCode] = useState("");
  const [selectedSdk, setSelectedSdk] = useState<"openai" | "azure">("openai");
  const [mcpEvents, setMCPEvents] = useState<MCPEvent[]>([]);
  const [temperature, setTemperature] = useState<number>(1.0);
  const [maxTokens, setMaxTokens] = useState<number>(2048);
  const [useAdvancedParams, setUseAdvancedParams] = useState<boolean>(false);
  const [mockTestFallbacks, setMockTestFallbacks] = useState<boolean>(false);

  // Code Interpreter state (using custom hook)
  const codeInterpreter = useCodeInterpreter();

  const chatEndRef = useRef<HTMLDivElement>(null);

  // Fetch MCP servers
  const loadMCPServers = async () => {
    const userApiKey = apiKeySource === "session" ? accessToken : apiKey;
    if (!userApiKey) return;

    setIsLoadingMCPServers(true);
    try {
      const servers = await fetchMCPServers(userApiKey);
      setMCPServers(Array.isArray(servers) ? servers : servers.data || []);
    } catch (error) {
      console.error("Error fetching MCP servers:", error);
    } finally {
      setIsLoadingMCPServers(false);
    }
  };

  // Fetch tools for a specific server
  const loadServerTools = async (serverId: string) => {
    const userApiKey = apiKeySource === "session" ? accessToken : apiKey;
    if (!userApiKey || serverToolsMap[serverId]) return;

    try {
      const response = await listMCPTools(userApiKey, serverId);
      setServerToolsMap((prev) => ({
        ...prev,
        [serverId]: response.tools || [],
      }));
    } catch (error) {
      console.error(`Error fetching tools for server ${serverId}:`, error);
    }
  };

  useEffect(() => {
    if (isGetCodeModalVisible) {
      const code = generateCodeSnippet({
        apiKeySource,
        accessToken,
        apiKey,
        inputMessage,
        chatHistory,
        selectedTags,
        selectedVectorStores,
        selectedGuardrails,
        selectedPolicies,
        selectedMCPServers,
        mcpServers,
        mcpServerToolRestrictions,
        endpointType,
        selectedModel,
        selectedSdk,
        selectedVoice,
        proxySettings,
      });
      setGeneratedCode(code);
    }
  }, [
    isGetCodeModalVisible,
    selectedSdk,
    apiKeySource,
    accessToken,
    apiKey,
    inputMessage,
    chatHistory,
    selectedTags,
    selectedVectorStores,
    selectedGuardrails,
    selectedPolicies,
    selectedMCPServers,
    mcpServers,
    mcpServerToolRestrictions,
    endpointType,
    selectedModel,
    proxySettings,
  ]);

  useEffect(() => {
    const handler = setTimeout(() => {
      sessionStorage.setItem("chatHistory", JSON.stringify(chatHistory));
    }, 500); // Debounce by 500ms

    return () => {
      clearTimeout(handler);
    };
  }, [chatHistory]);

  useEffect(() => {
    sessionStorage.setItem("apiKeySource", JSON.stringify(apiKeySource));
    sessionStorage.setItem("apiKey", apiKey);
    sessionStorage.setItem("endpointType", endpointType);
    sessionStorage.setItem("selectedTags", JSON.stringify(selectedTags));
    sessionStorage.setItem("selectedVectorStores", JSON.stringify(selectedVectorStores));
    sessionStorage.setItem("selectedGuardrails", JSON.stringify(selectedGuardrails));
    sessionStorage.setItem("selectedPolicies", JSON.stringify(selectedPolicies));
    sessionStorage.setItem("selectedMCPServers", JSON.stringify(selectedMCPServers));
    sessionStorage.setItem("mcpServerToolRestrictions", JSON.stringify(mcpServerToolRestrictions));
    sessionStorage.setItem("selectedVoice", selectedVoice);
    sessionStorage.removeItem("selectedMCPTools"); // Clean up old key

    if (selectedModel) {
      sessionStorage.setItem("selectedModel", selectedModel);
    } else {
      sessionStorage.removeItem("selectedModel");
    }
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
    // Note: codeInterpreterEnabled and selectedContainerId are persisted by useCodeInterpreter hook
  }, [
    apiKeySource,
    apiKey,
    selectedModel,
    endpointType,
    selectedTags,
    selectedVectorStores,
    selectedGuardrails,
    selectedPolicies,
    messageTraceId,
    responsesSessionId,
    useApiSessionManagement,
    selectedMCPServers,
    mcpServerToolRestrictions,
    selectedVoice,
  ]);

  useEffect(() => {
    let userApiKey = apiKeySource === "session" ? accessToken : apiKey;
    if (!userApiKey || !token || !userRole || !userID) {
      console.log("userApiKey or token or userRole or userID is missing = ", userApiKey, token, userRole, userID);
      return;
    }

    // Fetch model info and set the default selected model
    const loadModels = async () => {
      try {
        if (!userApiKey) {
          console.log("userApiKey is missing");
          return;
        }
        const uniqueModels = await fetchAvailableModels(userApiKey);

        console.log("Fetched models:", uniqueModels);

        setModelInfo(uniqueModels);

        // check for selection overlap or empty model list
        const hasSelection = uniqueModels.some((m) => m.model_group === selectedModel);
        if (!uniqueModels.length) {
          setSelectedModel(undefined);
        } else if (!hasSelection) {
          setSelectedModel(undefined);
        }
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };

    loadModels();
    loadMCPServers();
  }, [accessToken, userID, userRole, apiKeySource, apiKey, token]);

  // Load tools when MCP direct mode has a server selected
  useEffect(() => {
    if (
      endpointType === EndpointType.MCP &&
      selectedMCPServers.length === 1 &&
      selectedMCPServers[0] !== "__all__" &&
      !serverToolsMap[selectedMCPServers[0]]
    ) {
      loadServerTools(selectedMCPServers[0]);
    }
  }, [endpointType, selectedMCPServers, serverToolsMap]);

  // Fetch agents when A2A endpoint is selected
  useEffect(() => {
    const userApiKey = apiKeySource === "session" ? accessToken : apiKey;
    if (!userApiKey || endpointType !== EndpointType.A2A_AGENTS) {
      return;
    }

    const loadAgents = async () => {
      try {
        const agents = await fetchAvailableAgents(userApiKey, customProxyBaseUrl || undefined);
        setAgentInfo(agents);
        // Clear selection if current agent not in list
        if (selectedAgent && !agents.some((a) => a.agent_name === selectedAgent)) {
          setSelectedAgent(undefined);
        }
      } catch (error) {
        console.error("Error fetching agents:", error);
      }
    };

    loadAgents();
  }, [accessToken, apiKeySource, apiKey, endpointType, customProxyBaseUrl, selectedAgent]);

  useEffect(() => {
    // Scroll to the bottom of the chat whenever chatHistory updates
    if (chatEndRef.current) {
      // Add a small delay to ensure content is rendered
      setTimeout(() => {
        chatEndRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "end", // Keep the scroll position at the end
        });
      }, 100);
    }
  }, [chatHistory]);

  const updateTextUI = (role: string, chunk: string, model?: string) => {
    console.log("updateTextUI called with:", role, chunk, model);
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
    console.log("updateTimingData called with:", timeToFirstToken);
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];
      console.log("Current last message:", lastMessage);

      if (lastMessage && lastMessage.role === "assistant") {
        console.log("Updating assistant message with timeToFirstToken:", timeToFirstToken);
        const updatedHistory = [
          ...prevHistory.slice(0, prevHistory.length - 1),
          {
            ...lastMessage,
            timeToFirstToken,
          },
        ];
        console.log("Updated chat history:", updatedHistory);
        return updatedHistory;
      }
      // If the last message is a user message and no assistant message exists yet,
      // create a new assistant message with empty content
      else if (lastMessage && lastMessage.role === "user") {
        console.log("Creating new assistant message with timeToFirstToken:", timeToFirstToken);
        return [
          ...prevHistory,
          {
            role: "assistant",
            content: "",
            timeToFirstToken,
          },
        ];
      }

      console.log("No appropriate message found to update timing");
      return prevHistory;
    });
  };

  const updateUsageData = (usage: TokenUsage, toolName?: string) => {
    console.log("Received usage data:", usage);
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];

      if (lastMessage && lastMessage.role === "assistant") {
        console.log("Updating message with usage data:", usage);
        const updatedMessage = {
          ...lastMessage,
          usage,
          toolName,
        };
        console.log("Updated message:", updatedMessage);

        return [...prevHistory.slice(0, prevHistory.length - 1), updatedMessage];
      }

      return prevHistory;
    });
  };

  const updateA2AMetadata = (a2aMetadata: A2ATaskMetadata) => {
    console.log("Received A2A metadata:", a2aMetadata);
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
    console.log("Received search results:", searchResults);
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];

      if (lastMessage && lastMessage.role === "assistant") {
        console.log("Updating message with search results");
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
    console.log("Received response ID for session management:", responseId);
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
    console.log("ChatUI: Received MCP event:", event);
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
        console.log("ChatUI: Duplicate MCP event, skipping");
        return prev;
      }

      const newEvents = [...prev, event];
      console.log("ChatUI: Updated MCP events:", newEvents);
      return newEvents;
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

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault(); // Prevent default to avoid newline
      handleSendMessage();
    }
    // If Shift+Enter is pressed, the default behavior (inserting a newline) will occur
  };

  const handleCancelRequest = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsLoading(false);
      NotificationsManager.info("Request cancelled");
    }
  };

  const handleImageUpload = (file: File) => {
    setUploadedImages((prev) => [...prev, file]);
    const previewUrl = URL.createObjectURL(file);
    setImagePreviewUrls((prev) => [...prev, previewUrl]);
    return false; // Prevent default upload behavior
  };

  const handleRemoveImage = (index: number) => {
    if (imagePreviewUrls[index]) {
      URL.revokeObjectURL(imagePreviewUrls[index]);
    }
    setUploadedImages((prev) => prev.filter((_, i) => i !== index));
    setImagePreviewUrls((prev) => prev.filter((_, i) => i !== index));
  };

  const handleRemoveAllImages = () => {
    imagePreviewUrls.forEach((url) => {
      URL.revokeObjectURL(url);
    });
    setUploadedImages([]);
    setImagePreviewUrls([]);
  };

  const handleResponsesImageUpload = (file: File): false => {
    setResponsesUploadedImage(file);
    const previewUrl = URL.createObjectURL(file);
    setResponsesImagePreviewUrl(previewUrl);
    return false; // Prevent default upload behavior
  };

  const handleRemoveResponsesImage = () => {
    if (responsesImagePreviewUrl) {
      URL.revokeObjectURL(responsesImagePreviewUrl);
    }
    setResponsesUploadedImage(null);
    setResponsesImagePreviewUrl(null);
  };

  const handleChatImageUpload = (file: File): false => {
    setChatUploadedImage(file);
    const previewUrl = URL.createObjectURL(file);
    setChatImagePreviewUrl(previewUrl);
    return false; // Prevent default upload behavior
  };

  const handleRemoveChatImage = () => {
    if (chatImagePreviewUrl) {
      URL.revokeObjectURL(chatImagePreviewUrl);
    }
    setChatUploadedImage(null);
    setChatImagePreviewUrl(null);
  };

  const handleAudioUpload = (file: File): false => {
    setUploadedAudio(file);
    return false; // Prevent default upload behavior
  };

  const handleRemoveAudio = () => {
    setUploadedAudio(null);
  };

    const handleSendMessage = async () => {
    if (
      inputMessage.trim() === "" &&
      endpointType !== EndpointType.TRANSCRIPTION &&
      endpointType !== EndpointType.MCP
    )
      return;

    // For image edits, require both image and prompt
    if (endpointType === EndpointType.IMAGE_EDITS && uploadedImages.length === 0) {
      NotificationsManager.fromBackend("Please upload at least one image for editing");
      return;
    }

    // For audio transcriptions, require audio file
    if (endpointType === EndpointType.TRANSCRIPTION && !uploadedAudio) {
      NotificationsManager.fromBackend("Please upload an audio file for transcription");
      return;
    }

    // For A2A agents, require agent selection
    if (endpointType === EndpointType.A2A_AGENTS && !selectedAgent) {
      NotificationsManager.fromBackend("Please select an agent to send a message");
      return;
    }

    // For MCP direct mode, require server and tool selection, and get form values early
    let mcpToolArguments: Record<string, any> = {};
    if (endpointType === EndpointType.MCP) {
      const mcpServerId =
        selectedMCPServers.length === 1 && selectedMCPServers[0] !== "__all__"
          ? selectedMCPServers[0]
          : null;
      if (!mcpServerId) {
        NotificationsManager.fromBackend("Please select an MCP server to test");
        return;
      }
      if (!selectedMCPDirectTool) {
        NotificationsManager.fromBackend("Please select an MCP tool to call");
        return;
      }
      const mcpTool = (serverToolsMap[selectedMCPServers[0]] || []).find(
        (t: any) => t.name === selectedMCPDirectTool,
      );
      if (!mcpTool) {
        NotificationsManager.fromBackend("Please wait for tool schema to load");
        return;
      }
      try {
        mcpToolArguments = (await mcpToolArgsFormRef.current?.getSubmitValues()) ?? {};
      } catch (err) {
        NotificationsManager.fromBackend(
          err instanceof Error ? err.message : "Please fill in all required parameters",
        );
        return;
      }
    }

    // Require model selection for all model-based endpoints (MCP direct mode does not need a model)
    const modelRequiredEndpoints = [
      EndpointType.CHAT,
      EndpointType.IMAGE,
      EndpointType.SPEECH,
      EndpointType.IMAGE_EDITS,
      EndpointType.RESPONSES,
      EndpointType.ANTHROPIC_MESSAGES,
      EndpointType.EMBEDDINGS,
      EndpointType.TRANSCRIPTION,
    ];

    if (modelRequiredEndpoints.includes(endpointType as EndpointType) && !selectedModel) {
      NotificationsManager.fromBackend("Please select a model before sending a request");
      return;
    }

    if (!token || !userRole || !userID) {
      return;
    }

    const effectiveApiKey = apiKeySource === "session" ? accessToken : apiKey;

    if (!effectiveApiKey) {
      NotificationsManager.fromBackend("Please provide a Virtual Key or select Current UI Session");
      return;
    }

    // Create new abort controller for this request
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    // Create message object without model field for API call
    let newUserMessage: { role: string; content: string | any[] };

    // Handle image for responses API
    if (endpointType === EndpointType.RESPONSES && responsesUploadedImage) {
      try {
        newUserMessage = await createMultimodalMessage(inputMessage, responsesUploadedImage);
      } catch (error) {
        NotificationsManager.fromBackend("Failed to process image. Please try again.");
        return;
      }
    }
    // Handle image for chat completions API
    else if (endpointType === EndpointType.CHAT && chatUploadedImage) {
      try {
        newUserMessage = await createChatMultimodalMessage(inputMessage, chatUploadedImage);
      } catch (error) {
        NotificationsManager.fromBackend("Failed to process image. Please try again.");
        return;
      }
    } else {
      newUserMessage = { role: "user", content: inputMessage };
    }

    // Generate new trace ID for a new conversation or use existing one
    const traceId = messageTraceId || uuidv4();
    if (!messageTraceId) {
      setMessageTraceId(traceId);
    }

    // Update UI with full message object (always display as text for UI)
    let displayMessage: MessageType;
    if (endpointType === EndpointType.RESPONSES && responsesUploadedImage) {
      displayMessage = createDisplayMessage(
        inputMessage,
        true,
        responsesImagePreviewUrl || undefined,
        responsesUploadedImage.name,
      );
    } else if (endpointType === EndpointType.CHAT && chatUploadedImage) {
      displayMessage = createChatDisplayMessage(
        inputMessage,
        true,
        chatImagePreviewUrl || undefined,
        chatUploadedImage.name,
      );
    } else if (endpointType === EndpointType.TRANSCRIPTION && uploadedAudio) {
      // For audio transcription, show the audio file name and optional prompt
      const audioMessage = inputMessage
        ? `🎵 Audio file: ${uploadedAudio.name}\nPrompt: ${inputMessage}`
        : `🎵 Audio file: ${uploadedAudio.name}`;
      displayMessage = createDisplayMessage(audioMessage, false);
    } else if (endpointType === EndpointType.MCP && selectedMCPDirectTool) {
      // For MCP direct mode, show tool name and arguments from form
      const mcpMessage = `🔧 MCP Tool: ${selectedMCPDirectTool}\nArguments: ${JSON.stringify(mcpToolArguments, null, 2)}`;
      displayMessage = createDisplayMessage(mcpMessage, false);
    } else {
      displayMessage = createDisplayMessage(inputMessage, false);
    }

    setChatHistory([...chatHistory, displayMessage]);
    setMCPEvents([]); // Clear previous MCP events for new conversation turn
    codeInterpreter.clearResult(); // Clear previous code interpreter results
    setIsLoading(true);

    try {
      if (selectedModel) {
        if (endpointType === EndpointType.CHAT) {
          // Create chat history for API call - strip out model field and isImage field
          // For chat completions, we preserve the multimodal content structure
          const apiChatHistory = [
            ...chatHistory
              .filter((msg) => !msg.isImage && !msg.isAudio)
              .map(({ role, content }) => ({
                role,
                content: typeof content === "string" ? content : "",
              })),
            newUserMessage,
          ];

          await makeOpenAIChatCompletionRequest(
            apiChatHistory,
            (chunk, model) => updateTextUI("assistant", chunk, model),
            selectedModel,
            effectiveApiKey,
            selectedTags,
            signal,
            updateReasoningContent,
            updateTimingData,
            updateUsageData,
            traceId,
            selectedVectorStores.length > 0 ? selectedVectorStores : undefined,
            selectedGuardrails.length > 0 ? selectedGuardrails : undefined,
            selectedPolicies.length > 0 ? selectedPolicies : undefined,
            selectedMCPServers,
            updateChatImageUI,
            updateSearchResults,
            useAdvancedParams ? temperature : undefined,
            useAdvancedParams ? maxTokens : undefined,
            updateTotalLatency,
            customProxyBaseUrl || undefined,
            mcpServers,
            mcpServerToolRestrictions,
            handleMCPEvent,
            mockTestFallbacks,
          );
        } else if (endpointType === EndpointType.IMAGE) {
          // For image generation
          await makeOpenAIImageGenerationRequest(
            inputMessage,
            (imageUrl, model) => updateImageUI(imageUrl, model),
            selectedModel,
            effectiveApiKey,
            selectedTags,
            signal,
            customProxyBaseUrl || undefined,
          );
        } else if (endpointType === EndpointType.SPEECH) {
          // For audio speech
          await makeOpenAIAudioSpeechRequest(
            inputMessage,
            selectedVoice,
            (audioUrl, model) => updateAudioUI(audioUrl, model),
            selectedModel || "",
            effectiveApiKey,
            selectedTags,
            signal,
            undefined, // responseFormat
            undefined, // speed
            customProxyBaseUrl || undefined,
          );
        } else if (endpointType === EndpointType.IMAGE_EDITS) {
          // For image edits
          if (uploadedImages.length > 0) {
            await makeOpenAIImageEditsRequest(
              uploadedImages.length === 1 ? uploadedImages[0] : uploadedImages,
              inputMessage,
              (imageUrl, model) => updateImageUI(imageUrl, model),
              selectedModel,
              effectiveApiKey,
              selectedTags,
              signal,
              customProxyBaseUrl || undefined,
            );
          }
        } else if (endpointType === EndpointType.RESPONSES) {
          // Create chat history for API call - strip out model field and isImage field
          let apiChatHistory;

          if (useApiSessionManagement && responsesSessionId) {
            // When using API session management with existing session, only send the new message
            apiChatHistory = [newUserMessage];
          } else {
            // When using UI session management or starting new API session, send full history
            apiChatHistory = [
              ...chatHistory
                .filter((msg) => !msg.isImage && !msg.isAudio)
                .map(({ role, content }) => ({ role, content })),
              newUserMessage,
            ];
          }

          await makeOpenAIResponsesRequest(
            apiChatHistory,
            (role, delta, model) => updateTextUI(role, delta, model),
            selectedModel,
            effectiveApiKey,
            selectedTags,
            signal,
            updateReasoningContent,
            updateTimingData,
            updateUsageData,
            traceId,
            selectedVectorStores.length > 0 ? selectedVectorStores : undefined,
            selectedGuardrails.length > 0 ? selectedGuardrails : undefined,
            selectedPolicies.length > 0 ? selectedPolicies : undefined,
            selectedMCPServers, // Pass the selected servers array
            useApiSessionManagement ? responsesSessionId : null, // Only pass session ID if API mode is enabled
            handleResponseId, // Pass callback to capture new response ID
            handleMCPEvent, // Pass MCP event handler
            codeInterpreter.enabled, // Enable Code Interpreter tool
            codeInterpreter.setResult, // Handle code interpreter output
            customProxyBaseUrl || undefined,
            mcpServers,
            mcpServerToolRestrictions,
          );
        } else if (endpointType === EndpointType.ANTHROPIC_MESSAGES) {
          const apiChatHistory = [
            ...chatHistory
              .filter((msg) => !msg.isImage && !msg.isAudio)
              .map(({ role, content }) => ({ role, content })),
            newUserMessage,
          ];

          await makeAnthropicMessagesRequest(
            apiChatHistory,
            (role, delta, model) => updateTextUI(role, delta, model),
            selectedModel,
            effectiveApiKey,
            selectedTags,
            signal,
            updateReasoningContent,
            updateTimingData,
            updateUsageData,
            traceId,
            selectedVectorStores.length > 0 ? selectedVectorStores : undefined,
            selectedGuardrails.length > 0 ? selectedGuardrails : undefined,
            selectedPolicies.length > 0 ? selectedPolicies : undefined,
            selectedMCPServers, // Pass the selected tools array
            customProxyBaseUrl || undefined,
          );
        } else if (endpointType === EndpointType.EMBEDDINGS) {
          await makeOpenAIEmbeddingsRequest(
            inputMessage,
            (embeddings, model) => updateEmbeddingsUI(embeddings, model),
            selectedModel,
            effectiveApiKey,
            selectedTags,
            customProxyBaseUrl || undefined,
          );
        } else if (endpointType === EndpointType.TRANSCRIPTION) {
          // For audio transcriptions
          if (uploadedAudio) {
            await makeOpenAIAudioTranscriptionRequest(
              uploadedAudio,
              (transcription, model) => updateTextUI("assistant", transcription, model),
              selectedModel,
              effectiveApiKey,
              selectedTags,
              signal,
              undefined, // language
              undefined, // prompt
              undefined, // responseFormat
              undefined, // temperature
              customProxyBaseUrl || undefined,
            );
          }
        }
      }

      // Handle MCP direct tool calls (no chat completions)
      if (endpointType === EndpointType.MCP) {
        const mcpServerId =
          selectedMCPServers.length === 1 && selectedMCPServers[0] !== "__all__"
            ? selectedMCPServers[0]
            : null;
        if (mcpServerId && selectedMCPDirectTool) {
          const result = await callMCPTool(
            effectiveApiKey,
            mcpServerId,
            selectedMCPDirectTool,
            mcpToolArguments,
            selectedGuardrails.length > 0 ? { guardrails: selectedGuardrails } : undefined,
          );
          const resultText =
            result?.content?.length > 0
              ? JSON.stringify(
                  result.content.map((c: any) => (c.type === "text" ? c.text : c)).filter(Boolean),
                  null,
                  2,
                )
              : JSON.stringify(result, null, 2);
          updateTextUI("assistant", resultText || "Tool executed successfully.");
        }
      }

      // Handle A2A agent calls (separate from model-based calls) - use streaming
      if (endpointType === EndpointType.A2A_AGENTS && selectedAgent) {
        await makeA2ASendMessageRequest(
          selectedAgent,
          inputMessage,
          (chunk, model) => updateTextUI("assistant", chunk, model),
          effectiveApiKey,
          signal,
          updateTimingData,
          updateTotalLatency,
          updateA2AMetadata,
          customProxyBaseUrl || undefined,
          selectedGuardrails.length > 0 ? selectedGuardrails : undefined,
        );
      }
    } catch (error) {
      if (signal.aborted) {
        console.log("Request was cancelled");
      } else {
        console.error("Error fetching response", error);
        updateTextUI("assistant", "Error fetching response:" + error);
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
      // Clear image after successful request for image edits
      if (endpointType === EndpointType.IMAGE_EDITS) {
        handleRemoveAllImages();
      }
      // Clear image after successful request for responses API
      if (endpointType === EndpointType.RESPONSES && responsesUploadedImage) {
        handleRemoveResponsesImage();
      }
      // Clear image after successful request for chat completions API
      if (endpointType === EndpointType.CHAT && chatUploadedImage) {
        handleRemoveChatImage();
      }
      // Clear audio after successful request for transcription
      if (endpointType === EndpointType.TRANSCRIPTION && uploadedAudio) {
        handleRemoveAudio();
      }
    }

    setInputMessage("");
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
    handleRemoveAllImages(); // Clear any uploaded images for image edits
    handleRemoveResponsesImage(); // Clear any uploaded images for responses
    handleRemoveChatImage(); // Clear any uploaded images for chat completions
    handleRemoveAudio(); // Clear any uploaded audio for transcription
    sessionStorage.removeItem("chatHistory");
    sessionStorage.removeItem("messageTraceId");
    sessionStorage.removeItem("responsesSessionId");
    NotificationsManager.success("Chat history cleared.");
  };

  if (userRole && userRole === "Admin Viewer") {
    const { Title, Paragraph } = Typography;
    return (
      <div>
        <Title level={1}>Access Denied</Title>
        <Paragraph>Ask your proxy admin for access to test models</Paragraph>
      </div>
    );
  }

  const onModelChange = (value: string) => {
    console.log(`selected ${value}`);
    setSelectedModel(value);

    setShowCustomModelInput(value === "custom");
  };

  // Check if the selected model is a chat model
  const isChatModel = () => {
    if (!selectedModel || selectedModel === "custom") {
      return false;
    }
    const model = modelInfo.find((m) => m.model_group === selectedModel);
    if (!model) {
      return false;
    }
    // Check if mode is explicitly "chat" or undefined (which defaults to chat per backend)
    return !model.mode || model.mode === "chat";
  };

  const antIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;

  return (
    <div className="w-full p-4 pb-0 bg-white">
      <Card className="w-full rounded-xl shadow-md overflow-hidden">
        <div className="flex h-[80vh] w-full gap-4">
          {/* Left Sidebar with Controls */}
          <div className="w-1/4 p-4 bg-gray-50 overflow-y-auto">
            <Title className="text-xl font-semibold mb-6 mt-2">Configurations</Title>
            <div className="space-y-4">
              <div>
                <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                  <KeyOutlined className="mr-2" /> Virtual Key Source
                </Text>
                <Select
                  disabled={disabledPersonalKeyCreation}
                  value={apiKeySource}
                  style={{ width: "100%" }}
                  onChange={(value) => {
                    setApiKeySource(value as "session" | "custom");
                  }}
                  options={[
                    { value: "session", label: "Current UI Session" },
                    { value: "custom", label: "Virtual Key" },
                  ]}
                  className="rounded-md"
                />
                {apiKeySource === "custom" && (
                  <TextInput
                    className="mt-2"
                    placeholder="Enter custom Virtual Key"
                    type="password"
                    onValueChange={setApiKey}
                    value={apiKey}
                    icon={KeyOutlined}
                  />
                )}
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <Text className="font-medium block text-gray-700 flex items-center">
                    <SettingOutlined className="mr-2" /> Custom Proxy Base URL
                  </Text>
                  {proxySettings?.LITELLM_UI_API_DOC_BASE_URL && !customProxyBaseUrl && (
                    <Button
                      type="link"
                      size="small"
                      icon={<LinkOutlined />}
                      onClick={() => {
                        setCustomProxyBaseUrl(proxySettings.LITELLM_UI_API_DOC_BASE_URL || "");
                        sessionStorage.setItem("customProxyBaseUrl", proxySettings.LITELLM_UI_API_DOC_BASE_URL || "");
                      }}
                      className="text-gray-500 hover:text-gray-700"
                    >
                      Fill
                    </Button>
                  )}
                  {customProxyBaseUrl && (
                    <Button
                      type="link"
                      size="small"
                      icon={<ClearOutlined />}
                      onClick={() => {
                        setCustomProxyBaseUrl("");
                        sessionStorage.removeItem("customProxyBaseUrl");
                      }}
                      className="text-gray-500 hover:text-gray-700"
                    >
                      Clear
                    </Button>
                  )}
                </div>
                <TextInput
                  placeholder="Optional: Enter custom proxy URL (e.g., http://localhost:5000)"
                  onValueChange={(value) => {
                    setCustomProxyBaseUrl(value);
                    sessionStorage.setItem("customProxyBaseUrl", value);
                  }}
                  value={customProxyBaseUrl}
                  icon={ApiOutlined}
                />
                {customProxyBaseUrl && (
                  <Text className="text-xs text-gray-500 mt-1">API calls will be sent to: {customProxyBaseUrl}</Text>
                )}
              </div>

              <div>
                <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                  <ApiOutlined className="mr-2" /> Endpoint Type
                </Text>
                <EndpointSelector
                  endpointType={endpointType}
                  onEndpointChange={(value) => {
                    setEndpointType(value);
                    // Clear model/agent selection when switching endpoint type
                    setSelectedModel(undefined);
                    setSelectedAgent(undefined);
                    setShowCustomModelInput(false);
                    setSelectedMCPDirectTool(undefined);
                    // For MCP direct mode, require single server (clear __all__ or multiple)
                    if (value === EndpointType.MCP) {
                      setSelectedMCPServers((prev) =>
                        prev.length === 1 && prev[0] !== "__all__" ? prev : [],
                      );
                    }
                    try {
                      sessionStorage.removeItem("selectedModel");
                      sessionStorage.removeItem("selectedAgent");
                    } catch {}
                  }}
                  className="mb-4"
                />

                {/* Voice Selector for Speech Endpoint */}
                {endpointType === EndpointType.SPEECH && (
                  <div className="mb-4">
                    <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                      <SoundOutlined className="mr-2" />
                      Voice
                    </Text>
                    <Select
                      value={selectedVoice}
                      onChange={(value) => {
                        setSelectedVoice(value);
                        sessionStorage.setItem("selectedVoice", value);
                      }}
                      style={{ width: "100%" }}
                      className="rounded-md"
                      options={OPEN_AI_VOICE_SELECT_OPTIONS}
                    />
                  </div>
                )}

                {/* Session Management Component */}
                <SessionManagement
                  endpointType={endpointType}
                  responsesSessionId={responsesSessionId}
                  useApiSessionManagement={useApiSessionManagement}
                  onToggleSessionManagement={handleToggleSessionManagement}
                />
              </div>

              {/* Model Selector - shown when NOT using A2A Agents or MCP direct mode */}
              {endpointType !== EndpointType.A2A_AGENTS && endpointType !== EndpointType.MCP && (
                <div>
                  <Text className="font-medium block mb-2 text-gray-700 flex items-center justify-between">
                    <span className="flex items-center">
                      <RobotOutlined className="mr-2" /> Select Model
                    </span>
                    {isChatModel() ? (
                      <Popover
                        content={
                          <AdditionalModelSettings
                            temperature={temperature}
                            maxTokens={maxTokens}
                            useAdvancedParams={useAdvancedParams}
                            onTemperatureChange={setTemperature}
                            onMaxTokensChange={setMaxTokens}
                            onUseAdvancedParamsChange={setUseAdvancedParams}
                            mockTestFallbacks={mockTestFallbacks}
                            onMockTestFallbacksChange={setMockTestFallbacks}
                          />
                        }
                        title="Model Settings"
                        trigger="click"
                        placement="right"
                      >
                        <Button
                          type="text"
                          size="small"
                          icon={<SettingOutlined />}
                          className="text-gray-500 hover:text-gray-700"
                          aria-label="Model Settings"
                          data-testid="model-settings-button"
                        />
                      </Popover>
                    ) : (
                      <Tooltip title="Advanced parameters are only supported for chat models currently">
                        <Button
                          type="text"
                          size="small"
                          icon={<SettingOutlined />}
                          className="text-gray-300 cursor-not-allowed"
                          disabled
                        />
                      </Tooltip>
                    )}
                  </Text>
                  <Select
                    value={selectedModel}
                    placeholder="Select a Model"
                    onChange={onModelChange}
                    options={[
                      { value: "custom", label: "Enter custom model", key: "custom" },
                      ...Array.from(
                        new Set(
                          modelInfo
                            .filter((option) => {
                              if (!option.mode) {
                                //If no mode, show all models
                                return true;
                              }
                              const optionEndpoint = getEndpointType(option.mode);
                              // Show chat models for responses/anthropic_messages endpoints as they are compatible
                              if (
                                endpointType === EndpointType.RESPONSES ||
                                endpointType === EndpointType.ANTHROPIC_MESSAGES
                              ) {
                                return optionEndpoint === endpointType || optionEndpoint === EndpointType.CHAT;
                              }
                              // Show image models for image_edits endpoint as they are compatible
                              if (endpointType === EndpointType.IMAGE_EDITS) {
                                return optionEndpoint === endpointType || optionEndpoint === EndpointType.IMAGE;
                              }
                              return optionEndpoint === endpointType;
                            })
                            .map((option) => option.model_group),
                        ),
                      ).map((model_group, index) => ({
                        value: model_group,
                        label: model_group,
                        key: index,
                      })),
                    ]}
                    style={{ width: "100%" }}
                    showSearch={true}
                    className="rounded-md"
                  />
                  {showCustomModelInput && (
                    <TextInput
                      className="mt-2"
                      placeholder="Enter custom model name"
                      onValueChange={(value) => {
                        // Using setTimeout to create a simple debounce effect
                        if (customModelTimeout.current) {
                          clearTimeout(customModelTimeout.current);
                        }

                        customModelTimeout.current = setTimeout(() => {
                          setSelectedModel(value);
                        }, 500); // 500ms delay after typing stops
                      }}
                    />
                  )}
                </div>
              )}

              {/* Agent Selector - shown ONLY for A2A Agents endpoint */}
              {endpointType === EndpointType.A2A_AGENTS && (
                <div>
                  <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                    <RobotOutlined className="mr-2" /> Select Agent
                  </Text>
                  <Select
                    value={selectedAgent}
                    placeholder="Select an Agent"
                    onChange={(value) => setSelectedAgent(value)}
                    options={agentInfo.map((agent) => ({
                      value: agent.agent_name,
                      label: agent.agent_name || agent.agent_id,
                      key: agent.agent_id,
                    }))}
                    style={{ width: "100%" }}
                    showSearch={true}
                    className="rounded-md"
                    optionLabelProp="label"
                  >
                    {agentInfo.map((agent) => (
                      <Select.Option
                        key={agent.agent_id}
                        value={agent.agent_name}
                        label={agent.agent_name || agent.agent_id}
                      >
                        <div className="flex flex-col py-1">
                          <span className="font-medium">{agent.agent_name || agent.agent_id}</span>
                          {agent.agent_card_params?.description && (
                            <span className="text-xs text-gray-500 mt-1">{agent.agent_card_params.description}</span>
                          )}
                        </div>
                      </Select.Option>
                    ))}
                  </Select>
                  {agentInfo.length === 0 && (
                    <Text className="text-xs text-gray-500 mt-2 block">
                      No agents found. Create agents via /v1/agents endpoint.
                    </Text>
                  )}
                </div>
              )}

              <div>
                <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                  <TagsOutlined className="mr-2" /> Tags
                </Text>
                <TagSelector
                  value={selectedTags}
                  onChange={setSelectedTags}
                  className="mb-4"
                  accessToken={accessToken || ""}
                />
              </div>

              {/* MCP Server Selection */}
              <div>
                <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                  <ToolOutlined className="mr-2" />
                  {endpointType === EndpointType.MCP ? "MCP Server" : "MCP Servers"}
                  <Tooltip
                    className="ml-1"
                    title={
                      endpointType === EndpointType.MCP
                        ? "Select an MCP server to test tools directly."
                        : "Select MCP servers to use in your conversation."
                    }
                  >
                    <InfoCircleOutlined />
                  </Tooltip>
                </Text>
                <Select
                  mode={endpointType === EndpointType.MCP ? undefined : "multiple"}
                  style={{ width: "100%" }}
                  placeholder={
                    endpointType === EndpointType.MCP ? "Select MCP server" : "Select MCP servers"
                  }
                  value={
                    endpointType === EndpointType.MCP
                      ? selectedMCPServers[0] !== "__all__" && selectedMCPServers.length === 1
                        ? selectedMCPServers[0]
                        : undefined
                      : selectedMCPServers
                  }
                  onChange={(value) => {
                    if (endpointType === EndpointType.MCP) {
                      const serverId = value as string | undefined;
                      setSelectedMCPServers(serverId ? [serverId] : []);
                      setSelectedMCPDirectTool(undefined);
                      if (serverId && !serverToolsMap[serverId]) {
                        loadServerTools(serverId);
                      }
                    } else {
                      if ((value as string[]).includes("__all__")) {
                        setSelectedMCPServers(["__all__"]);
                        setMCPServerToolRestrictions({});
                      } else {
                        setSelectedMCPServers(value as string[]);
                        setMCPServerToolRestrictions((prev) => {
                          const updated = { ...prev };
                          Object.keys(updated).forEach((serverId) => {
                            if (!(value as string[]).includes(serverId)) delete updated[serverId];
                          });
                          return updated;
                        });
                        (value as string[]).forEach((serverId) => {
                          if (!serverToolsMap[serverId]) {
                            loadServerTools(serverId);
                          }
                        });
                      }
                    }
                  }}
                  loading={isLoadingMCPServers}
                  className="mb-2"
                  allowClear
                  optionLabelProp="label"
                  disabled={!MCP_SUPPORTED_ENDPOINTS.has(endpointType as EndpointType)}
                  maxTagCount={endpointType === EndpointType.MCP ? 1 : "responsive"}
                >
                  {/* All MCP Servers option - hidden for MCP direct mode */}
                  {endpointType !== EndpointType.MCP && (
                    <Select.Option key="__all__" value="__all__" label="All MCP Servers">
                      <div className="flex flex-col py-1">
                        <span className="font-medium">All MCP Servers</span>
                        <span className="text-xs text-gray-500 mt-1">Use all available MCP servers</span>
                      </div>
                    </Select.Option>
                  )}

                  {/* Individual servers */}
                  {mcpServers.map((server) => (
                    <Select.Option
                      key={server.server_id}
                      value={server.server_id}
                      label={server.alias || server.server_name || server.server_id}
                      disabled={
                        endpointType === EndpointType.MCP ? false : selectedMCPServers.includes("__all__")
                      }
                    >
                      <div className="flex flex-col py-1">
                        <span className="font-medium">{server.alias || server.server_name || server.server_id}</span>
                        {server.description && <span className="text-xs text-gray-500 mt-1">{server.description}</span>}
                      </div>
                    </Select.Option>
                  ))}
                </Select>

                {/* MCP Tool selector - only for MCP direct mode */}
                {endpointType === EndpointType.MCP &&
                  selectedMCPServers.length === 1 &&
                  selectedMCPServers[0] !== "__all__" && (
                    <div className="mt-3">
                      <Text className="text-xs text-gray-600 mb-1 block">Select Tool</Text>
                      <Select
                        style={{ width: "100%" }}
                        placeholder="Select a tool to call"
                        value={selectedMCPDirectTool}
                        onChange={(value) => setSelectedMCPDirectTool(value)}
                        options={(serverToolsMap[selectedMCPServers[0]] || []).map((tool: any) => ({
                          value: tool.name,
                          label: tool.name,
                        }))}
                        allowClear
                        className="rounded-md"
                      />
                    </div>
                  )}

                {/* Tool restrictions UI (optional) - hidden for MCP direct mode */}
                {selectedMCPServers.length > 0 &&
                  !selectedMCPServers.includes("__all__") &&
                  endpointType !== EndpointType.MCP &&
                  MCP_SUPPORTED_ENDPOINTS.has(endpointType as EndpointType) && (
                    <div className="mt-3 space-y-2">
                      {selectedMCPServers.map((serverId) => {
                        const server = mcpServers.find((s) => s.server_id === serverId);
                        const tools = serverToolsMap[serverId] || [];
                        if (tools.length === 0) return null;

                        return (
                          <div key={serverId} className="border rounded p-2">
                            <Text className="text-xs text-gray-600 mb-1">
                              Limit tools for {server?.alias || server?.server_name || serverId}:
                            </Text>
                            <Select
                              mode="multiple"
                              size="small"
                              style={{ width: "100%" }}
                              placeholder="All tools (default)"
                              value={mcpServerToolRestrictions[serverId] || []}
                              onChange={(selectedTools) => {
                                setMCPServerToolRestrictions((prev) => ({
                                  ...prev,
                                  [serverId]: selectedTools,
                                }));
                              }}
                              options={tools.map((tool) => ({
                                value: tool.name,
                                label: tool.name,
                              }))}
                              maxTagCount={2}
                            />
                          </div>
                        );
                      })}
                    </div>
                  )}
              </div>

              <div>
                <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                  <DatabaseOutlined className="mr-2" /> Vector Store
                  <Tooltip
                    className="ml-1"
                    title={
                      <span>
                        Select vector store(s) to use for this LLM API call. You can set up your vector store{" "}
                        <a href="?page=vector-stores" style={{ color: "#1890ff" }}>
                          here
                        </a>
                        .
                      </span>
                    }
                  >
                    <InfoCircleOutlined />
                  </Tooltip>
                </Text>
                <VectorStoreSelector
                  value={selectedVectorStores}
                  onChange={setSelectedVectorStores}
                  className="mb-4"
                  accessToken={accessToken || ""}
                />
              </div>

              <div>
                <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                  <SafetyOutlined className="mr-2" /> Guardrails
                  <Tooltip
                    className="ml-1"
                    title={
                      <span>
                        Select guardrail(s) to use for this LLM API call. You can set up your guardrails{" "}
                        <a href="?page=guardrails" style={{ color: "#1890ff" }}>
                          here
                        </a>
                        .
                      </span>
                    }
                  >
                    <InfoCircleOutlined />
                  </Tooltip>
                </Text>
                <GuardrailSelector
                  value={selectedGuardrails}
                  onChange={setSelectedGuardrails}
                  className="mb-4"
                  accessToken={accessToken || ""}
                />
              </div>

              <div>
                <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                  <SafetyOutlined className="mr-2" /> Policies
                  <Tooltip
                    className="ml-1"
                    title={
                      <span>
                        Select policy/policies to apply to this LLM API call. Policies define which guardrails are applied based on conditions. You can set up your policies{" "}
                        <a href="?page=policies" style={{ color: "#1890ff" }}>
                          here
                        </a>
                        .
                      </span>
                    }
                  >
                    <InfoCircleOutlined />
                  </Tooltip>
                </Text>
                <PolicySelector
                  value={selectedPolicies}
                  onChange={setSelectedPolicies}
                  className="mb-4"
                  accessToken={accessToken || ""}
                />
              </div>

              {/* Code Interpreter Toggle - Only for Responses endpoint */}
              {endpointType === EndpointType.RESPONSES && (
                <div>
                  <CodeInterpreterTool
                    accessToken={apiKeySource === "session" ? accessToken || "" : apiKey}
                    enabled={codeInterpreter.enabled}
                    onEnabledChange={codeInterpreter.setEnabled}
                    selectedContainerId={null}
                    onContainerChange={() => { }}
                    selectedModel={selectedModel || ""}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Main Chat Area */}
          <div className="w-3/4 flex flex-col bg-white">
            <div className="p-4 border-b border-gray-200 flex justify-between items-center">
              <Title className="text-xl font-semibold mb-0">Test Key</Title>
              <div className="flex gap-2">
                <TremorButton
                  onClick={clearChatHistory}
                  className="bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
                  icon={ClearOutlined}
                >
                  Clear Chat
                </TremorButton>
                <TremorButton
                  onClick={() => setIsGetCodeModalVisible(true)}
                  className="bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
                  icon={CodeOutlined}
                >
                  Get Code
                </TremorButton>
              </div>
            </div>
            <div className="flex-1 overflow-auto p-4 pb-0">
              {chatHistory.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-gray-400">
                  <RobotOutlined style={{ fontSize: "48px", marginBottom: "16px" }} />
                  <Text>Start a conversation, generate an image, or handle audio</Text>
                </div>
              )}

              {chatHistory.map((message, index) => (
                <div key={index}>
                  <div className={`mb-4 ${message.role === "user" ? "text-right" : "text-left"}`}>
                    <div
                      className="inline-block max-w-[80%] rounded-lg shadow-sm p-3.5 px-4"
                      style={{
                        backgroundColor: message.role === "user" ? "#f0f8ff" : "#ffffff",
                        border: message.role === "user" ? "1px solid #e6f0fa" : "1px solid #f0f0f0",
                        textAlign: "left",
                      }}
                    >
                      <div className="flex items-center gap-2 mb-1.5">
                        <div
                          className="flex items-center justify-center w-6 h-6 rounded-full mr-1"
                          style={{
                            backgroundColor: message.role === "user" ? "#e6f0fa" : "#f5f5f5",
                          }}
                        >
                          {message.role === "user" ? (
                            <UserOutlined style={{ fontSize: "12px", color: "#2563eb" }} />
                          ) : (
                            <RobotOutlined style={{ fontSize: "12px", color: "#4b5563" }} />
                          )}
                        </div>
                        <strong className="text-sm capitalize">{message.role}</strong>
                        {message.role === "assistant" && message.model && (
                          <span className="text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-600 font-normal">
                            {message.model}
                          </span>
                        )}
                      </div>
                      {message.reasoningContent && <ReasoningContent reasoningContent={message.reasoningContent} />}

                      {/* Show MCP events at the start of assistant messages */}
                      {message.role === "assistant" &&
                        index === chatHistory.length - 1 &&
                        mcpEvents.length > 0 &&
                        (endpointType === EndpointType.RESPONSES || endpointType === EndpointType.CHAT) && (
                          <div className="mb-3">
                            <MCPEventsDisplay events={mcpEvents} />
                          </div>
                        )}

                      {/* Show search results at the start of assistant messages */}
                      {message.role === "assistant" && message.searchResults && (
                        <SearchResultsDisplay searchResults={message.searchResults} />
                      )}

                      {/* Show Code Interpreter output for the last assistant message */}
                      {message.role === "assistant" &&
                        index === chatHistory.length - 1 &&
                        codeInterpreter.result &&
                        endpointType === EndpointType.RESPONSES && (
                          <CodeInterpreterOutput
                            code={codeInterpreter.result.code}
                            containerId={codeInterpreter.result.containerId}
                            annotations={codeInterpreter.result.annotations}
                            accessToken={apiKeySource === "session" ? accessToken || "" : apiKey}
                          />
                        )}

                      <div
                        className="whitespace-pre-wrap break-words max-w-full message-content"
                        style={{
                          wordWrap: "break-word",
                          overflowWrap: "break-word",
                          wordBreak: "break-word",
                          hyphens: "auto",
                        }}
                      >
                        {message.isImage ? (
                          <img
                            src={typeof message.content === "string" ? message.content : ""}
                            alt="Generated image"
                            className="max-w-full rounded-md border border-gray-200 shadow-sm"
                            style={{ maxHeight: "500px" }}
                          />
                        ) : message.isAudio ? (
                          <AudioRenderer message={message} />
                        ) : (
                          <>
                            {/* Show attached image for user messages based on current endpoint */}
                            {endpointType === EndpointType.RESPONSES && <ResponsesImageRenderer message={message} />}
                            {endpointType === EndpointType.CHAT && <ChatImageRenderer message={message} />}

                            <ReactMarkdown
                              components={{
                                code({
                                  node,
                                  inline,
                                  className,
                                  children,
                                  ...props
                                }: React.ComponentPropsWithoutRef<"code"> & {
                                  inline?: boolean;
                                  node?: any;
                                }) {
                                  const match = /language-(\w+)/.exec(className || "");
                                  return !inline && match ? (
                                    <SyntaxHighlighter
                                      style={coy as any}
                                      language={match[1]}
                                      PreTag="div"
                                      className="rounded-md my-2"
                                      wrapLines={true}
                                      wrapLongLines={true}
                                      {...props}
                                    >
                                      {String(children).replace(/\n$/, "")}
                                    </SyntaxHighlighter>
                                  ) : (
                                    <code
                                      className={`${className} px-1.5 py-0.5 rounded bg-gray-100 text-sm font-mono`}
                                      style={{ wordBreak: "break-word" }}
                                      {...props}
                                    >
                                      {children}
                                    </code>
                                  );
                                },
                                pre: ({ node, ...props }) => (
                                  <pre style={{ overflowX: "auto", maxWidth: "100%" }} {...props} />
                                ),
                              }}
                            >
                              {typeof message.content === "string" ? message.content : ""}
                            </ReactMarkdown>

                            {/* Show generated image from chat completions */}
                            {message.image && (
                              <div className="mt-3">
                                <img
                                  src={message.image.url}
                                  alt="Generated image"
                                  className="max-w-full rounded-md border border-gray-200 shadow-sm"
                                  style={{ maxHeight: "500px" }}
                                />
                              </div>
                            )}
                          </>
                        )}

                        {message.role === "assistant" &&
                          (message.timeToFirstToken || message.totalLatency || message.usage) &&
                          !message.a2aMetadata && (
                            <ResponseMetrics
                              timeToFirstToken={message.timeToFirstToken}
                              totalLatency={message.totalLatency}
                              usage={message.usage}
                              toolName={message.toolName}
                            />
                          )}

                        {/* A2A Metrics - show for A2A agent responses */}
                        {message.role === "assistant" && message.a2aMetadata && (
                          <A2AMetrics
                            a2aMetadata={message.a2aMetadata}
                            timeToFirstToken={message.timeToFirstToken}
                            totalLatency={message.totalLatency}
                          />
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {/* Show MCP events during loading if no assistant message exists yet */}
              {isLoading &&
                mcpEvents.length > 0 &&
                (endpointType === EndpointType.RESPONSES || endpointType === EndpointType.CHAT) &&
                chatHistory.length > 0 &&
                chatHistory[chatHistory.length - 1].role === "user" && (
                  <div className="text-left mb-4">
                    <div
                      className="inline-block max-w-[80%] rounded-lg shadow-sm p-3.5 px-4"
                      style={{
                        backgroundColor: "#ffffff",
                        border: "1px solid #f0f0f0",
                        textAlign: "left",
                      }}
                    >
                      <div className="flex items-center gap-2 mb-1.5">
                        <div
                          className="flex items-center justify-center w-6 h-6 rounded-full mr-1"
                          style={{
                            backgroundColor: "#f5f5f5",
                          }}
                        >
                          <RobotOutlined style={{ fontSize: "12px", color: "#4b5563" }} />
                        </div>
                        <strong className="text-sm capitalize">Assistant</strong>
                      </div>
                      <MCPEventsDisplay events={mcpEvents} />
                    </div>
                  </div>
                )}

              {isLoading && (
                <div className="flex justify-center items-center my-4">
                  <Spin indicator={antIcon} />
                </div>
              )}
              <div ref={chatEndRef} style={{ height: "1px" }} />
            </div>

            <div className="p-4 border-t border-gray-200 bg-white">
              {/* Image Upload Section for Image Edits */}
              {endpointType === EndpointType.IMAGE_EDITS && (
                <div className="mb-4">
                  {uploadedImages.length === 0 ? (
                    <Dragger beforeUpload={handleImageUpload} accept="image/*" showUploadList={false}>
                      <p className="ant-upload-drag-icon">
                        <PictureOutlined style={{ fontSize: "24px", color: "#666" }} />
                      </p>
                      <p className="ant-upload-text text-sm">Click or drag images to upload</p>
                      <p className="ant-upload-hint text-xs text-gray-500">
                        Support for PNG, JPG, JPEG formats. Multiple images supported.
                      </p>
                    </Dragger>
                  ) : (
                    <div className="flex flex-wrap gap-2">
                      {uploadedImages.map((file, index) => (
                        <div key={index} className="relative inline-block">
                          <img
                            src={imagePreviewUrls[index] || ""}
                            alt={`Upload preview ${index + 1}`}
                            className="max-w-32 max-h-32 rounded-md border border-gray-200 object-cover"
                          />
                          <button
                            className="absolute top-1 right-1 bg-white shadow-sm border border-gray-200 rounded px-1 py-1 text-red-500 hover:bg-red-50 text-xs"
                            onClick={() => handleRemoveImage(index)}
                          >
                            <DeleteOutlined />
                          </button>
                        </div>
                      ))}
                      {/* Add more images button */}
                      <div
                        className="flex items-center justify-center w-32 h-32 border-2 border-dashed border-gray-300 rounded-md hover:border-gray-400 cursor-pointer"
                        onClick={() => document.getElementById("additional-image-upload")?.click()}
                      >
                        <div className="text-center">
                          <PictureOutlined style={{ fontSize: "24px", color: "#666" }} />
                          <p className="text-xs text-gray-500 mt-1">Add more</p>
                        </div>
                        <input
                          id="additional-image-upload"
                          type="file"
                          accept="image/*"
                          multiple
                          style={{ display: "none" }}
                          onChange={(e) => {
                            const files = Array.from(e.target.files || []);
                            files.forEach((file) => handleImageUpload(file));
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Audio Upload Section for Transcriptions */}
              {endpointType === EndpointType.TRANSCRIPTION && (
                <div className="mb-4">
                  {!uploadedAudio ? (
                    <Dragger
                      beforeUpload={handleAudioUpload}
                      accept="audio/*,.mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm"
                      showUploadList={false}
                    >
                      <p className="ant-upload-drag-icon">
                        <SoundOutlined style={{ fontSize: "24px", color: "#666" }} />
                      </p>
                      <p className="ant-upload-text text-sm">Click or drag audio file to upload</p>
                      <p className="ant-upload-hint text-xs text-gray-500">
                        Support for MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM formats. Max file size: 25 MB.
                      </p>
                    </Dragger>
                  ) : (
                    <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <div className="flex items-center gap-2 flex-1">
                        <SoundOutlined style={{ fontSize: "20px", color: "#666" }} />
                        <span className="text-sm font-medium">{uploadedAudio.name}</span>
                        <span className="text-xs text-gray-500">
                          ({(uploadedAudio.size / 1024 / 1024).toFixed(2)} MB)
                        </span>
                      </div>
                      <button
                        className="bg-white shadow-sm border border-gray-200 rounded px-2 py-1 text-red-500 hover:bg-red-50 text-xs"
                        onClick={handleRemoveAudio}
                      >
                        <DeleteOutlined /> Remove
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Show file previews above input when files are uploaded */}
              {endpointType === EndpointType.RESPONSES && responsesUploadedImage && (
                <div className="mb-2">
                  <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="relative inline-block">
                      {responsesUploadedImage.name.toLowerCase().endsWith(".pdf") ? (
                        <div className="w-10 h-10 rounded-md bg-red-500 flex items-center justify-center">
                          <FilePdfOutlined style={{ fontSize: "16px", color: "white" }} />
                        </div>
                      ) : (
                        <img
                          src={responsesImagePreviewUrl || ""}
                          alt="Upload preview"
                          className="w-10 h-10 rounded-md border border-gray-200 object-cover"
                        />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-gray-900 truncate">{responsesUploadedImage.name}</div>
                      <div className="text-xs text-gray-500">
                        {responsesUploadedImage.name.toLowerCase().endsWith(".pdf") ? "PDF" : "Image"}
                      </div>
                    </div>
                    <button
                      className="flex items-center justify-center w-6 h-6 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded-full transition-colors"
                      onClick={handleRemoveResponsesImage}
                    >
                      <DeleteOutlined style={{ fontSize: "12px" }} />
                    </button>
                  </div>
                </div>
              )}

              {endpointType === EndpointType.CHAT && chatUploadedImage && (
                <div className="mb-2">
                  <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="relative inline-block">
                      {chatUploadedImage.name.toLowerCase().endsWith(".pdf") ? (
                        <div className="w-10 h-10 rounded-md bg-red-500 flex items-center justify-center">
                          <FilePdfOutlined style={{ fontSize: "16px", color: "white" }} />
                        </div>
                      ) : (
                        <img
                          src={chatImagePreviewUrl || ""}
                          alt="Upload preview"
                          className="w-10 h-10 rounded-md border border-gray-200 object-cover"
                        />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-gray-900 truncate">{chatUploadedImage.name}</div>
                      <div className="text-xs text-gray-500">
                        {chatUploadedImage.name.toLowerCase().endsWith(".pdf") ? "PDF" : "Image"}
                      </div>
                    </div>
                    <button
                      className="flex items-center justify-center w-6 h-6 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded-full transition-colors"
                      onClick={handleRemoveChatImage}
                    >
                      <DeleteOutlined style={{ fontSize: "12px" }} />
                    </button>
                  </div>
                </div>
              )}

              {/* Code Interpreter indicator and sample prompts when enabled */}
              {endpointType === EndpointType.RESPONSES && codeInterpreter.enabled && (
                <div className="mb-2 space-y-2">
                  <div className="px-3 py-2 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {isLoading ? (
                        <>
                          <LoadingOutlined className="text-blue-500" spin />
                          <span className="text-sm text-blue-700 font-medium">Running Python code...</span>
                        </>
                      ) : (
                        <>
                          <CodeOutlined className="text-blue-500" />
                          <span className="text-sm text-blue-700 font-medium">Code Interpreter Active</span>
                        </>
                      )}
                    </div>
                    <button
                      className="text-xs text-blue-500 hover:text-blue-700"
                      onClick={() => codeInterpreter.setEnabled(false)}
                    >
                      Disable
                    </button>
                  </div>
                  {/* Sample prompts - only show when not loading */}
                  {!isLoading && (
                    <div className="flex flex-wrap gap-2">
                      {[
                        "Generate sample sales data CSV and create a chart",
                        "Create a PNG bar chart comparing AI gateway providers including LiteLLM",
                        "Generate a CSV of LLM pricing data and visualize it as a line chart",
                      ].map((prompt, idx) => (
                        <button
                          key={idx}
                          className="text-xs px-3 py-1.5 bg-white border border-gray-200 rounded-full hover:bg-blue-50 hover:border-blue-300 hover:text-blue-600 transition-colors"
                          onClick={() => setInputMessage(prompt)}
                        >
                          {prompt}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Suggested prompts - show when chat is empty and not loading (skip for MCP - uses structured form) */}
              {chatHistory.length === 0 && !isLoading && endpointType !== EndpointType.MCP && (
                <div className="flex items-center gap-2 mb-3 overflow-x-auto">
                  {(endpointType === EndpointType.A2A_AGENTS
                    ? ["What can you help me with?", "Tell me about yourself", "What tasks can you perform?"]
                    : ["Write me a poem", "Explain quantum computing", "Draft a polite email requesting a meeting"]
                  ).map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      className="shrink-0 rounded-full border border-gray-200 px-3 py-1 text-xs font-medium text-gray-600 transition-colors hover:bg-blue-50 hover:border-blue-300 hover:text-blue-600 cursor-pointer"
                      onClick={() => setInputMessage(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              )}

              <div className="flex items-center gap-2">
                <div className="flex items-center flex-1 bg-white border border-gray-300 rounded-xl px-3 py-1 min-h-[44px]">
                  {/* Left: attachment and code interpreter icons */}
                  <div className="flex-shrink-0 mr-2 flex items-center gap-1">
                    {endpointType === EndpointType.RESPONSES && !responsesUploadedImage && (
                      <ResponsesImageUpload
                        responsesUploadedImage={responsesUploadedImage}
                        responsesImagePreviewUrl={responsesImagePreviewUrl}
                        onImageUpload={handleResponsesImageUpload}
                        onRemoveImage={handleRemoveResponsesImage}
                      />
                    )}
                    {endpointType === EndpointType.CHAT && !chatUploadedImage && (
                      <ChatImageUpload
                        chatUploadedImage={chatUploadedImage}
                        chatImagePreviewUrl={chatImagePreviewUrl}
                        onImageUpload={handleChatImageUpload}
                        onRemoveImage={handleRemoveChatImage}
                      />
                    )}
                    {/* Quick Code Interpreter toggle for Responses */}
                    {endpointType === EndpointType.RESPONSES && (
                      <Tooltip
                        title={
                          codeInterpreter.enabled
                            ? "Code Interpreter enabled (click to disable)"
                            : "Enable Code Interpreter"
                        }
                      >
                        <button
                          className={`p-1.5 rounded-md transition-colors ${codeInterpreter.enabled
                            ? "bg-blue-100 text-blue-600"
                            : "text-gray-400 hover:text-gray-600 hover:bg-gray-100"
                            }`}
                          onClick={() => {
                            codeInterpreter.toggle();
                            if (!codeInterpreter.enabled) {
                              NotificationsManager.success("Code Interpreter enabled!");
                            }
                          }}
                        >
                          <CodeOutlined style={{ fontSize: "16px" }} />
                        </button>
                      </Tooltip>
                    )}
                  </div>

                  {/* Middle: input field or MCP structured form */}
                  {endpointType === EndpointType.MCP &&
                  selectedMCPServers.length === 1 &&
                  selectedMCPServers[0] !== "__all__" &&
                  selectedMCPDirectTool ? (
                    <div className="flex-1 overflow-y-auto max-h-48 min-h-[44px] p-2 border border-gray-200 rounded-lg bg-gray-50/50">
                      {(() => {
                        const mcpTool = (serverToolsMap[selectedMCPServers[0]] || []).find(
                          (t: any) => t.name === selectedMCPDirectTool,
                        );
                        return mcpTool ? (
                          <MCPToolArgumentsForm
                            ref={mcpToolArgsFormRef}
                            tool={mcpTool}
                            className="space-y-2"
                          />
                        ) : (
                          <div className="flex items-center justify-center h-10 text-sm text-gray-500">
                            Loading tool schema...
                          </div>
                        );
                      })()}
                    </div>
                  ) : (
                    <TextArea
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder={
                        endpointType === EndpointType.CHAT ||
                        endpointType === EndpointType.EMBEDDINGS ||
                        endpointType === EndpointType.RESPONSES ||
                        endpointType === EndpointType.ANTHROPIC_MESSAGES
                          ? "Type your message... (Shift+Enter for new line)"
                          : endpointType === EndpointType.A2A_AGENTS
                            ? "Send a message to the A2A agent..."
                            : endpointType === EndpointType.IMAGE_EDITS
                              ? "Describe how you want to edit the image..."
                              : endpointType === EndpointType.SPEECH
                                ? "Enter text to convert to speech..."
                                : endpointType === EndpointType.TRANSCRIPTION
                                  ? "Optional: Add context or prompt for transcription..."
                                  : "Describe the image you want to generate..."
                      }
                      disabled={isLoading}
                      className="flex-1"
                      autoSize={{ minRows: 1, maxRows: 4 }}
                      style={{
                        resize: "none",
                        border: "none",
                        boxShadow: "none",
                        background: "transparent",
                        padding: "4px 0",
                        fontSize: "14px",
                        lineHeight: "20px",
                      }}
                    />
                  )}

                  {/* Right: send button - matching blue theme */}
                  <TremorButton
                    onClick={handleSendMessage}
                    disabled={
                      isLoading ||
                      (endpointType === EndpointType.MCP
                        ? !(
                            selectedMCPServers.length === 1 &&
                            selectedMCPServers[0] !== "__all__" &&
                            selectedMCPDirectTool
                          )
                        : endpointType === EndpointType.TRANSCRIPTION
                          ? !uploadedAudio
                          : !inputMessage.trim())
                    }
                    className="flex-shrink-0 ml-2 !w-8 !h-8 !min-w-8 !p-0 !rounded-full !bg-blue-600 hover:!bg-blue-700 disabled:!bg-gray-300 !border-none !text-white disabled:!text-gray-500 !flex !items-center !justify-center"
                  >
                    <ArrowUpOutlined style={{ fontSize: "14px" }} />
                  </TremorButton>
                </div>

                {isLoading && (
                  <TremorButton
                    onClick={handleCancelRequest}
                    className="bg-red-50 hover:bg-red-100 text-red-600 border-red-200"
                    icon={DeleteOutlined}
                  >
                    Cancel
                  </TremorButton>
                )}
              </div>
            </div>
          </div>
        </div>
      </Card>
      <Modal
        title="Generated Code"
        open={isGetCodeModalVisible}
        onCancel={() => setIsGetCodeModalVisible(false)}
        footer={null}
        width={800}
      >
        <div className="flex justify-between items-end my-4">
          <div>
            <Text className="font-medium block mb-1 text-gray-700">SDK Type</Text>
            <Select
              value={selectedSdk}
              onChange={(value) => setSelectedSdk(value as "openai" | "azure")}
              style={{ width: 150 }}
              options={[
                { value: "openai", label: "OpenAI SDK" },
                { value: "azure", label: "Azure SDK" },
              ]}
            />
          </div>
          <Button
            onClick={() => {
              navigator.clipboard.writeText(generatedCode);
              NotificationsManager.success("Copied to clipboard!");
            }}
          >
            Copy to Clipboard
          </Button>
        </div>
        <SyntaxHighlighter
          language="python"
          style={coy as any}
          wrapLines={true}
          wrapLongLines={true}
          className="rounded-md"
          customStyle={{
            maxHeight: "60vh",
            overflowY: "auto",
          }}
        >
          {generatedCode}
        </SyntaxHighlighter>
      </Modal>
    </div>
  );
};

export default ChatUI;
