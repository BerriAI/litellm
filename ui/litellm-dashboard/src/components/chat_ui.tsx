import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import {
  Card,
  Title,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Grid,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Metric,
  Col,
  Text,
  SelectItem,
  TextInput,
  Button,
  Divider,
} from "@tremor/react";
import { v4 as uuidv4 } from 'uuid';

import { message, Select, Spin, Typography, Tooltip, Input, Upload } from "antd";
import { makeOpenAIChatCompletionRequest } from "./chat_ui/llm_calls/chat_completion";
import { makeOpenAIImageGenerationRequest } from "./chat_ui/llm_calls/image_generation";
import { makeOpenAIImageEditsRequest } from "./chat_ui/llm_calls/image_edits";
import { makeOpenAIResponsesRequest } from "./chat_ui/llm_calls/responses_api";
import { fetchAvailableModels, ModelGroup  } from "./chat_ui/llm_calls/fetch_models";
import { litellmModeMapping, ModelMode, EndpointType, getEndpointType } from "./chat_ui/mode_endpoint_mapping";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from 'react-syntax-highlighter/dist/esm/styles/prism';
import EndpointSelector from "./chat_ui/EndpointSelector";
import TagSelector from "./tag_management/TagSelector";
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";
import GuardrailSelector from "./guardrails/GuardrailSelector";
import { determineEndpointType } from "./chat_ui/EndpointUtils";
import { MessageType } from "./chat_ui/types";
import ReasoningContent from "./chat_ui/ReasoningContent";
import ResponseMetrics, { TokenUsage } from "./chat_ui/ResponseMetrics";
import { 
  SendOutlined, 
  ApiOutlined, 
  KeyOutlined, 
  ClearOutlined, 
  RobotOutlined, 
  UserOutlined,
  DeleteOutlined,
  LoadingOutlined,
  TagsOutlined,
  DatabaseOutlined,
  InfoCircleOutlined,
  SafetyOutlined,
  UploadOutlined,
  PictureOutlined
} from "@ant-design/icons";

const { TextArea } = Input;
const { Dragger } = Upload;

interface ChatUIProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  disabledPersonalKeyCreation: boolean;
}

const ChatUI: React.FC<ChatUIProps> = ({
  accessToken,
  token,
  userRole,
  userID,
  disabledPersonalKeyCreation,
}) => {
  const [apiKeySource, setApiKeySource] = useState<'session' | 'custom'>(
    disabledPersonalKeyCreation ? 'custom' : 'session'
  );
  const [apiKey, setApiKey] = useState("");
  const [inputMessage, setInputMessage] = useState("");
  const [chatHistory, setChatHistory] = useState<MessageType[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | undefined>(
    undefined
  );
  const [showCustomModelInput, setShowCustomModelInput] = useState<boolean>(false);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const customModelTimeout = useRef<NodeJS.Timeout | null>(null);
  const [endpointType, setEndpointType] = useState<string>(EndpointType.CHAT);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [selectedVectorStores, setSelectedVectorStores] = useState<string[]>([]);
  const [selectedGuardrails, setSelectedGuardrails] = useState<string[]>([]);
  const [messageTraceId, setMessageTraceId] = useState<string | null>(null);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let userApiKey = apiKeySource === 'session' ? accessToken : apiKey;
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
        const uniqueModels = await fetchAvailableModels(
          userApiKey,
        );
  
        console.log("Fetched models:", uniqueModels);
  
        if (uniqueModels.length > 0) {
          setModelInfo(uniqueModels);
          setSelectedModel(uniqueModels[0].model_group);
          
          // Auto-set endpoint based on the first model's mode
          if (uniqueModels[0].mode) {
            const initialEndpointType = determineEndpointType(uniqueModels[0].model_group, uniqueModels);
            setEndpointType(initialEndpointType);
          }
        }
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };
  
    loadModels();
  }, [accessToken, userID, userRole, apiKeySource, apiKey]);
  

  useEffect(() => {
    // Scroll to the bottom of the chat whenever chatHistory updates
    if (chatEndRef.current) {
      // Add a small delay to ensure content is rendered
      setTimeout(() => {
        chatEndRef.current?.scrollIntoView({ 
          behavior: "smooth",
          block: "end" // Keep the scroll position at the end
        });
      }, 100);
    }
  }, [chatHistory]);

  const updateTextUI = (role: string, chunk: string, model?: string) => {
    console.log("updateTextUI called with:", role, chunk, model);
    setChatHistory((prev) => {
      const last = prev[prev.length - 1];
      // if the last message is already from this same role, append
      if (last && last.role === role && !last.isImage) {
        // build a new object, but only set `model` if it wasn't there already
        const updated: MessageType = {
          ...last,
          content: last.content + chunk,
          model: last.model ?? model,      // ← only use the passed‐in model on the first chunk
        };
        return [...prev.slice(0, -1), updated];
      } else {
        // otherwise start a brand new assistant bubble
        return [
          ...prev,
          {
            role,
            content: chunk,
            model,                          // model set exactly once here
          },
        ];
      }
    });
  };

  const updateReasoningContent = (chunk: string) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];
      
      if (lastMessage && lastMessage.role === "assistant" && !lastMessage.isImage) {
        return [
          ...prevHistory.slice(0, prevHistory.length - 1),
          { 
            ...lastMessage,
            reasoningContent: (lastMessage.reasoningContent || "") + chunk 
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
              reasoningContent: chunk 
            }
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
            timeToFirstToken
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
            timeToFirstToken 
          }
        ];
      }
      
      console.log("No appropriate message found to update timing");
      return prevHistory;
    });
  };

  const updateUsageData = (usage: TokenUsage) => {
    console.log("Received usage data:", usage);
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];
      
      if (lastMessage && lastMessage.role === "assistant") {
        console.log("Updating message with usage data:", usage);
        const updatedMessage = { 
          ...lastMessage,
          usage
        };
        console.log("Updated message:", updatedMessage);
        
        return [
          ...prevHistory.slice(0, prevHistory.length - 1),
          updatedMessage
        ];
      }
      
      return prevHistory;
    });
  };

  const updateImageUI = (imageUrl: string, model: string) => {
    setChatHistory((prevHistory) => [
      ...prevHistory,
      { role: "assistant", content: imageUrl, model, isImage: true }
    ]);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
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
      message.info("Request cancelled");
    }
  };

  const handleImageUpload = (file: File) => {
    setUploadedImage(file);
    const previewUrl = URL.createObjectURL(file);
    setImagePreviewUrl(previewUrl);
    return false; // Prevent default upload behavior
  };

  const handleRemoveImage = () => {
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
    }
    setUploadedImage(null);
    setImagePreviewUrl(null);
  };

  const handleSendMessage = async () => {
    if (inputMessage.trim() === "") return;

    // For image edits, require both image and prompt
    if (endpointType === EndpointType.IMAGE_EDITS && !uploadedImage) {
      message.error("Please upload an image for editing");
      return;
    }

    if (!token || !userRole || !userID) {
      return;
    }

    const effectiveApiKey = apiKeySource === 'session' ? accessToken : apiKey;

    if (!effectiveApiKey) {
      message.error("Please provide an API key or select Current UI Session");
      return;
    }

    // Create new abort controller for this request
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    // Create message object without model field for API call
    const newUserMessage = { role: "user", content: inputMessage };
    
    // Generate new trace ID for a new conversation or use existing one
    const traceId = messageTraceId || uuidv4();
    if (!messageTraceId) {
      setMessageTraceId(traceId);
    }
    
    // Update UI with full message object
    setChatHistory([...chatHistory, newUserMessage]);
    setIsLoading(true);

    try {
      if (selectedModel) {
        if (endpointType === EndpointType.CHAT) {
          // Create chat history for API call - strip out model field and isImage field
          const apiChatHistory = [...chatHistory.filter(msg => !msg.isImage).map(({ role, content }) => ({ role, content })), newUserMessage];
          
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
            selectedGuardrails.length > 0 ? selectedGuardrails : undefined
          );
        } else if (endpointType === EndpointType.IMAGE) {
          // For image generation
          await makeOpenAIImageGenerationRequest(
            inputMessage,
            (imageUrl, model) => updateImageUI(imageUrl, model),
            selectedModel,
            effectiveApiKey,
            selectedTags,
            signal
          );
        } else if (endpointType === EndpointType.IMAGE_EDITS) {
          // For image edits
          if (uploadedImage) {
            await makeOpenAIImageEditsRequest(
              uploadedImage,
              inputMessage,
              (imageUrl, model) => updateImageUI(imageUrl, model),
              selectedModel,
              effectiveApiKey,
              selectedTags,
              signal
            );
          }
        } else if (endpointType === EndpointType.RESPONSES) {
          // Create chat history for API call - strip out model field and isImage field
          const apiChatHistory = [...chatHistory.filter(msg => !msg.isImage).map(({ role, content }) => ({ role, content })), newUserMessage];
          
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
            selectedGuardrails.length > 0 ? selectedGuardrails : undefined
          );
        }
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
        handleRemoveImage();
      }
    }

    setInputMessage("");
  };

  const clearChatHistory = () => {
    setChatHistory([]);
    setMessageTraceId(null);
    handleRemoveImage(); // Clear any uploaded images
    message.success("Chat history cleared.");
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
    
    // Use the utility function to determine the endpoint type
    if (value !== 'custom') {
      const newEndpointType = determineEndpointType(value, modelInfo);
      setEndpointType(newEndpointType);
    }
    
    setShowCustomModelInput(value === 'custom');
  };

  const handleEndpointChange = (value: string) => {
    setEndpointType(value);
  };

  const antIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;

  return (
    <div className="w-full h-screen p-4 bg-white">
    <Card className="w-full rounded-xl shadow-md overflow-hidden">
      <div className="flex h-[80vh] w-full">
        {/* Left Sidebar with Controls */}
        <div className="w-1/4 p-4 border-r border-gray-200 bg-gray-50">
          <div className="mb-6">
            <div className="space-y-6">
              <div>
                <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                  <KeyOutlined className="mr-2" /> API Key Source
                </Text>
                <Select
                  disabled={disabledPersonalKeyCreation}
                  defaultValue="session"
                  style={{ width: "100%" }}
                  onChange={(value) => setApiKeySource(value as "session" | "custom")}
                  options={[
                    { value: 'session', label: 'Current UI Session' },
                    { value: 'custom', label: 'Virtual Key' },
                  ]}
                  className="rounded-md"
                />
                {apiKeySource === 'custom' && (
                  <TextInput
                    className="mt-2"
                    placeholder="Enter custom API key"
                    type="password"
                    onValueChange={setApiKey}
                    value={apiKey}
                    icon={KeyOutlined}
                  />
                )}
              </div>
              
              <div>
                <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                  <RobotOutlined className="mr-2" /> Select Model
                </Text>
                <Select
                  placeholder="Select a Model"
                  onChange={onModelChange}
                  options={[
                    ...Array.from(new Set(modelInfo.map(option => option.model_group)))
                      .map((model_group, index) => ({
                        value: model_group,
                        label: model_group,
                        key: index
                      })),
                    { value: 'custom', label: 'Enter custom model', key: 'custom' }
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
              
              <div>
                <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                  <ApiOutlined className="mr-2" /> Endpoint Type
                </Text>
                <EndpointSelector 
                  endpointType={endpointType}
                  onEndpointChange={handleEndpointChange}
                  className="mb-4"
                />  
              </div>

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

              <div>
                <Text className="font-medium block mb-2 text-gray-700 flex items-center">
                  <DatabaseOutlined className="mr-2" /> Vector Store
                  <Tooltip 
                    className="ml-1"
                    title={
                        <span>
                          Select vector store(s) to use for this LLM API call. You can set up your vector store <a href="?page=vector-stores" style={{ color: '#1890ff' }}>here</a>.
                        </span>
                      }>
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
                          Select guardrail(s) to use for this LLM API call. You can set up your guardrails <a href="?page=guardrails" style={{ color: '#1890ff' }}>here</a>.
                        </span>
                      }>
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
              
              <Button
                onClick={clearChatHistory}
                className="w-full bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300 mt-4"
                icon={ClearOutlined}
              >
                Clear Chat
              </Button>
            </div>
          </div>
        </div>
        
        {/* Main Chat Area */}
        <div className="w-3/4 flex flex-col bg-white">
          <div className="flex-1 overflow-auto p-4 pb-0">
            {chatHistory.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-gray-400">
                <RobotOutlined style={{ fontSize: '48px', marginBottom: '16px' }} />
                <Text>Start a conversation or generate an image</Text>
              </div>
            )}
            
            {chatHistory.map((message, index) => (
              <div 
                key={index} 
                className={`mb-4 ${message.role === "user" ? "text-right" : "text-left"}`}
              >
                <div className="inline-block max-w-[80%] rounded-lg shadow-sm p-3.5 px-4" style={{
                  backgroundColor: message.role === "user" ? '#f0f8ff' : '#ffffff',
                  border: message.role === "user" ? '1px solid #e6f0fa' : '1px solid #f0f0f0',
                  textAlign: 'left'
                }}>
                  <div className="flex items-center gap-2 mb-1.5">
                    <div className="flex items-center justify-center w-6 h-6 rounded-full mr-1" style={{
                      backgroundColor: message.role === "user" ? '#e6f0fa' : '#f5f5f5',
                    }}>
                      {message.role === "user" ? 
                        <UserOutlined style={{ fontSize: '12px', color: '#2563eb' }} /> : 
                        <RobotOutlined style={{ fontSize: '12px', color: '#4b5563' }} />
                      }
                    </div>
                    <strong className="text-sm capitalize">{message.role}</strong>
                    {message.role === "assistant" && message.model && (
                      <span className="text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-600 font-normal">
                        {message.model}
                      </span>
                    )}
                  </div>
                  {message.reasoningContent && (
                    <ReasoningContent reasoningContent={message.reasoningContent} />
                  )}
                  <div className="whitespace-pre-wrap break-words max-w-full message-content" 
                       style={{ 
                         wordWrap: 'break-word', 
                         overflowWrap: 'break-word',
                         wordBreak: 'break-word',
                         hyphens: 'auto'
                       }}>
                    {message.isImage ? (
                      <img 
                        src={message.content} 
                        alt="Generated image" 
                        className="max-w-full rounded-md border border-gray-200 shadow-sm" 
                        style={{ maxHeight: '500px' }} 
                      />
                    ) : (
                      <ReactMarkdown
                        components={{
                          code({node, inline, className, children, ...props}: React.ComponentPropsWithoutRef<'code'> & {
                            inline?: boolean;
                            node?: any;
                          }) {
                            const match = /language-(\w+)/.exec(className || '');
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
                                {String(children).replace(/\n$/, '')}
                              </SyntaxHighlighter>
                            ) : (
                              <code className={`${className} px-1.5 py-0.5 rounded bg-gray-100 text-sm font-mono`} style={{ wordBreak: 'break-word' }} {...props}>
                                {children}
                              </code>
                            );
                          },
                          pre: ({ node, ...props }) => (
                            <pre style={{ overflowX: 'auto', maxWidth: '100%' }} {...props} />
                          )
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    )}
                                        
                    {message.role === "assistant" && (message.timeToFirstToken || message.usage) && (
                      <ResponseMetrics 
                        timeToFirstToken={message.timeToFirstToken}
                        usage={message.usage}
                      />
                    )}
                  </div>
                </div>
              </div>
            ))}
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
                {!uploadedImage ? (
                  <Dragger
                    beforeUpload={handleImageUpload}
                    accept="image/*"
                    showUploadList={false}
                    className="border-dashed border-2 border-gray-300 rounded-lg p-4"
                  >
                    <p className="ant-upload-drag-icon">
                      <PictureOutlined style={{ fontSize: '24px', color: '#666' }} />
                    </p>
                    <p className="ant-upload-text text-sm">Click or drag image to upload</p>
                    <p className="ant-upload-hint text-xs text-gray-500">
                      Support for PNG, JPG, JPEG formats
                    </p>
                  </Dragger>
                ) : (
                  <div className="relative inline-block">
                    <img 
                      src={imagePreviewUrl || ''} 
                      alt="Upload preview" 
                      className="max-w-32 max-h-32 rounded-md border border-gray-200 object-cover"
                    />
                    <button
                      className="absolute top-1 right-1 bg-white shadow-sm border border-gray-200 rounded px-1 py-1 text-red-500 hover:bg-red-50 text-xs"
                      onClick={handleRemoveImage}
                    >
                      <DeleteOutlined />
                    </button>
                  </div>
                )}
              </div>
            )}
            
            <div className="flex items-center">
              <TextArea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  endpointType === EndpointType.CHAT || endpointType === EndpointType.RESPONSES
                    ? "Type your message... (Shift+Enter for new line)" 
                    : endpointType === EndpointType.IMAGE_EDITS
                    ? "Describe how you want to edit the image..."
                    : "Describe the image you want to generate..."
                }
                disabled={isLoading}
                className="flex-1"
                autoSize={{ minRows: 1, maxRows: 6 }}
                style={{ resize: 'none', paddingRight: '10px', paddingLeft: '10px' }}
              />
              {isLoading ? (
                <Button
                  onClick={handleCancelRequest}
                  className="ml-2 bg-red-50 hover:bg-red-100 text-red-600 border-red-200"
                  icon={DeleteOutlined}
                >
                  Cancel
                </Button>
              ) : (
                <Button
                  onClick={handleSendMessage}
                  className="ml-2 text-white"
                  icon={endpointType === EndpointType.CHAT ? SendOutlined : RobotOutlined}
                >
                  {endpointType === EndpointType.CHAT ? "Send" : 
                   endpointType === EndpointType.IMAGE_EDITS ? "Edit" : "Generate"}
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </Card>
    </div>
  );
};

export default ChatUI;
