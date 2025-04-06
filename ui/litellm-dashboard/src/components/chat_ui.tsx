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

import { message, Select, Spin, Typography, Tooltip } from "antd";
import { makeOpenAIChatCompletionRequest } from "./chat_ui/llm_calls/chat_completion";
import { makeOpenAIImageGenerationRequest } from "./chat_ui/llm_calls/image_generation";
import { fetchAvailableModels, ModelGroup  } from "./chat_ui/llm_calls/fetch_models";
import { litellmModeMapping, ModelMode, EndpointType, getEndpointType } from "./chat_ui/mode_endpoint_mapping";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from 'react-syntax-highlighter/dist/esm/styles/prism';
import EndpointSelector from "./chat_ui/EndpointSelector";
import { determineEndpointType } from "./chat_ui/EndpointUtils";
import { 
  SendOutlined, 
  ApiOutlined, 
  KeyOutlined, 
  ClearOutlined, 
  RobotOutlined, 
  UserOutlined,
  DeleteOutlined,
  LoadingOutlined
} from "@ant-design/icons";

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
  const [chatHistory, setChatHistory] = useState<{ role: string; content: string; model?: string; isImage?: boolean }[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | undefined>(
    undefined
  );
  const [showCustomModelInput, setShowCustomModelInput] = useState<boolean>(false);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const customModelTimeout = useRef<NodeJS.Timeout | null>(null);
  const [endpointType, setEndpointType] = useState<string>(EndpointType.CHAT);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const abortControllerRef = useRef<AbortController | null>(null);

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
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];

      if (lastMessage && lastMessage.role === role && !lastMessage.isImage) {
        return [
          ...prevHistory.slice(0, prevHistory.length - 1),
          { role, content: lastMessage.content + chunk, model },
        ];
      } else {
        return [...prevHistory, { role, content: chunk, model }];
      }
    });
  };

  const updateImageUI = (imageUrl: string, model: string) => {
    setChatHistory((prevHistory) => [
      ...prevHistory,
      { role: "assistant", content: imageUrl, model, isImage: true }
    ]);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      handleSendMessage();
    }
  };

  const handleCancelRequest = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsLoading(false);
      message.info("Request cancelled");
    }
  };

  const handleSendMessage = async () => {
    if (inputMessage.trim() === "") return;

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
    
    // Update UI with full message object
    setChatHistory([...chatHistory, newUserMessage]);
    setIsLoading(true);

    try {
      if (selectedModel) {
        // Use EndpointType enum for comparison
        if (endpointType === EndpointType.CHAT) {
          // Create chat history for API call - strip out model field and isImage field
          const apiChatHistory = [...chatHistory.filter(msg => !msg.isImage).map(({ role, content }) => ({ role, content })), newUserMessage];
          
          await makeOpenAIChatCompletionRequest(
            apiChatHistory,
            (chunk, model) => updateTextUI("assistant", chunk, model),
            selectedModel,
            effectiveApiKey,
            signal
          );
        } else if (endpointType === EndpointType.IMAGE) {
          // For image generation
          await makeOpenAIImageGenerationRequest(
            inputMessage,
            (imageUrl, model) => updateImageUI(imageUrl, model),
            selectedModel,
            effectiveApiKey,
            signal
          );
        }
      }
    } catch (error) {
      if (signal.aborted) {
        console.log("Request was cancelled");
      } else {
        console.error("Error fetching response", error);
        updateTextUI("assistant", "Error fetching response");
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }

    setInputMessage("");
  };

  const clearChatHistory = () => {
    setChatHistory([]);
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
                    ...modelInfo.map((option) => ({
                      value: option.model_group,
                      label: option.model_group
                    })),
                    { value: 'custom', label: 'Enter custom model' }
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
                  <div className="whitespace-pre-wrap break-words max-w-full message-content">
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
                                {...props}
                              >
                                {String(children).replace(/\n$/, '')}
                              </SyntaxHighlighter>
                            ) : (
                              <code className={`${className} px-1.5 py-0.5 rounded bg-gray-100 text-sm font-mono`} {...props}>
                                {children}
                              </code>
                            );
                          }
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
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
            <div className="flex items-center">
              <TextInput
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  endpointType === EndpointType.CHAT 
                    ? "Type your message..." 
                    : "Describe the image you want to generate..."
                }
                disabled={isLoading}
                className="flex-1"
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
                  {endpointType === EndpointType.CHAT ? "Send" : "Generate"}
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
