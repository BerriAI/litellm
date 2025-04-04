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
} from "@tremor/react";

import { message, Select } from "antd";
import { makeOpenAIChatCompletionRequest } from "./chat_ui/llm_calls/chat_completion";
import { makeOpenAIImageGenerationRequest } from "./chat_ui/llm_calls/image_generation";
import { fetchAvailableModels, ModelGroup  } from "./chat_ui/llm_calls/fetch_models";
import { litellmModeMapping, ModelMode, EndpointType, getEndpointType } from "./chat_ui/mode_endpoint_mapping";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { Typography } from "antd";
import { coy } from 'react-syntax-highlighter/dist/esm/styles/prism';
import EndpointSelector from "./chat_ui/EndpointSelector";
import { determineEndpointType } from "./chat_ui/EndpointUtils";

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

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let useApiKey = apiKeySource === 'session' ? accessToken : apiKey;
    console.log("useApiKey:", useApiKey);
    if (!useApiKey || !token || !userRole || !userID) {
      console.log("useApiKey or token or userRole or userID is missing = ", useApiKey, token, userRole, userID);
      return;
    }

    // Fetch model info and set the default selected model
    const loadModels = async () => {
      try {
        const uniqueModels = await fetchAvailableModels(
          useApiKey,
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

    // Create message object without model field for API call
    const newUserMessage = { role: "user", content: inputMessage };
    
    // Update UI with full message object
    setChatHistory([...chatHistory, newUserMessage]);

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
            effectiveApiKey
          );
        } else if (endpointType === EndpointType.IMAGE) {
          // For image generation
          await makeOpenAIImageGenerationRequest(
            inputMessage,
            (imageUrl, model) => updateImageUI(imageUrl, model),
            selectedModel,
            effectiveApiKey
          );
        }
      }
    } catch (error) {
      console.error("Error fetching response", error);
      updateTextUI("assistant", "Error fetching response");
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

  return (
    <div style={{ width: "100%", position: "relative" }}>
      <Grid className="gap-2 p-8 h-[80vh] w-full mt-2">
        <Card>
          
          <TabGroup>
            <TabList>
              <Tab>Chat</Tab>
            </TabList>
            <TabPanels>
              <TabPanel>
                <div className="sm:max-w-2xl">
                  <Grid numItems={2}>
                    <Col>
                      <Text>API Key Source</Text>
                      <Select
                        disabled={disabledPersonalKeyCreation}
                        defaultValue="session"
                        style={{ width: "100%" }}
                        onChange={(value) => setApiKeySource(value as "session" | "custom")}
                        options={[
                          { value: 'session', label: 'Current UI Session' },
                          { value: 'custom', label: 'Virtual Key' },
                        ]}
                      />
                      {apiKeySource === 'custom' && (
                        <TextInput
                          className="mt-2"
                          placeholder="Enter custom API key"
                          type="password"
                          onValueChange={setApiKey}
                          value={apiKey}
                        />
                      )}
                    </Col>
                    <Col className="mx-2">
                      <Text>Select Model:</Text>
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
                        style={{ width: "350px" }}
                        showSearch={true}
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
                      <EndpointSelector 
                        endpointType={endpointType}
                        onEndpointChange={handleEndpointChange}
                        className="mt-2"
                      />
                    </Col>

                  </Grid>

                  {/* Clear Chat Button */}
                  <Button
                    onClick={clearChatHistory}
                    className="mt-4"
                  >
                    Clear Chat
                  </Button>
                </div>
                <Table
                  className="mt-5"
                  style={{
                    display: "block",
                    maxHeight: "60vh",
                    overflowY: "auto",
                  }}
                >
                  <TableHead>
                    <TableRow>
                      <TableCell>
                        {/* <Title>Chat</Title> */}
                      </TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {chatHistory.map((message, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          <div style={{ 
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            marginBottom: '4px'
                          }}>
                            <strong>{message.role}</strong>
                            {message.role === "assistant" && message.model && (
                              <span style={{
                                fontSize: '12px',
                                color: '#666',
                                backgroundColor: '#f5f5f5',
                                padding: '2px 6px',
                                borderRadius: '4px',
                                fontWeight: 'normal'
                              }}>
                                {message.model}
                              </span>
                            )}
                          </div>
                          <div style={{ 
                            whiteSpace: "pre-wrap", 
                            wordBreak: "break-word",
                            maxWidth: "100%"
                          }}>
                            {message.isImage ? (
                              <img 
                                src={message.content} 
                                alt="Generated image" 
                                style={{ maxWidth: '100%', maxHeight: '500px' }} 
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
                                        {...props}
                                      >
                                        {String(children).replace(/\n$/, '')}
                                      </SyntaxHighlighter>
                                    ) : (
                                      <code className={className} {...props}>
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
                        </TableCell>
                      </TableRow>
                    ))}
                    <TableRow>
                      <TableCell>
                        <div ref={chatEndRef} style={{ height: "1px" }} />
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
                <div
                  className="mt-3"
                  style={{ position: "absolute", bottom: 5, width: "95%" }}
                >
                  <div className="flex" style={{ marginTop: "16px" }}>
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
                    />
                    <Button
                      onClick={handleSendMessage}
                      className="ml-2"
                    >
                      {endpointType === EndpointType.CHAT ? "Send" : "Generate"}
                    </Button>
                  </div>
                </div>
              </TabPanel>
              
            </TabPanels>
          </TabGroup>
        </Card>
      </Grid>
    </div>
  );
};

export default ChatUI;
