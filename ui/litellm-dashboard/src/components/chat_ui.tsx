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
import { modelAvailableCall } from "./networking";
import openai from "openai";
import { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { Typography } from "antd";
import { coy } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface ChatUIProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  disabledPersonalKeyCreation: boolean;
}

function getImageFormatFromBase64(base64String: string): string {
  const raw = atob(base64String.substring(0, 12));
  const hex = Array.from(raw).map(c => c.charCodeAt(0).toString(16).padStart(2, "0")).join(" ");

  if (hex.startsWith("89 50 4e 47")) return "image/png";
  if (hex.startsWith("ff d8 ff")) return "image/jpeg";
  if (hex.startsWith("47 49 46 38")) return "image/gif";
  if (hex.startsWith("42 4d")) return "image/bmp";
  if (hex.startsWith("52 49 46 46")) return "image/webp";

  return "unknown";
}

async function generateModelResponse(
  chatHistory: { role: string; content: string }[],
  updateUI: (chunk: string, model: string) => void,
  selectedModel: string,
  accessToken: string
) {
  // base url should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal !== true) {
    console.log = function () {};
  }
  console.log("isLocal:", isLocal);
  const proxyBaseUrl = isLocal
    ? "http://localhost:4000"
    : window.location.origin;
  const client = new openai.OpenAI({
    apiKey: accessToken, // Replace with your OpenAI API key
    baseURL: proxyBaseUrl, // Replace with your OpenAI API base URL
    dangerouslyAllowBrowser: true, // using a temporary litellm proxy key
  });

  try {
    const image_models = ["flux", "stable-diffusion"];
    if (image_models.some(model => selectedModel.toLowerCase().includes(model))) {
      const response = await client.images.generate({
        model: selectedModel,
        prompt: chatHistory.map(msg => msg.content).filter(content => !content.startsWith("!")).join("\n"),
        size: "512x512",
        response_format: "b64_json",
        n: 1
      }); 
      if (response.data[0]) {
        const base64Image = response.data[0].b64_json;
        const mime = getImageFormatFromBase64(base64Image!);
        updateUI(`![Generated with ${selectedModel}](data:${mime};base64,${base64Image})`, selectedModel);
      } else {
        updateUI("It wasn't possible to generate the image!", selectedModel);
      }
    } else {
      const response = await client.chat.completions.create({
        model: selectedModel,
        stream: true,
        messages: chatHistory as ChatCompletionMessageParam[],
      });

      for await (const chunk of response) {
        if (chunk.choices[0].delta.content) {
          updateUI(chunk.choices[0].delta.content, chunk.model);
        }
      }
    }
  } catch (error) {
    message.error(`Error occurred while generating model response. Please try again. Error: ${error}`, 20);
  }
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
  const [chatHistory, setChatHistory] = useState<{ role: string; content: string; model?: string }[]>([]);
  const [isSpinnerVisible, setSpinnerVisible] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | undefined>(
    undefined
  );
  const [modelInfo, setModelInfo] = useState<any[]>([]);

  const chatEndRef = useRef<HTMLDivElement>(null);

  const toggleSpinner = () => {
    setSpinnerVisible(isSpinnerVisible => !isSpinnerVisible);
  };
  
  useEffect(() => {
    if (!accessToken || !token || !userRole || !userID) {
      return;
    }

    // Fetch model info and set the default selected model
    const fetchModelInfo = async () => {
      try {
        const fetchedAvailableModels = await modelAvailableCall(
          accessToken,
          userID,
          userRole
        );
  
        console.log("model_info:", fetchedAvailableModels);
  
        if (fetchedAvailableModels?.data.length > 0) {
          // Create a Map to store unique models using the model ID as key
          const uniqueModelsMap = new Map();
          
          fetchedAvailableModels["data"].forEach((item: { id: string }) => {
            uniqueModelsMap.set(item.id, {
              value: item.id,
              label: item.id
            });
          });

          // Convert Map values back to array
          const uniqueModels = Array.from(uniqueModelsMap.values());

          // Sort models alphabetically
          uniqueModels.sort((a, b) => a.label.localeCompare(b.label));

          setModelInfo(uniqueModels);
          setSelectedModel(uniqueModels[0].value);
        }
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };
  
    fetchModelInfo();
  }, [accessToken, userID, userRole]);
  

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

  const updateUI = (role: string, chunk: string, model?: string) => {
    setChatHistory((prevHistory) => {
      const lastMessage = prevHistory[prevHistory.length - 1];

      if (lastMessage && lastMessage.role === role) {
        return [
          ...prevHistory.slice(0, prevHistory.length - 1),
          { role, content: lastMessage.content + chunk, model },
        ];
      } else {
        return [...prevHistory, { role, content: chunk, model }];
      }
    });
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
    toggleSpinner();
    setInputMessage("");

    // Create message object without model field for API call
    const newUserMessage = { role: "user", content: inputMessage };
    
    // Create chat history for API call - strip out model field
    const apiChatHistory = [...chatHistory.map(({ role, content }) => ({ role, content })), newUserMessage];
    
    // Update UI with full message object (including model field for display)
    setChatHistory([...chatHistory, newUserMessage]);

    try {
      if (selectedModel) {
        await generateModelResponse(
          apiChatHistory,
          (chunk, model) => updateUI("assistant", chunk, model),
          selectedModel,
          effectiveApiKey
        );
        toggleSpinner();
      }
    } catch (error) {
      console.error("Error fetching model response", error);
      updateUI("assistant", "Error fetching model response");
      toggleSpinner();
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

  const onChange = (value: string) => {
    console.log(`selected ${value}`);
    setSelectedModel(value);
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
                <div className="sm:max-w-4xl">
                  <Grid numItems={3}>
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
                        onChange={onChange}
                        options={modelInfo}
                        style={{ width: "350px" }}
                        showSearch={true}
                      />
                    </Col>
                    <Col>
                  {/* Clear Chat Button */}
                  <Button
                    onClick={clearChatHistory}
                    className="mt-4"
                  >
                    Clear Chat
                  </Button>
                    </Col>
                  </Grid>
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
                            <ReactMarkdown
                              urlTransform={(value: string) => value}
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
                                },
                                img({ node, ...props }) {
                                  const { alt, src } = props;
                                  return <img {...props} style={{ maxWidth: '100%' }} alt={alt || "Generated Image"} />;
                              }}
                            >
                              {message.content}
                            </ReactMarkdown>
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
                    <div className={isSpinnerVisible ? 'spinner' : 'spinner hidden'} role="status">
                        <svg aria-hidden="true" className="w-8 h-8 text-gray-200 animate-spin dark:text-gray-600 fill-blue-600 mr-4" viewBox="0 0 100 101" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z" fill="currentColor"/>
                            <path d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z" fill="currentFill"/>
                        </svg>
                        <span className="sr-only">Loading...</span>
                    </div>
                    <TextInput
                      type="text"
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Type your message..."
                      readOnly={isSpinnerVisible === true}
                    />
                    <Button
                      onClick={handleSendMessage}
                      className="ml-2"
                      disabled={isSpinnerVisible === true}
                    >
                      Send
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
