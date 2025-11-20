import React, { useEffect, useRef, useState } from "react";
import {
  ArrowUpOutlined,
  ClearOutlined,
  LoadingOutlined,
  RobotOutlined,
  UserOutlined,
} from "@ant-design/icons";
import { Button as TremorButton } from "@tremor/react";
import { Button, Input, Spin } from "antd";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import NotificationsManager from "../../molecules/notifications_manager";
import ResponseMetrics, { TokenUsage } from "../../playground/chat_ui/ResponseMetrics";
import { PromptType } from "./types";
import { convertToDotPrompt, extractVariables } from "./utils";
import { getProxyBaseUrl } from "../../networking";

const { TextArea } = Input;

interface ConversationPanelProps {
  prompt: PromptType;
  accessToken: string | null;
}

interface Message {
  role: string;
  content: string;
  model?: string;
  timeToFirstToken?: number;
  totalLatency?: number;
  usage?: TokenUsage;
}

const ConversationPanel: React.FC<ConversationPanelProps> = ({ prompt, accessToken }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [variablesFilled, setVariablesFilled] = useState(false);
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const extractedVariables = extractVariables(prompt);

  // Check if all variables are filled
  const allVariablesFilled = extractedVariables.every(
    (varName) => variables[varName] && variables[varName].trim() !== ""
  );

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "end",
        });
      }, 100);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("Access token is required");
      return;
    }

    // Check if variables are filled
    if (extractedVariables.length > 0 && !allVariablesFilled) {
      NotificationsManager.fromBackend("Please fill in all template variables");
      return;
    }

    if (!inputMessage.trim()) {
      return;
    }

    // Mark variables as filled on first send
    if (!variablesFilled && extractedVariables.length > 0) {
      setVariablesFilled(true);
    }

    const userMessage: Message = { role: "user", content: inputMessage };
    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");

    const controller = new AbortController();
    setAbortController(controller);
    setIsLoading(true);

    const startTime = Date.now();
    let timeToFirstToken: number | undefined;

    try {
      const dotpromptContent = convertToDotPrompt(prompt);
      const proxyBaseUrl = getProxyBaseUrl();

      const promptVariablesWithInput = {
        ...variables,
        ...(messages.length === 0 && { user_message: inputMessage }),
      };

      const response = await fetch(`${proxyBaseUrl}/prompts/test`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          dotprompt_content: dotpromptContent,
          prompt_variables: promptVariablesWithInput,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, ${errorText}`);
      }

      if (!response.body) {
        throw new Error("No response body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let assistantMessage = "";
      let model: string | undefined;
      let usage: TokenUsage | undefined;
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") {
              continue;
            }

            try {
              const parsed = JSON.parse(data);
              
              if (!model && parsed.model) {
                model = parsed.model;
              }

              if (parsed.usage) {
                usage = parsed.usage;
              }

              const content = parsed.choices?.[0]?.delta?.content;
              if (content) {
                if (!timeToFirstToken) {
                  timeToFirstToken = Date.now() - startTime;
                }
                assistantMessage += content;
                setMessages((prev) => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1] = {
                    role: "assistant",
                    content: assistantMessage,
                    model,
                    timeToFirstToken,
                  };
                  return newMessages;
                });
              }
            } catch (e) {
              console.error("Error parsing chunk:", e);
            }
          }
        }
      }

      const totalLatency = Date.now() - startTime;
      setMessages((prev) => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          ...newMessages[newMessages.length - 1],
          totalLatency,
          usage,
        };
        return newMessages;
      });
    } catch (error: any) {
      if (error.name === "AbortError") {
        console.log("Request was cancelled");
      } else {
        console.error("Error testing prompt:", error);
        setMessages((prev) => {
          const lastMsg = prev[prev.length - 1];
          if (lastMsg && lastMsg.role === "assistant" && lastMsg.content === "") {
            return [
              ...prev.slice(0, -1),
              { role: "assistant", content: `Error: ${error.message}` },
            ];
          }
          return [
            ...prev,
            { role: "assistant", content: `Error: ${error.message}` },
          ];
        });
      }
    } finally {
      setIsLoading(false);
      setAbortController(null);
    }
  };

  const handleCancelRequest = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      setIsLoading(false);
      NotificationsManager.info("Request cancelled");
    }
  };

  const handleClearConversation = () => {
    setMessages([]);
    setVariablesFilled(false);
    NotificationsManager.success("Chat history cleared.");
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const antIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;

  return (
    <div className="flex flex-col h-full bg-white">
      {!variablesFilled && extractedVariables.length > 0 && (
        <div className="p-4 border-b border-gray-200 bg-blue-50">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">
            Fill in template variables to start testing
          </h3>
          <div className="space-y-2">
            {extractedVariables.map((varName) => (
              <div key={varName}>
                <label className="block text-xs text-gray-600 mb-1 font-medium">
                  {"{{"}{varName}{"}}"}
                </label>
                <Input
                  value={variables[varName] || ""}
                  onChange={(e) =>
                    setVariables({ ...variables, [varName]: e.target.value })
                  }
                  placeholder={`Enter value for ${varName}`}
                  size="small"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex-1 overflow-auto p-4 pb-0">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-gray-400">
            <RobotOutlined style={{ fontSize: "48px", marginBottom: "16px" }} />
            <span className="text-base">
              {extractedVariables.length > 0
                ? "Fill in the variables above, then type a message to start testing"
                : "Type a message below to start testing your prompt"}
            </span>
          </div>
        )}

        {messages.map((message, index) => (
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

                <div
                  className="whitespace-pre-wrap break-words max-w-full message-content"
                  style={{
                    wordWrap: "break-word",
                    overflowWrap: "break-word",
                    wordBreak: "break-word",
                    hyphens: "auto",
                  }}
                >
                  {message.role === "assistant" ? (
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
                      {message.content}
                    </ReactMarkdown>
                  ) : (
                    <div className="whitespace-pre-wrap">{message.content}</div>
                  )}

                  {message.role === "assistant" &&
                    (message.timeToFirstToken || message.totalLatency || message.usage) && (
                      <ResponseMetrics
                        timeToFirstToken={message.timeToFirstToken}
                        totalLatency={message.totalLatency}
                        usage={message.usage}
                      />
                    )}
                </div>
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-center items-center my-4">
            <Spin indicator={antIcon} />
          </div>
        )}
        <div ref={messagesEndRef} style={{ height: "1px" }} />
      </div>

      <div className="p-4 border-t border-gray-200 bg-white">
        {messages.length > 0 && (
          <div className="mb-2 flex justify-end">
            <TremorButton
              onClick={handleClearConversation}
              className="bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
              icon={ClearOutlined}
            >
              Clear Chat
            </TremorButton>
          </div>
        )}
        
        {/* Show warning if variables aren't filled */}
        {extractedVariables.length > 0 && !allVariablesFilled && (
          <div className="mb-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-start gap-2">
              <span className="text-yellow-600 text-sm">⚠️</span>
              <div className="flex-1">
                <p className="text-sm text-yellow-800 font-medium mb-1">
                  Please fill in all template variables above
                </p>
                <p className="text-xs text-yellow-700">
                  Missing:{" "}
                  {extractedVariables
                    .filter((varName) => !variables[varName] || variables[varName].trim() === "")
                    .map((varName) => `{{${varName}}}`)
                    .join(", ")}
                </p>
              </div>
            </div>
          </div>
        )}
        
        <div className="flex items-center gap-2">
          <div className="flex items-center flex-1 bg-white border border-gray-300 rounded-xl px-3 py-1 min-h-[44px]">
            <TextArea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message... (Shift+Enter for new line)"
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

            <TremorButton
              onClick={handleSendMessage}
              disabled={
                isLoading || !inputMessage.trim() || (extractedVariables.length > 0 && !allVariablesFilled)
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
            >
              Cancel
            </TremorButton>
          )}
        </div>
      </div>
    </div>
  );
};

export default ConversationPanel;

