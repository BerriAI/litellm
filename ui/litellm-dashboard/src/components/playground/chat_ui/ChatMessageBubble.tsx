import { RobotOutlined, UserOutlined } from "@ant-design/icons";
import React from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import type { MCPEvent } from "../../mcp_tools/types";
import type { CodeInterpreterResult } from "../llm_calls/code_interpreter_handler";
import A2AMetrics from "./A2AMetrics";
import AudioRenderer from "./AudioRenderer";
import ChatImageRenderer from "./ChatImageRenderer";
import CodeInterpreterOutput from "./CodeInterpreterOutput";
import MCPEventsDisplay from "./MCPEventsDisplay";
import { EndpointType } from "./mode_endpoint_mapping";
import ReasoningContent from "./ReasoningContent";
import ResponseMetrics from "./ResponseMetrics";
import ResponsesImageRenderer from "./ResponsesImageRenderer";
import { SearchResultsDisplay } from "./SearchResultsDisplay";
import { MessageType } from "./types";

interface ChatMessageBubbleProps {
  message: MessageType;
  index: number;
  isLastMessage: boolean;
  endpointType: string;
  mcpEvents: MCPEvent[];
  codeInterpreterResult: CodeInterpreterResult | null;
  accessToken: string;
}

function ChatMessageBubble({
  message,
  index,
  isLastMessage,
  endpointType,
  mcpEvents,
  codeInterpreterResult,
  accessToken,
}: ChatMessageBubbleProps) {
  return (
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
            isLastMessage &&
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
            isLastMessage &&
            codeInterpreterResult &&
            endpointType === EndpointType.RESPONSES && (
              <CodeInterpreterOutput
                code={codeInterpreterResult.code}
                containerId={codeInterpreterResult.containerId}
                annotations={codeInterpreterResult.annotations}
                accessToken={accessToken}
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
                      node?: unknown;
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
  );
}

export default ChatMessageBubble;
