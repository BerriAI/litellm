import { Bot, User } from "lucide-react";
import React from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";

import { useSyntaxTheme } from "@/hooks/useSyntaxTheme";
import { CodeInterpreterResult } from "@/components/llm_calls/code_interpreter_handler";
import A2AMetrics from "./A2AMetrics";
import AudioRenderer from "./AudioRenderer";
import ChatImageRenderer from "./ChatImageRenderer";
import CodeInterpreterOutput from "./CodeInterpreterOutput";
import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import MCPEventsDisplay from "@/components/chat_ui/MCPEventsDisplay";
import type { MCPEvent } from "@/components/mcp_tools/types";
import ReasoningContent from "@/components/chat_ui/ReasoningContent";
import ResponseMetrics from "@/components/chat_ui/ResponseMetrics";
import ResponsesImageRenderer from "./ResponsesImageRenderer";
import { SearchResultsDisplay } from "./SearchResultsDisplay";
import { MessageType } from "@/components/chat_ui/types";

interface ChatMessageBubbleProps {
  message: MessageType;
  /** Whether this is the last message in the chat history. */
  isLastMessage: boolean;
  endpointType: EndpointType;
  /** MCP events to display on the last assistant message. */
  mcpEvents: MCPEvent[];
  /** Code interpreter result to display on the last assistant message. */
  codeInterpreterResult: CodeInterpreterResult | null;
  /** API key used to fetch code interpreter file downloads. */
  accessToken: string;
}

function ChatMessageBubble({
  message,
  isLastMessage,
  endpointType,
  mcpEvents,
  codeInterpreterResult,
  accessToken,
}: ChatMessageBubbleProps) {
  const syntaxTheme = useSyntaxTheme(coy);
  const isUser = message.role === "user";

  return (
    <div className={`mb-4 min-w-0 ${isUser ? "text-right" : "text-left"}`}>
      <div
        className={`inline-block min-w-0 max-w-[92%] overflow-hidden rounded-lg border p-3 text-left text-card-foreground shadow-xs sm:max-w-[85%] sm:px-4 ${
          isUser ? "border-info/20 bg-info/10" : "border-border bg-card"
        }`}
      >
        {/* Header: role icon + name + model badge */}
        <div className="mb-1.5 flex min-w-0 items-center gap-2">
          <div
            className={`flex items-center justify-center w-6 h-6 rounded-full mr-1 ${
              isUser ? "bg-info/20" : "bg-muted"
            }`}
          >
            {isUser ? (
              <User className="size-3 text-info" aria-hidden="true" />
            ) : (
              <Bot className="size-3 text-muted-foreground" aria-hidden="true" />
            )}
          </div>
          <strong className="text-sm capitalize">{message.role}</strong>
          {message.role === "assistant" && message.model && (
            <span className="max-w-48 truncate rounded-sm bg-muted px-2 py-0.5 text-xs font-normal text-muted-foreground sm:max-w-80">
              {message.model}
            </span>
          )}
        </div>

        {/* Reasoning content (chain-of-thought) */}
        {message.reasoningContent && <ReasoningContent reasoningContent={message.reasoningContent} />}

        {/* MCP events at the start of the last assistant message */}
        {message.role === "assistant" &&
          isLastMessage &&
          mcpEvents.length > 0 &&
          (endpointType === EndpointType.RESPONSES || endpointType === EndpointType.CHAT) && (
            <div className="mb-3">
              <MCPEventsDisplay events={mcpEvents} />
            </div>
          )}

        {/* Search results */}
        {message.role === "assistant" && message.searchResults && (
          <SearchResultsDisplay searchResults={message.searchResults} />
        )}

        {/* Code Interpreter output for the last assistant message */}
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

        {/* Message body */}
        <div
          className="whitespace-pre-wrap wrap-break-word max-w-full message-content"
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
              className="max-w-full rounded-md border border-border shadow-xs"
              style={{ maxHeight: "500px" }}
            />
          ) : message.isAudio ? (
            <AudioRenderer message={message} />
          ) : (
            <>
              {/* Attached image for user messages based on endpoint */}
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
                        {...props}
                        style={syntaxTheme}
                        language={match[1]}
                        PreTag="div"
                        className="rounded-md my-2"
                        wrapLines={true}
                        wrapLongLines={true}
                      >
                        {String(children).replace(/\n$/, "")}
                      </SyntaxHighlighter>
                    ) : (
                      <code
                        className={`${className} px-1.5 py-0.5 rounded-sm bg-muted text-sm font-mono`}
                        style={{ wordBreak: "break-word" }}
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  },
                  pre: ({ node, ...props }) => <pre style={{ overflowX: "auto", maxWidth: "100%" }} {...props} />,
                }}
              >
                {typeof message.content === "string" ? message.content : ""}
              </ReactMarkdown>

              {/* Generated image from chat completions */}
              {message.image && (
                <div className="mt-3">
                  <img
                    src={message.image.url}
                    alt="Generated image"
                    className="max-w-full rounded-md border border-border shadow-xs"
                    style={{ maxHeight: "500px" }}
                  />
                </div>
              )}
            </>
          )}

          {/* Response metrics */}
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

          {/* A2A Metrics */}
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
  );
}

export default ChatMessageBubble;
