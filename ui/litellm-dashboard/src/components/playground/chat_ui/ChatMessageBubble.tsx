import { Bot, User } from "lucide-react";
import { cn } from "@/lib/utils";
import React from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import { CodeInterpreterResult } from "../llm_calls/code_interpreter_handler";
import A2AMetrics from "./A2AMetrics";
import AudioRenderer from "./AudioRenderer";
import ChatImageRenderer from "./ChatImageRenderer";
import CodeInterpreterOutput from "./CodeInterpreterOutput";
import { EndpointType } from "./mode_endpoint_mapping";
import MCPEventsDisplay from "./MCPEventsDisplay";
import type { MCPEvent } from "../../mcp_tools/types";
import ReasoningContent from "./ReasoningContent";
import ResponseMetrics from "./ResponseMetrics";
import ResponsesImageRenderer from "./ResponsesImageRenderer";
import { SearchResultsDisplay } from "./SearchResultsDisplay";
import { MessageType } from "./types";

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
  const isUser = message.role === "user";

  return (
    <div className={cn("mb-4", isUser ? "text-right" : "text-left")}>
      <div
        className={cn(
          "inline-block max-w-[80%] rounded-lg shadow-sm p-3.5 px-4 text-left border",
          isUser
            ? "bg-blue-50 dark:bg-blue-950/30 border-blue-100 dark:border-blue-900"
            : "bg-background border-border",
        )}
      >
        {/* Header: role icon + name + model badge */}
        <div className="flex items-center gap-2 mb-1.5">
          <div
            className={cn(
              "flex items-center justify-center w-6 h-6 rounded-full mr-1",
              isUser
                ? "bg-blue-100 dark:bg-blue-950/60"
                : "bg-muted",
            )}
          >
            {isUser ? (
              <User className="h-3 w-3 text-blue-600 dark:text-blue-400" />
            ) : (
              <Bot className="h-3 w-3 text-muted-foreground" />
            )}
          </div>
          <strong className="text-sm capitalize">{message.role}</strong>
          {message.role === "assistant" && message.model && (
            <span className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground font-normal">
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
          className="whitespace-pre-wrap break-words max-w-full message-content"
          style={{
            wordWrap: "break-word",
            overflowWrap: "break-word",
            wordBreak: "break-word",
            hyphens: "auto",
          }}
        >
          {message.isImage ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={typeof message.content === "string" ? message.content : ""}
              alt="Generated image"
              className="max-w-full rounded-md border border-border shadow-sm"
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
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
                        className={`${className} px-1.5 py-0.5 rounded bg-muted text-sm font-mono break-words`}
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  },
                  pre: ({ ...props }) => (
                    <pre className="overflow-x-auto max-w-full" {...props} />
                  ),
                }}
              >
                {typeof message.content === "string" ? message.content : ""}
              </ReactMarkdown>

              {/* Generated image from chat completions */}
              {message.image && (
                <div className="mt-3">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={message.image.url}
                    alt="Generated image"
                    className="max-w-full rounded-md border border-border shadow-sm"
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
