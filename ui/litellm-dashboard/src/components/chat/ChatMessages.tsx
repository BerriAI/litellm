"use client";

import { ToolOutlined } from "@ant-design/icons";
import { Collapse } from "antd";
import React, { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import ReasoningContent from "../playground/chat_ui/ReasoningContent";
import { ChatMessage } from "./types";

const { Panel } = Collapse;

// Keys whose values must be redacted in tool args display
const REDACTED_KEY_PATTERNS = /token|key|secret|password|auth/i;

function redactSensitiveValues(obj: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (REDACTED_KEY_PATTERNS.test(k)) {
      result[k] = "[redacted]";
    } else if (v !== null && typeof v === "object" && !Array.isArray(v)) {
      result[k] = redactSensitiveValues(v as Record<string, unknown>);
    } else {
      result[k] = v;
    }
  }
  return result;
}

function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

// Shared markdown code renderer matching ReasoningContent style
function MarkdownCodeRenderer({
  node,
  inline,
  className,
  children,
  ...props
}: React.ComponentPropsWithoutRef<"code"> & { inline?: boolean; node?: unknown }) {
  const match = /language-(\w+)/.exec(className || "");
  return !inline && match ? (
    <SyntaxHighlighter
      style={coy as Record<string, React.CSSProperties>}
      language={match[1]}
      PreTag="div"
      className="rounded-md my-2"
      {...(props as Record<string, unknown>)}
    >
      {String(children).replace(/\n$/, "")}
    </SyntaxHighlighter>
  ) : (
    <code
      className={`${className ?? ""} px-1.5 py-0.5 rounded bg-gray-100 text-sm font-mono`}
      {...props}
    >
      {children}
    </code>
  );
}

// ------- Sub-components -------

interface UserBubbleProps {
  message: ChatMessage;
}

function UserBubble({ message }: UserBubbleProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
      <div
        style={{
          maxWidth: "72%",
          backgroundColor: "#f0f2f5",
          borderRadius: 16,
          padding: "10px 14px",
          fontSize: 14,
          lineHeight: "1.6",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          color: "#111827",
        }}
      >
        {message.content}
      </div>
      <span
        style={{
          fontSize: 11,
          color: "#9ca3af",
          marginTop: 4,
        }}
      >
        {formatTimestamp(message.timestamp)}
      </span>
    </div>
  );
}

interface AssistantBubbleProps {
  message: ChatMessage;
  isLastMessage: boolean;
  isStreaming: boolean;
  isTypingIndicator: boolean;
}

function AssistantBubble({
  message,
  isLastMessage,
  isStreaming,
  isTypingIndicator,
}: AssistantBubbleProps) {
  // Ref to control ReasoningContent collapse on streaming end.
  // ReasoningContent manages its own expanded state; we use a key to
  // remount it (collapsed by default) when streaming finishes.
  const reasoningKeyRef = useRef<number>(0);
  const prevStreamingRef = useRef<boolean>(isStreaming);

  useEffect(() => {
    if (prevStreamingRef.current && !isStreaming) {
      // Streaming just stopped — bump the key to remount ReasoningContent
      // with isExpanded default (false won't work since it starts expanded).
      // ReasoningContent always starts expanded on mount; we accept that
      // behaviour and leave collapse-on-finish as a best-effort remount.
      reasoningKeyRef.current += 1;
    }
    prevStreamingRef.current = isStreaming;
  }, [isStreaming]);

  const showReasoningPlaceholder =
    isLastMessage && isStreaming && !message.reasoningContent;

  const showReasoning =
    !!message.reasoningContent || showReasoningPlaceholder;

  if (isTypingIndicator) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 4, padding: "10px 4px" }}>
          <TypingDots />
        </div>
      </div>
    );
  }

  // Split content at trailing "[stopped]"
  let mainContent = message.content;
  let stoppedSuffix = false;
  if (mainContent.endsWith("[stopped]")) {
    mainContent = mainContent.slice(0, -"[stopped]".length);
    stoppedSuffix = true;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", maxWidth: "80%" }}>
      {showReasoning && (
        showReasoningPlaceholder ? (
          <ThinkingPlaceholder />
        ) : (
          <ReasoningContent
            key={reasoningKeyRef.current}
            reasoningContent={message.reasoningContent!}
          />
        )
      )}

      <div
        style={{
          fontSize: 14,
          lineHeight: "1.7",
          color: "#111827",
          wordBreak: "break-word",
        }}
      >
        <ReactMarkdown
          components={{
            code: MarkdownCodeRenderer as React.ComponentType<React.ComponentPropsWithoutRef<"code">>,
          }}
        >
          {mainContent}
        </ReactMarkdown>
        {stoppedSuffix && (
          <span style={{ color: "#9ca3af", fontStyle: "italic" }}> [stopped]</span>
        )}
      </div>

      <span style={{ fontSize: 11, color: "#9ca3af", marginTop: 4 }}>
        {formatTimestamp(message.timestamp)}
      </span>
    </div>
  );
}

function ThinkingPlaceholder() {
  return (
    <>
      <style>{`
        @keyframes thinking-pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
        .chat-thinking-text {
          animation: thinking-pulse 1.4s ease-in-out infinite;
        }
      `}</style>
      <div
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          padding: "4px 10px",
          marginBottom: 8,
          backgroundColor: "#f9fafb",
          border: "1px solid #e5e7eb",
          borderRadius: 8,
          fontSize: 12,
          color: "#6b7280",
        }}
      >
        <span className="chat-thinking-text">Thinking...</span>
      </div>
    </>
  );
}

function TypingDots() {
  return (
    <>
      <style>{`
        @keyframes chat-typing-bounce {
          0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
          30% { transform: translateY(-4px); opacity: 1; }
        }
        .chat-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background-color: #9ca3af;
          animation: chat-typing-bounce 1.2s ease-in-out infinite;
        }
        .chat-dot:nth-child(2) { animation-delay: 0.2s; }
        .chat-dot:nth-child(3) { animation-delay: 0.4s; }
      `}</style>
      <div className="chat-dot" />
      <div className="chat-dot" />
      <div className="chat-dot" />
    </>
  );
}

interface ToolCardProps {
  message: ChatMessage;
}

function ToolCard({ message }: ToolCardProps) {
  const redactedArgs =
    message.toolArgs ? redactSensitiveValues(message.toolArgs) : undefined;

  return (
    <div style={{ maxWidth: "80%" }}>
      <Collapse
        size="small"
        style={{
          backgroundColor: "#fafafa",
          border: "1px solid #e5e7eb",
          borderRadius: 8,
        }}
      >
        <Panel
          header={
            <span style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}>
              <ToolOutlined style={{ color: "#6b7280" }} />
              <span style={{ color: "#374151", fontWeight: 500 }}>
                {message.toolName ?? "Tool call"}
              </span>
            </span>
          }
          key="tool"
        >
          {redactedArgs !== undefined && (
            <div style={{ marginBottom: message.toolResult ? 12 : 0 }}>
              <div
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  color: "#9ca3af",
                  marginBottom: 4,
                }}
              >
                Arguments
              </div>
              <pre
                style={{
                  margin: 0,
                  padding: "8px 10px",
                  backgroundColor: "#f3f4f6",
                  borderRadius: 6,
                  fontSize: 12,
                  fontFamily:
                    'ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  color: "#374151",
                }}
              >
                {JSON.stringify(redactedArgs, null, 2)}
              </pre>
            </div>
          )}

          {message.toolResult && (
            <div>
              <div
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  color: "#9ca3af",
                  marginBottom: 4,
                }}
              >
                Result
              </div>
              <div
                style={{
                  fontSize: 13,
                  color: "#374151",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  fontFamily:
                    'ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                }}
              >
                {message.toolResult}
              </div>
            </div>
          )}
        </Panel>
      </Collapse>
      <div style={{ fontSize: 11, color: "#9ca3af", marginTop: 4 }}>
        {formatTimestamp(message.timestamp)}
      </div>
    </div>
  );
}

// ------- Main component -------

interface Props {
  messages: ChatMessage[];
  isStreaming: boolean;
}

const ChatMessages: React.FC<Props> = ({ messages, isStreaming }) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const lastIndex = messages.length - 1;
  const lastMsg = messages[lastIndex] ?? null;
  const isTypingIndicator =
    isStreaming &&
    lastMsg !== null &&
    lastMsg.role === "assistant" &&
    lastMsg.content === "";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {messages.map((msg, idx) => {
        const isLastMessage = idx === lastIndex;

        if (msg.role === "user") {
          return <UserBubble key={msg.id} message={msg} />;
        }

        if (msg.role === "tool") {
          return <ToolCard key={msg.id} message={msg} />;
        }

        // assistant
        return (
          <AssistantBubble
            key={msg.id}
            message={msg}
            isLastMessage={isLastMessage}
            isStreaming={isStreaming}
            isTypingIndicator={isLastMessage && isTypingIndicator}
          />
        );
      })}

      {/* Bottom sentinel for auto-scroll */}
      <div ref={bottomRef} />
    </div>
  );
};

export default ChatMessages;
