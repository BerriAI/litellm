"use client";

import { ToolOutlined, CopyOutlined, CheckOutlined, EditOutlined } from "@ant-design/icons";
import { Collapse, Tooltip } from "antd";
import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import ReasoningContent from "../playground/chat_ui/ReasoningContent";
import MCPEventsDisplay, { MCPEvent } from "../playground/chat_ui/MCPEventsDisplay";
import { ChatMessage } from "./types";

const { Panel } = Collapse;

// Keys whose values must be redacted in tool args display
const REDACTED_KEY_PATTERNS = /token|key|secret|password|auth/i;

function redactSensitiveValues(obj: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (REDACTED_KEY_PATTERNS.test(k)) {
      result[k] = "[redacted]";
    } else if (Array.isArray(v)) {
      result[k] = v.map((item) =>
        item !== null && typeof item === "object" && !Array.isArray(item)
          ? redactSensitiveValues(item as Record<string, unknown>)
          : item,
      );
    } else if (v !== null && typeof v === "object") {
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

// Shared markdown code renderer matching ReasoningContent style.
// react-markdown v9 removed the `inline` prop; detect fenced blocks via language className.
function MarkdownCodeRenderer({
  node,
  className,
  children,
  ...props
}: React.ComponentPropsWithoutRef<"code"> & { node?: unknown }) {
  const match = /language-(\w+)/.exec(className || "");
  return match ? (
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
  onEdit?: (messageId: string, newContent: string) => void;
  isStreaming?: boolean;
}

function UserBubble({ message, onEdit, isStreaming }: UserBubbleProps) {
  const [hovered, setHovered] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState(message.content);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (editing && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.selectionStart = textareaRef.current.value.length;
    }
  }, [editing]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${ta.scrollHeight}px`;
  }, [editValue, editing]);

  const handleSave = () => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== message.content && onEdit) {
      onEdit(message.id, trimmed);
    }
    setEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSave();
    }
    if (e.key === "Escape") {
      setEditValue(message.content);
      setEditing(false);
    }
  };

  if (editing) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
        <div style={{
          width: "72%",
          background: "#fff",
          border: "1.5px solid #1677ff",
          borderRadius: 12,
          overflow: "hidden",
          boxShadow: "0 0 0 3px rgba(22,119,255,0.1)",
        }}>
          <textarea
            ref={textareaRef}
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onKeyDown={handleKeyDown}
            style={{
              width: "100%",
              padding: "10px 14px",
              border: "none",
              outline: "none",
              resize: "none",
              fontSize: 14,
              lineHeight: "1.6",
              color: "#111827",
              fontFamily: "inherit",
              background: "transparent",
              boxSizing: "border-box",
              minHeight: 40,
            }}
          />
          <div style={{
            display: "flex",
            justifyContent: "flex-end",
            gap: 8,
            padding: "6px 10px 8px",
            borderTop: "1px solid #f0f0f0",
          }}>
            <button
              onClick={() => { setEditValue(message.content); setEditing(false); }}
              style={{
                padding: "4px 12px", borderRadius: 6, border: "1px solid #d1d5db",
                background: "#fff", color: "#374151", fontSize: 13, cursor: "pointer",
              }}
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={!editValue.trim()}
              style={{
                padding: "4px 12px", borderRadius: 6, border: "none",
                background: editValue.trim() ? "#1677ff" : "#f3f4f6",
                color: editValue.trim() ? "#fff" : "#9ca3af",
                fontSize: 13, fontWeight: 500, cursor: editValue.trim() ? "pointer" : "not-allowed",
              }}
            >
              Save &amp; Send
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", width: "100%" }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div style={{ display: "flex", alignItems: "flex-end", gap: 6, maxWidth: "72%" }}>
        {/* Edit button — appears on hover, to the left of the bubble */}
        {hovered && !isStreaming && onEdit && (
          <Tooltip title="Edit message">
            <button
              onClick={() => { setEditValue(message.content); setEditing(true); }}
              style={{
                background: "none", border: "none", cursor: "pointer",
                padding: "4px 6px", borderRadius: 5,
                color: "#9ca3af", fontSize: 13, flexShrink: 0,
                display: "flex", alignItems: "center",
                transition: "color 0.15s",
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.color = "#6b7280"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.color = "#9ca3af"; }}
            >
              <EditOutlined />
            </button>
          </Tooltip>
        )}
        <div
          style={{
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
      </div>
      <span style={{ fontSize: 11, color: "#9ca3af", marginTop: 4 }}>
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
  mcpEvents?: MCPEvent[];
}

function AssistantBubble({
  message,
  isLastMessage,
  isStreaming,
  isTypingIndicator,
  mcpEvents,
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

      {/* MCP tool events (list tools, tool calls, results) */}
      {mcpEvents && mcpEvents.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <MCPEventsDisplay events={mcpEvents} />
        </div>
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
          remarkPlugins={[remarkGfm]}
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

      <CopyButton text={mainContent} />
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {
      // clipboard not available (non-HTTPS or permission denied) — silently no-op
    });
  };

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 6 }}>
      <Tooltip title={copied ? "Copied!" : "Copy"}>
        <button
          onClick={handleCopy}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            padding: "4px 6px",
            borderRadius: 5,
            color: copied ? "#52c41a" : "#9ca3af",
            fontSize: 13,
            display: "flex",
            alignItems: "center",
            gap: 4,
            transition: "color 0.15s",
          }}
          onMouseEnter={(e) => {
            if (!copied) (e.currentTarget as HTMLButtonElement).style.color = "#6b7280";
          }}
          onMouseLeave={(e) => {
            if (!copied) (e.currentTarget as HTMLButtonElement).style.color = "#9ca3af";
          }}
        >
          {copied ? <CheckOutlined /> : <CopyOutlined />}
        </button>
      </Tooltip>
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
  onEditMessage?: (messageId: string, newContent: string) => void;
  mcpEvents?: MCPEvent[];
}

const ChatMessages: React.FC<Props> = ({ messages, isStreaming, onEditMessage, mcpEvents }) => {
  // Scrolling is managed by ChatPage.tsx (scroll lock during streaming,
  // scroll-to-bottom on new message). No auto-scroll here.

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
          return <UserBubble key={msg.id} message={msg} onEdit={onEditMessage} isStreaming={isStreaming} />;
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
            mcpEvents={isLastMessage ? mcpEvents : undefined}
          />
        );
      })}

    </div>
  );
};

export default ChatMessages;
