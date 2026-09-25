"use client";

import { Wrench, Copy, Check, Pencil } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { Bubble, BubbleContent } from "@/components/ui/Bubble";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";

import { useSyntaxTheme } from "@/hooks/useSyntaxTheme";
import ReasoningContent from "@/components/chat_ui/ReasoningContent";
import MCPEventsDisplay from "@/components/chat_ui/MCPEventsDisplay";
import ResponseMetrics from "@/components/chat_ui/ResponseMetrics";
import { ChatMessage } from "./types";

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

function MarkdownCodeRenderer({
  node,
  className,
  children,
  ...props
}: React.ComponentPropsWithoutRef<"code"> & { node?: unknown }) {
  const syntaxTheme = useSyntaxTheme(coy);
  const match = /language-(\w+)/.exec(className || "");
  return match ? (
    <SyntaxHighlighter {...props} style={syntaxTheme} language={match[1]} PreTag="div" className="rounded-md my-2">
      {String(children).replace(/\n$/, "")}
    </SyntaxHighlighter>
  ) : (
    <code className={`${className ?? ""} px-1.5 py-0.5 rounded bg-muted text-sm font-mono`} {...props}>
      {children}
    </code>
  );
}

const markdownComponents: Components = {
  code: MarkdownCodeRenderer,
  pre: ({ node, ...props }) => <pre className="max-w-full overflow-x-auto" {...props} />,
  p: ({ node, ...props }) => <p className="my-3 first:mt-0 last:mb-0" {...props} />,
  ul: ({ node, ...props }) => <ul className="my-3 list-disc space-y-1 pl-5" {...props} />,
  ol: ({ node, ...props }) => <ol className="my-3 list-decimal space-y-1 pl-5" {...props} />,
  table: ({ node, ...props }) => <Table {...props} />,
  thead: ({ node, ...props }) => <TableHeader {...props} />,
  tbody: ({ node, ...props }) => <TableBody {...props} />,
  tr: ({ node, ...props }) => <TableRow {...props} />,
  th: ({ node, ...props }) => <TableHead {...props} />,
  td: ({ node, ...props }) => <TableCell {...props} />,
};

const markdownWithoutImages: Components = {
  ...markdownComponents,
  img: ({ alt }) => <span>{alt || "Image omitted"}</span>,
};

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
  const showEditAction = hovered && !isStreaming && Boolean(onEdit);

  useEffect(() => {
    if (editing && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.selectionStart = textareaRef.current.value.length;
    }
  }, [editing]);

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
      <div className="flex flex-col items-end">
        <div className="w-[72%] bg-background border-2 border-primary rounded-xl overflow-hidden shadow-[0_0_0_3px_rgba(var(--primary)/0.1)]">
          <Textarea
            ref={textareaRef}
            aria-label="Edit message"
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full px-3.5 py-2.5 border-none outline-none resize-none text-sm leading-relaxed text-foreground font-[inherit] bg-transparent box-border min-h-[40px]"
          />
          <div className="flex justify-end gap-2 px-2.5 py-1.5 border-t">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setEditValue(message.content);
                setEditing(false);
              }}
            >
              Cancel
            </Button>
            <Button size="sm" onClick={handleSave} disabled={!editValue.trim()}>
              Save & Send
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="flex flex-col items-end w-full"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex min-w-0 max-w-[80%] items-end gap-1.5">
        {showEditAction && (
          <TooltipProvider delay={300}>
            <Tooltip>
              <TooltipTrigger
                render={
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    aria-label="Edit message"
                    onClick={() => {
                      setEditValue(message.content);
                      setEditing(true);
                    }}
                    className="text-muted-foreground hover:text-foreground shrink-0"
                  >
                    <Pencil className="size-3.5" />
                  </Button>
                }
              />
              <TooltipContent>
                <p>Edit message</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
        <Bubble variant="muted" align="end" className="max-w-full">
          <BubbleContent className="whitespace-pre-wrap">{message.content}</BubbleContent>
        </Bubble>
      </div>
      <span className="text-[11px] text-muted-foreground mt-1">{formatTimestamp(message.timestamp)}</span>
    </div>
  );
}

interface AssistantBubbleProps {
  message: ChatMessage;
  allowImages: boolean;
  isLastMessage: boolean;
  isStreaming: boolean;
  isTypingIndicator: boolean;
  mcpEvents?: ChatMessage["mcpEvents"];
}

function AssistantBubble({
  message,
  allowImages,
  isLastMessage,
  isStreaming,
  isTypingIndicator,
  mcpEvents,
}: AssistantBubbleProps) {
  const [reasoningKey, setReasoningKey] = useState(0);
  const prevStreamingRef = useRef<boolean>(isStreaming);

  useEffect(() => {
    if (prevStreamingRef.current && !isStreaming) {
      setReasoningKey((k) => k + 1);
    }
    prevStreamingRef.current = isStreaming;
  }, [isStreaming]);

  const showReasoningPlaceholder = isLastMessage && isStreaming && !message.reasoningContent;
  const showReasoning = !!message.reasoningContent || showReasoningPlaceholder;

  if (isTypingIndicator) {
    return (
      <div className="flex flex-col items-start">
        <div className="flex items-center gap-1 px-1 py-2.5">
          <TypingDots />
        </div>
      </div>
    );
  }

  let mainContent = message.content;
  let stoppedSuffix = false;
  if (mainContent.endsWith("[stopped]")) {
    mainContent = mainContent.slice(0, -"[stopped]".length);
    stoppedSuffix = true;
  }

  return (
    <Bubble variant="ghost" className="w-full items-start">
      {showReasoning &&
        (showReasoningPlaceholder ? (
          <ThinkingPlaceholder />
        ) : (
          <ReasoningContent key={reasoningKey} reasoningContent={message.reasoningContent!} />
        ))}

      <BubbleContent className="w-full text-foreground">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={allowImages ? markdownComponents : markdownWithoutImages}
        >
          {mainContent}
        </ReactMarkdown>
        {stoppedSuffix && <span className="text-muted-foreground italic"> [stopped]</span>}
      </BubbleContent>

      <CopyButton text={mainContent} />
      {mcpEvents && mcpEvents.length > 0 && (
        <div className="mt-2 max-w-full">
          <MCPEventsDisplay events={mcpEvents} />
        </div>
      )}

      <ResponseMetrics
        timeToFirstToken={message.timeToFirstToken}
        totalLatency={message.totalLatency}
        usage={message.usage}
      />
    </Bubble>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(() => {});
  };

  return (
    <div className="flex items-center gap-1 mt-1.5">
      <TooltipProvider delay={300}>
        <Tooltip>
          <TooltipTrigger
            render={
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label={copied ? "Copied message" : "Copy message"}
                onClick={handleCopy}
                className={copied ? "text-success" : "text-muted-foreground hover:text-foreground"}
              >
                {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
              </Button>
            }
          />
          <TooltipContent>
            <p>{copied ? "Copied!" : "Copy"}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
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
      <div className="inline-flex items-center gap-1.5 px-2.5 mb-2 bg-muted/50 border rounded-lg text-xs text-muted-foreground">
        <span className="chat-thinking-text py-1">Thinking...</span>
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
          background-color: var(--color-muted-foreground);
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
  const redactedArgs = message.toolArgs ? redactSensitiveValues(message.toolArgs) : undefined;
  const [open, setOpen] = useState(false);

  return (
    <div className="max-w-[80%]">
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger className="flex items-center gap-1.5 text-[13px] px-3 py-2 border rounded-lg bg-muted/50 hover:bg-muted transition-colors w-full text-left">
          <Wrench className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="font-medium text-foreground">{message.toolName ?? "Tool call"}</span>
        </CollapsibleTrigger>
        <CollapsibleContent className="border border-t-0 rounded-b-lg px-3 py-2 bg-muted/30">
          {redactedArgs !== undefined && (
            <div className={message.toolResult ? "mb-3" : ""}>
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                Arguments
              </div>
              <pre className="m-0 p-2 bg-muted rounded-md text-xs font-mono whitespace-pre-wrap break-words text-foreground">
                {JSON.stringify(redactedArgs, null, 2)}
              </pre>
            </div>
          )}
          {message.toolResult && (
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                Result
              </div>
              <div className="text-[13px] text-foreground whitespace-pre-wrap break-words font-mono">
                {message.toolResult}
              </div>
            </div>
          )}
        </CollapsibleContent>
      </Collapsible>
      <div className="text-[11px] text-muted-foreground mt-1">{formatTimestamp(message.timestamp)}</div>
    </div>
  );
}

interface Props {
  messages: ChatMessage[];
  isStreaming: boolean;
  onEditMessage?: (messageId: string, newContent: string) => void;
}

interface ChatMessageContentProps {
  message: ChatMessage;
  allowImages?: boolean;
  isStreaming?: boolean;
  isLastMessage?: boolean;
  onEditMessage?: (messageId: string, newContent: string) => void;
}

export function ChatMessageContent({
  message,
  allowImages = true,
  isStreaming = false,
  isLastMessage = true,
  onEditMessage,
}: ChatMessageContentProps) {
  if (message.role === "user") {
    return <UserBubble message={message} onEdit={onEditMessage} isStreaming={isStreaming} />;
  }

  if (message.role === "tool") {
    return <ToolCard message={message} />;
  }

  return (
    <AssistantBubble
      message={message}
      allowImages={allowImages}
      isLastMessage={isLastMessage}
      isStreaming={isStreaming}
      isTypingIndicator={isLastMessage && isStreaming && message.content === ""}
      mcpEvents={message.mcpEvents}
    />
  );
}

const ChatMessages: React.FC<Props> = ({ messages, isStreaming, onEditMessage }) => {
  return (
    <div className="flex flex-col gap-4">
      {messages.map((message, index) => (
        <ChatMessageContent
          key={message.id}
          message={message}
          isLastMessage={index === messages.length - 1}
          isStreaming={isStreaming}
          onEditMessage={onEditMessage}
        />
      ))}
    </div>
  );
};

export default ChatMessages;
