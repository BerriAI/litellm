import { Bot, Loader2, UserRound, Edit2, RotateCw } from "lucide-react";
import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Button, Input } from "antd";
import ChatImageRenderer from "../../chat_ui/ChatImageRenderer";
import ReasoningContent from "../../chat_ui/ReasoningContent";
import ResponseMetrics from "../../chat_ui/ResponseMetrics";
import { SearchResultsDisplay } from "../../chat_ui/SearchResultsDisplay";
import type { MessageType } from "../../chat_ui/types";

const { TextArea } = Input;

interface MessageDisplayProps {
  messages: MessageType[];
  isLoading: boolean;
  onEditMessage?: (index: number, newContent: string) => void;
  onRetryMessage?: (index: number) => void;
}

export function MessageDisplay({ messages, isLoading, onEditMessage, onRetryMessage }: MessageDisplayProps) {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editedContent, setEditedContent] = useState<string>("");

  if (messages.length === 0) {
    return <div className="h-full" />;
  }

  const handleStartEdit = (index: number, content: string) => {
    setEditingIndex(index);
    setEditedContent(typeof content === "string" ? content : "");
  };

  const handleSaveEdit = (index: number) => {
    if (onEditMessage && editedContent.trim()) {
      onEditMessage(index, editedContent);
    }
    setEditingIndex(null);
    setEditedContent("");
  };

  const handleCancelEdit = () => {
    setEditingIndex(null);
    setEditedContent("");
  };

  const conversationBlocks: Array<{
    user?: MessageType;
    assistant?: MessageType;
  }> = [];
  let index = 0;
  while (index < messages.length) {
    const current = messages[index];
    if (current.role === "user") {
      const next = messages[index + 1];
      if (next?.role === "assistant") {
        conversationBlocks.push({
          user: current,
          assistant: next,
        });
        index += 2;
        continue;
      }
      conversationBlocks.push({
        user: current,
      });
    } else if (current.role === "assistant") {
      conversationBlocks.push({
        assistant: current,
      });
    }
    index += 1;
  }

  const renderMessageBody = (message: MessageType) => (
    <div
      className="whitespace-pre-wrap break-words"
      style={{
        wordWrap: "break-word",
        overflowWrap: "break-word",
        wordBreak: "break-word",
        hyphens: "auto",
      }}
    >
      <ChatImageRenderer message={message} />
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
              <code className={`${className} px-1.5 py-0.5 rounded bg-gray-100 text-sm font-mono`} {...props}>
                {children}
              </code>
            );
          },
          pre: ({ node, ...props }) => <pre style={{ overflowX: "auto", maxWidth: "100%" }} {...props} />,
        }}
      >
        {typeof message.content === "string" ? message.content : ""}
      </ReactMarkdown>
    </div>
  );

  return (
    <div className="flex flex-col gap-6 min-w-0 w-full p-4">
      {conversationBlocks.map((block, blockIndex) => {
        const assistantMessage = block.assistant;
        const displayModel = assistantMessage?.model || "Assistant";
        return (
          <div key={blockIndex} className="space-y-4">
            {block.user && (
              <div className="space-y-2 min-w-0 group relative">
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 text-blue-600">
                    <UserRound size={16} />
                  </div>
                  <div className="text-sm font-semibold text-gray-700">You</div>
                  {!isLoading && editingIndex === null && onEditMessage && block.user && (
                    <button
                      onClick={() => {
                        const userIndex = messages.findIndex((m) => m === block.user);
                        const content = typeof block.user?.content === "string" ? block.user.content : "";
                        handleStartEdit(userIndex, content);
                      }}
                      className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-md hover:bg-blue-50 text-blue-600"
                      title="Edit message"
                    >
                      <Edit2 size={14} />
                    </button>
                  )}
                </div>
                {editingIndex === messages.findIndex((m) => m === block.user) ? (
                  <div className="space-y-2">
                    <TextArea
                      value={editedContent}
                      onChange={(e) => setEditedContent(e.target.value)}
                      autoSize={{ minRows: 2, maxRows: 10 }}
                      className="w-full"
                    />
                    <div className="flex gap-2">
                      <Button
                        type="primary"
                        size="small"
                        onClick={() => handleSaveEdit(messages.findIndex((m) => m === block.user))}
                      >
                        Save & Resend
                      </Button>
                      <Button size="small" onClick={handleCancelEdit}>
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  renderMessageBody(block.user)
                )}
              </div>
            )}

            <div className="border-t border-gray-200" />

            {assistantMessage ? (
              <div className="space-y-3 min-w-0 group relative">
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gray-100 text-gray-600">
                    <Bot size={16} />
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-700">{displayModel}</span>
                    {assistantMessage.toolName && (
                      <span className="rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-600">
                        {assistantMessage.toolName}
                      </span>
                    )}
                  </div>
                  {!isLoading && editingIndex === null && onRetryMessage && block.user && (
                    <button
                      onClick={() => {
                        const assistantIndex = messages.findIndex((m) => m === assistantMessage);
                        onRetryMessage(assistantIndex);
                      }}
                      className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-md hover:bg-green-50 text-green-600"
                      title="Regenerate response"
                    >
                      <RotateCw size={14} />
                    </button>
                  )}
                </div>
                {assistantMessage.reasoningContent && (
                  <ReasoningContent reasoningContent={assistantMessage.reasoningContent} />
                )}
                {assistantMessage.searchResults && (
                  <SearchResultsDisplay searchResults={assistantMessage.searchResults} />
                )}
                {renderMessageBody(assistantMessage)}
                {(assistantMessage.timeToFirstToken || assistantMessage.totalLatency || assistantMessage.usage) && (
                  <ResponseMetrics
                    timeToFirstToken={assistantMessage.timeToFirstToken}
                    totalLatency={assistantMessage.totalLatency}
                    usage={assistantMessage.usage}
                    toolName={assistantMessage.toolName}
                  />
                )}
              </div>
            ) : isLoading && blockIndex === conversationBlocks.length - 1 ? (
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <Loader2 size={18} className="animate-spin" />
                <span>Generating response...</span>
              </div>
            ) : (
              <div className="text-sm text-gray-500">Waiting for a response...</div>
            )}
          </div>
        );
      })}
      {isLoading && conversationBlocks.length === 0 && (
        <div className="flex items-center gap-2 text-gray-500">
          <Loader2 size={18} className="animate-spin" />
          <span>Generating response...</span>
        </div>
      )}
    </div>
  );
}
