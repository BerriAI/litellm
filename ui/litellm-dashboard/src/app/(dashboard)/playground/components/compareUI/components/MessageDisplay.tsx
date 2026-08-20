import { Bot, Loader2, UserRound } from "lucide-react";
import React from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";

import { useSyntaxTheme } from "@/hooks/useSyntaxTheme";
import ChatImageRenderer from "../../chat_ui/ChatImageRenderer";
import ReasoningContent from "@/components/chat_ui/ReasoningContent";
import ResponseMetrics from "@/components/chat_ui/ResponseMetrics";
import { SearchResultsDisplay } from "../../chat_ui/SearchResultsDisplay";
import type { MessageType } from "@/components/chat_ui/types";

interface MessageDisplayProps {
  messages: MessageType[];
  isLoading: boolean;
}

export function MessageDisplay({ messages, isLoading }: MessageDisplayProps) {
  const syntaxTheme = useSyntaxTheme(coy);
  if (messages.length === 0) {
    return <div className="h-full" />;
  }

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
      className="whitespace-pre-wrap wrap-break-word"
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
              <code className={`${className} px-1.5 py-0.5 rounded-sm bg-muted text-sm font-mono`} {...props}>
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
              <div className="space-y-2 min-w-0">
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-info/15 text-info">
                    <UserRound size={16} />
                  </div>
                  <div className="text-sm font-semibold text-foreground">You</div>
                </div>
                {renderMessageBody(block.user)}
              </div>
            )}

            <div className="border-t border-border" />

            {assistantMessage ? (
              <div className="space-y-3 min-w-0">
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
                    <Bot size={16} />
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-foreground">{displayModel}</span>
                    {assistantMessage.toolName && (
                      <span className="rounded-sm bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                        {assistantMessage.toolName}
                      </span>
                    )}
                  </div>
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
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 size={18} className="animate-spin" />
                <span>Generating response...</span>
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">Waiting for a response...</div>
            )}
          </div>
        );
      })}
      {isLoading && conversationBlocks.length === 0 && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 size={18} className="animate-spin" />
          <span>Generating response...</span>
        </div>
      )}
    </div>
  );
}
