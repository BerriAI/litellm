import React from "react";
import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import ResponseMetrics from "../../../playground/chat_ui/ResponseMetrics";
import { Message } from "./types";
import { cn } from "@/lib/utils";

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === "user";
  return (
    <div
      className={cn(
        "mb-4 flex",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      <div
        className={cn(
          "max-w-[85%] rounded-lg shadow-sm p-3.5 px-4 border",
          isUser
            ? "bg-blue-50 border-blue-100 dark:bg-blue-950/40 dark:border-blue-900"
            : "bg-background border-border",
        )}
      >
        <div className="flex items-center gap-2 mb-1.5">
          <div
            className={cn(
              "flex items-center justify-center w-6 h-6 rounded-full mr-1",
              isUser
                ? "bg-blue-100 dark:bg-blue-950"
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
          {!isUser && message.model && (
            <span className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground font-normal">
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
          {!isUser ? (
            <ReactMarkdown
              components={{
                code({
                  inline,
                  className,
                  children,
                  ...props
                }: React.ComponentPropsWithoutRef<"code"> & {
                  inline?: boolean;
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  node?: any;
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
                      className={`${className} px-1.5 py-0.5 rounded bg-muted text-sm font-mono`}
                      style={{ wordBreak: "break-word" }}
                      {...props}
                    >
                      {children}
                    </code>
                  );
                },
                pre: ({ ...props }) => (
                  <pre
                    style={{ overflowX: "auto", maxWidth: "100%" }}
                    {...props}
                  />
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          ) : (
            <div className="whitespace-pre-wrap">{message.content}</div>
          )}

          {!isUser &&
            (message.timeToFirstToken ||
              message.totalLatency ||
              message.usage) && (
              <ResponseMetrics
                timeToFirstToken={message.timeToFirstToken}
                totalLatency={message.totalLatency}
                usage={message.usage}
              />
            )}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;
