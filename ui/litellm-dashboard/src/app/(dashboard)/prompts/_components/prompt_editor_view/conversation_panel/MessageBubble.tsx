import React from "react";
import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";

import { useSyntaxTheme } from "@/hooks/useSyntaxTheme";
import ResponseMetrics from "@/components/chat_ui/ResponseMetrics";
import { Message } from "./types";

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const syntaxTheme = useSyntaxTheme(coy);
  return (
    <div className={`mb-4 flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-lg border border-border p-3.5 px-4 shadow-xs ${
          message.role === "user" ? "bg-accent" : "bg-card"
        }`}
      >
        <div className="flex items-center gap-2 mb-1.5">
          <div
            className={`flex h-6 w-6 items-center justify-center rounded-full mr-1 ${
              message.role === "user" ? "bg-primary/10" : "bg-muted"
            }`}
          >
            {message.role === "user" ? (
              <User className="size-3 text-primary" aria-hidden="true" />
            ) : (
              <Bot className="size-3 text-muted-foreground" aria-hidden="true" />
            )}
          </div>
          <strong className="text-sm capitalize">{message.role}</strong>
          {message.role === "assistant" && message.model && (
            <span className="text-xs px-2 py-0.5 rounded-sm bg-muted text-muted-foreground font-normal">
              {message.model}
            </span>
          )}
        </div>

        <div
          className="whitespace-pre-wrap wrap-break-word max-w-full message-content"
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
              {message.content}
            </ReactMarkdown>
          ) : (
            <div className="whitespace-pre-wrap">{message.content}</div>
          )}

          {message.role === "assistant" && (message.timeToFirstToken || message.totalLatency || message.usage) && (
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
