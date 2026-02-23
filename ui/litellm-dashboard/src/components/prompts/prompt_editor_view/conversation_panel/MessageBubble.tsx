import React from "react";
import { RobotOutlined, UserOutlined } from "@ant-design/icons";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import ResponseMetrics from "../../../playground/chat_ui/ResponseMetrics";
import { Message } from "./types";

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  return (
    <div className={`mb-4 flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
      <div
        className="max-w-[85%] rounded-lg shadow-sm p-3.5 px-4"
        style={{
          backgroundColor: message.role === "user" ? "#f0f8ff" : "#ffffff",
          border: message.role === "user" ? "1px solid #e6f0fa" : "1px solid #f0f0f0",
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

        <div
          className="whitespace-pre-wrap break-words max-w-full message-content"
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
              {message.content}
            </ReactMarkdown>
          ) : (
            <div className="whitespace-pre-wrap">{message.content}</div>
          )}

          {message.role === "assistant" &&
            (message.timeToFirstToken || message.totalLatency || message.usage) && (
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

