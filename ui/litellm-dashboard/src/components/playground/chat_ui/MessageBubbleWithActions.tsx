import React from "react";
import { Button, Input, Tooltip } from "antd";
import {
  CheckOutlined,
  CloseOutlined,
  EditOutlined,
  RedoOutlined,
  RobotOutlined,
  UserOutlined,
} from "@ant-design/icons";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import { MessageType } from "./types";

const { TextArea } = Input;

interface MessageBubbleWithActionsProps {
  message: MessageType;
  index: number;
  isEditing: boolean;
  editingContent: string;
  isLoading: boolean;
  onEditingContentChange: (content: string) => void;
  onStartEdit: (index: number) => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
  onRetry: (index: number) => void;
  children?: React.ReactNode;
}

export const MessageBubbleWithActions: React.FC<MessageBubbleWithActionsProps> = ({
  message,
  index,
  isEditing,
  editingContent,
  isLoading,
  onEditingContentChange,
  onStartEdit,
  onSaveEdit,
  onCancelEdit,
  onRetry,
  children,
}) => {
  const isUser = message.role === "user";

  return (
    <div className="group">
      <div className={`mb-4 ${isUser ? "text-right" : "text-left"}`}>
        <div
          className="inline-block max-w-[80%] rounded-lg shadow-sm p-3.5 px-4 relative"
          style={{
            backgroundColor: isUser ? "#f0f8ff" : "#ffffff",
            border: isUser ? "1px solid #e6f0fa" : "1px solid #f0f0f0",
            textAlign: "left",
          }}
        >
          <div className="flex items-center gap-2 mb-1.5 justify-between">
            <div className="flex items-center gap-2">
              <div
                className="flex items-center justify-center w-6 h-6 rounded-full mr-1"
                style={{
                  backgroundColor: isUser ? "#e6f0fa" : "#f5f5f5",
                }}
              >
                {isUser ? (
                  <UserOutlined style={{ fontSize: "12px", color: "#2563eb" }} />
                ) : (
                  <RobotOutlined style={{ fontSize: "12px", color: "#4b5563" }} />
                )}
              </div>
              <strong className="text-sm capitalize">{message.role}</strong>
              {!isUser && message.model && (
                <span className="text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-600 font-normal">
                  {message.model}
                </span>
              )}
            </div>

            {isUser && !isLoading && !isEditing && (
              <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <Tooltip title="Edit message">
                  <button
                    onClick={() => onStartEdit(index)}
                    className="p-1 hover:bg-blue-100 rounded text-blue-600 transition-colors"
                  >
                    <EditOutlined style={{ fontSize: "14px" }} />
                  </button>
                </Tooltip>
                <Tooltip title="Retry">
                  <button
                    onClick={() => onRetry(index)}
                    className="p-1 hover:bg-blue-100 rounded text-blue-600 transition-colors"
                  >
                    <RedoOutlined style={{ fontSize: "14px" }} />
                  </button>
                </Tooltip>
              </div>
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
            {isUser && isEditing ? (
              <div className="space-y-2">
                <TextArea
                  value={editingContent}
                  onChange={(e) => onEditingContentChange(e.target.value)}
                  autoSize={{ minRows: 2, maxRows: 10 }}
                  className="w-full"
                  style={{ fontSize: "14px", lineHeight: "1.5" }}
                />
                <div className="flex items-center gap-2 justify-end">
                  <Button size="small" onClick={onCancelEdit} icon={<CloseOutlined />}>
                    Cancel
                  </Button>
                  <Button
                    type="primary"
                    size="small"
                    onClick={onSaveEdit}
                    icon={<CheckOutlined />}
                    disabled={!editingContent.trim()}
                  >
                    Save & Submit
                  </Button>
                </div>
              </div>
            ) : (
              <>
                {children || (
                  <ReactMarkdown
                    components={{
                      code({ node, inline, className, children, ...props }: any) {
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
                      pre: ({ node, ...props }: any) => (
                        <pre style={{ overflowX: "auto", maxWidth: "100%" }} {...props} />
                      ),
                    }}
                  >
                    {typeof message.content === "string" ? message.content : ""}
                  </ReactMarkdown>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
