import {
  CheckOutlined,
  CloseOutlined,
  EditOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import { Input, Tooltip } from "antd";
import React, { useEffect, useRef, useState } from "react";

const { TextArea } = Input;

interface MessageActionsProps {
  role: string;
  content: string;
  messageIndex: number;
  isLastAssistantMessage: boolean;
  isLoading: boolean;
  isImage?: boolean;
  isAudio?: boolean;
  isEmbeddings?: boolean;
  onRetry: (messageIndex: number) => void;
  onEditSubmit: (messageIndex: number, newContent: string) => void;
}

const MessageActions: React.FC<MessageActionsProps> = ({
  role,
  content,
  messageIndex,
  isLastAssistantMessage,
  isLoading,
  isImage,
  isAudio,
  isEmbeddings,
  onRetry,
  onEditSubmit,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(content);
  const textAreaRef = useRef<any>(null);

  useEffect(() => {
    if (isEditing && textAreaRef.current) {
      const ta = textAreaRef.current?.resizableTextArea?.textArea;
      if (ta) {
        ta.focus();
        ta.setSelectionRange(ta.value.length, ta.value.length);
      }
    }
  }, [isEditing]);

  const handleStartEdit = () => {
    setEditValue(content);
    setIsEditing(true);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditValue(content);
  };

  const handleConfirmEdit = () => {
    if (editValue.trim() === "") return;
    setIsEditing(false);
    onEditSubmit(messageIndex, editValue);
  };

  const handleEditKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleConfirmEdit();
    }
    if (e.key === "Escape") {
      handleCancelEdit();
    }
  };

  const isSpecialMessage = isImage || isAudio || isEmbeddings;
  const canEdit = role === "user" && !isLoading && !isSpecialMessage;
  const canRetryUser = role === "user" && !isLoading && !isSpecialMessage;
  const canRetryAssistant = role === "assistant" && isLastAssistantMessage && !isLoading && !isSpecialMessage;

  if (isEditing) {
    return (
      <div className="mt-2 pt-2 border-t border-gray-100" data-testid="edit-message-container">
        <TextArea
          ref={textAreaRef}
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onKeyDown={handleEditKeyDown}
          autoSize={{ minRows: 1, maxRows: 8 }}
          style={{
            fontSize: "14px",
            borderRadius: "8px",
            border: "1px solid #d1d5db",
            padding: "8px 12px",
          }}
        />
        <div className="flex gap-2 justify-end mt-2">
          <button
            className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-md
                       text-gray-600 bg-gray-50 border border-gray-200
                       hover:bg-gray-100 hover:border-gray-300 transition-all"
            onClick={handleCancelEdit}
            data-testid="cancel-edit-button"
          >
            <CloseOutlined style={{ fontSize: "10px" }} />
            Cancel
          </button>
          <button
            className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-md
                       text-white bg-blue-600 border border-blue-600
                       hover:bg-blue-700 transition-all
                       disabled:opacity-40 disabled:cursor-not-allowed"
            onClick={handleConfirmEdit}
            disabled={editValue.trim() === ""}
            data-testid="confirm-edit-button"
          >
            <CheckOutlined style={{ fontSize: "10px" }} />
            Submit
          </button>
        </div>
      </div>
    );
  }

  if (!canEdit && !canRetryUser && !canRetryAssistant) return null;

  return (
    <div
      className="message-actions flex gap-0.5 mt-1.5 pt-1 opacity-0 group-hover:opacity-100 transition-opacity duration-150"
      data-testid="message-actions"
    >
      {canEdit && (
        <Tooltip title="Edit">
          <button
            className="inline-flex items-center justify-center w-7 h-7 rounded-md
                       text-gray-400 hover:text-gray-700 hover:bg-black/[0.05] transition-colors"
            onClick={handleStartEdit}
            data-testid="edit-message-button"
          >
            <EditOutlined style={{ fontSize: "13px" }} />
          </button>
        </Tooltip>
      )}
      {canRetryUser && (
        <Tooltip title="Resend">
          <button
            className="inline-flex items-center justify-center w-7 h-7 rounded-md
                       text-gray-400 hover:text-gray-700 hover:bg-black/[0.05] transition-colors"
            onClick={() => onRetry(messageIndex)}
            data-testid="retry-message-button"
          >
            <ReloadOutlined style={{ fontSize: "13px" }} />
          </button>
        </Tooltip>
      )}
      {canRetryAssistant && (
        <Tooltip title="Regenerate">
          <button
            className="inline-flex items-center justify-center w-7 h-7 rounded-md
                       text-gray-400 hover:text-gray-700 hover:bg-black/[0.05] transition-colors"
            onClick={() => onRetry(messageIndex)}
            data-testid="retry-message-button"
          >
            <ReloadOutlined style={{ fontSize: "13px" }} />
          </button>
        </Tooltip>
      )}
    </div>
  );
};

export default MessageActions;
