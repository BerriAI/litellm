import {
  CheckOutlined,
  CloseOutlined,
  EditOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import { Button, Input, Tooltip } from "antd";
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
  onRetry: () => void;
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
  const canRetry = role === "assistant" && isLastAssistantMessage && !isLoading && !isSpecialMessage;

  if (isEditing) {
    return (
      <div className="mt-2" data-testid="edit-message-container">
        <TextArea
          ref={textAreaRef}
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onKeyDown={handleEditKeyDown}
          autoSize={{ minRows: 1, maxRows: 8 }}
          className="mb-2"
          style={{ fontSize: "14px" }}
        />
        <div className="flex gap-1.5 justify-end">
          <Tooltip title="Cancel (Esc)">
            <Button
              size="small"
              icon={<CloseOutlined />}
              onClick={handleCancelEdit}
              data-testid="cancel-edit-button"
            />
          </Tooltip>
          <Tooltip title="Save & Submit (Enter)">
            <Button
              size="small"
              type="primary"
              icon={<CheckOutlined />}
              onClick={handleConfirmEdit}
              disabled={editValue.trim() === ""}
              data-testid="confirm-edit-button"
            />
          </Tooltip>
        </div>
      </div>
    );
  }

  if (!canEdit && !canRetry) return null;

  return (
    <div
      className="message-actions mt-1 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity"
      data-testid="message-actions"
    >
      {canEdit && (
        <Tooltip title="Edit message">
          <button
            className="p-1 rounded hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors"
            onClick={handleStartEdit}
            data-testid="edit-message-button"
          >
            <EditOutlined style={{ fontSize: "13px" }} />
          </button>
        </Tooltip>
      )}
      {canRetry && (
        <Tooltip title="Retry">
          <button
            className="p-1 rounded hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors"
            onClick={onRetry}
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
