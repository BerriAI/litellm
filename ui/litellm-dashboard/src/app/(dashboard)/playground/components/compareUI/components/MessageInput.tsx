import React from "react";
import ChatComposer from "../../chat_ui/ChatComposer";

interface MessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled?: boolean;
  hasAttachment?: boolean;
  uploadComponent?: React.ReactNode;
}

export function MessageInput({ value, onChange, onSend, disabled, hasAttachment, uploadComponent }: MessageInputProps) {
  const canSend = !disabled && (value.trim().length > 0 || Boolean(hasAttachment));

  return (
    <ChatComposer
      value={value}
      onChange={onChange}
      onSubmit={onSend}
      placeholder="Type your message... (Shift+Enter for new line)"
      disabled={disabled}
      submitDisabled={!canSend}
      tools={uploadComponent}
    />
  );
}
