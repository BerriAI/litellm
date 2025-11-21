import React from "react";
import { Input, Button } from "antd";
import { ArrowUpOutlined } from "@ant-design/icons";

const { TextArea } = Input;

interface MessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled?: boolean;
}

export function MessageInput({ value, onChange, onSend, disabled }: MessageInputProps) {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) {
        onSend();
      }
    }
  };

  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center flex-1 bg-white border border-gray-300 rounded-xl px-3 py-1 min-h-[44px]">
        <TextArea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Shift+Enter for new line)"
          disabled={disabled}
          className="flex-1"
          autoSize={{ minRows: 1, maxRows: 4 }}
          style={{
            resize: "none",
            border: "none",
            boxShadow: "none",
            background: "transparent",
            padding: "4px 0",
            fontSize: "14px",
            lineHeight: "20px",
          }}
        />
        <Button onClick={onSend} disabled={disabled || !value.trim()} icon={<ArrowUpOutlined />} shape="circle" />
      </div>
    </div>
  );
}
