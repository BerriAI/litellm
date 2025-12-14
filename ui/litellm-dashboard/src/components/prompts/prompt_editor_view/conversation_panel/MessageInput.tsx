import React from "react";
import { ArrowUpOutlined } from "@ant-design/icons";
import { Button as TremorButton } from "@tremor/react";
import { Input } from "antd";

const { TextArea } = Input;

interface MessageInputProps {
  inputMessage: string;
  isLoading: boolean;
  isDisabled: boolean;
  onInputChange: (value: string) => void;
  onSend: () => void;
  onKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onCancel: () => void;
}

const MessageInput: React.FC<MessageInputProps> = ({
  inputMessage,
  isLoading,
  isDisabled,
  onInputChange,
  onSend,
  onKeyDown,
  onCancel,
}) => {
  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center flex-1 bg-white border border-gray-300 rounded-xl px-3 py-1 min-h-[44px]">
        <TextArea
          value={inputMessage}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type your message... (Shift+Enter for new line)"
          disabled={isLoading}
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

        <TremorButton
          onClick={onSend}
          disabled={isDisabled}
          className="flex-shrink-0 ml-2 !w-8 !h-8 !min-w-8 !p-0 !rounded-full !bg-blue-600 hover:!bg-blue-700 disabled:!bg-gray-300 !border-none !text-white disabled:!text-gray-500 !flex !items-center !justify-center"
        >
          <ArrowUpOutlined style={{ fontSize: "14px" }} />
        </TremorButton>
      </div>

      {isLoading && (
        <TremorButton
          onClick={onCancel}
          className="bg-red-50 hover:bg-red-100 text-red-600 border-red-200"
        >
          Cancel
        </TremorButton>
      )}
    </div>
  );
};

export default MessageInput;

