import React from "react";
import { LoadingOutlined } from "@ant-design/icons";
import { Spin } from "antd";
import EmptyState from "./EmptyState";
import MessageBubble from "./MessageBubble";
import { Message } from "./types";

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
  hasVariables: boolean;
  messagesEndRef: React.RefObject<HTMLDivElement>;
}

const MessageList: React.FC<MessageListProps> = ({
  messages,
  isLoading,
  hasVariables,
  messagesEndRef,
}) => {
  const antIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;

  return (
    <div className="flex-1 overflow-y-auto p-4 pb-0">
      {messages.length === 0 && <EmptyState hasVariables={hasVariables} />}

      {messages.map((message, index) => (
        <MessageBubble key={index} message={message} />
      ))}

      {isLoading && (
        <div className="flex justify-center items-center my-4">
          <Spin indicator={antIcon} />
        </div>
      )}
      <div ref={messagesEndRef} style={{ height: "1px" }} />
    </div>
  );
};

export default MessageList;

