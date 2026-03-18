import { LoadingOutlined, RobotOutlined } from "@ant-design/icons";
import { Text } from "@tremor/react";
import { Spin } from "antd";
import React from "react";
import type { MCPEvent } from "../../mcp_tools/types";
import type { CodeInterpreterResult } from "../llm_calls/code_interpreter_handler";
import ChatMessageBubble from "./ChatMessageBubble";
import MCPEventsDisplay from "./MCPEventsDisplay";
import { EndpointType } from "./mode_endpoint_mapping";
import { MessageType } from "./types";

interface ChatMessageListProps {
  chatHistory: MessageType[];
  endpointType: string;
  mcpEvents: MCPEvent[];
  codeInterpreterResult: CodeInterpreterResult | null;
  isLoading: boolean;
  accessToken: string;
  chatEndRef: React.RefObject<HTMLDivElement | null>;
}

const antIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;

function ChatMessageList({
  chatHistory,
  endpointType,
  mcpEvents,
  codeInterpreterResult,
  isLoading,
  accessToken,
  chatEndRef,
}: ChatMessageListProps) {
  return (
    <div className="flex-1 overflow-auto p-4 pb-0">
      {chatHistory.length === 0 && (
        <div className="h-full flex flex-col items-center justify-center text-gray-400">
          <RobotOutlined style={{ fontSize: "48px", marginBottom: "16px" }} />
          <Text>Start a conversation, generate an image, or handle audio</Text>
        </div>
      )}

      {chatHistory.map((message, index) => (
        <ChatMessageBubble
          key={index}
          message={message}
          index={index}
          isLastMessage={index === chatHistory.length - 1}
          endpointType={endpointType}
          mcpEvents={mcpEvents}
          codeInterpreterResult={codeInterpreterResult}
          accessToken={accessToken}
        />
      ))}

      {/* Show MCP events during loading if no assistant message exists yet */}
      {isLoading &&
        mcpEvents.length > 0 &&
        (endpointType === EndpointType.RESPONSES || endpointType === EndpointType.CHAT) &&
        chatHistory.length > 0 &&
        chatHistory[chatHistory.length - 1].role === "user" && (
          <div className="text-left mb-4">
            <div
              className="inline-block max-w-[80%] rounded-lg shadow-sm p-3.5 px-4"
              style={{
                backgroundColor: "#ffffff",
                border: "1px solid #f0f0f0",
                textAlign: "left",
              }}
            >
              <div className="flex items-center gap-2 mb-1.5">
                <div
                  className="flex items-center justify-center w-6 h-6 rounded-full mr-1"
                  style={{
                    backgroundColor: "#f5f5f5",
                  }}
                >
                  <RobotOutlined style={{ fontSize: "12px", color: "#4b5563" }} />
                </div>
                <strong className="text-sm capitalize">Assistant</strong>
              </div>
              <MCPEventsDisplay events={mcpEvents} />
            </div>
          </div>
        )}

      {isLoading && (
        <div className="flex justify-center items-center my-4">
          <Spin indicator={antIcon} />
        </div>
      )}
      <div ref={chatEndRef} style={{ height: "1px" }} />
    </div>
  );
}

export default ChatMessageList;
