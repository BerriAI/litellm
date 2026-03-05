import React, { useState } from "react";
import { Button, Input, Popover, Tooltip } from "antd";
import { ApiOutlined, BorderOutlined, PaperClipOutlined, SendOutlined } from "@ant-design/icons";

interface Props {
  onSend: (text: string) => void;
  isStreaming: boolean;
  onStop: () => void;
  selectedMCPServers: string[];
  onMCPChange: (servers: string[]) => void;
  isLoadingModels: boolean;
  accessToken: string;
}

const ChatInputBar: React.FC<Props> = ({
  onSend,
  isStreaming,
  onStop,
  selectedMCPServers,
  onMCPChange,
  isLoadingModels,
  accessToken,
}) => {
  const [text, setText] = useState<string>("");
  const [mcpPopoverOpen, setMcpPopoverOpen] = useState<boolean>(false);

  const handleSend = () => {
    if (text.trim() === "" || isStreaming || isLoadingModels) return;
    onSend(text.trim());
    setText("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const mcpButtonLabel =
    selectedMCPServers.length > 0
      ? `MCP (${selectedMCPServers.length})`
      : "MCP";

  const mcpPopoverContent = (
    <div style={{ minWidth: 200 }}>
      {/* MCPConnectPicker - LIT-2170 */}
    </div>
  );

  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-end",
        gap: 8,
        padding: "12px 16px",
        borderTop: "1px solid #e5e7eb",
        backgroundColor: "#ffffff",
      }}
    >
      <Popover
        content={mcpPopoverContent}
        title="MCP Servers"
        trigger="click"
        open={mcpPopoverOpen}
        onOpenChange={setMcpPopoverOpen}
        placement="topLeft"
      >
        <Button icon={<ApiOutlined />}>
          {mcpButtonLabel}
        </Button>
      </Popover>

      <Tooltip title="Coming soon">
        <Button icon={<PaperClipOutlined />} disabled />
      </Tooltip>

      <Input.TextArea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Message..."
        autoSize={{ minRows: 1, maxRows: 5 }}
        style={{ flex: 1 }}
      />

      {isStreaming ? (
        <Button
          icon={<BorderOutlined />}
          onClick={onStop}
          type="primary"
          danger
        />
      ) : (
        <Button
          icon={<SendOutlined />}
          onClick={handleSend}
          type="primary"
          disabled={isStreaming || isLoadingModels || text.trim() === ""}
        />
      )}
    </div>
  );
};

export default ChatInputBar;
