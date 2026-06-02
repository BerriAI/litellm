"use client";

/**
 * MessageBubble — one entry in the conversation pane.
 *
 * Renders user/assistant/tool/system messages with role-specific styling.
 * Tool calls are rendered separately (ToolCallCard); this is for plain
 * text content from the conversation list and assistant_message events.
 */
import { Typography, Tag } from "antd";
import type { CloudAgentConversationRole } from "@/types/cloud-agents";

const { Paragraph, Text } = Typography;

const ROLE_TO_LABEL: Record<CloudAgentConversationRole, string> = {
  user: "You",
  assistant: "Agent",
  tool: "Tool",
  system: "System",
};

const ROLE_TO_COLOR: Record<CloudAgentConversationRole, string> = {
  user: "blue",
  assistant: "purple",
  tool: "default",
  system: "default",
};

interface MessageBubbleProps {
  role: CloudAgentConversationRole;
  content: string;
  timestamp?: string;
}

export default function MessageBubble({ role, content, timestamp }: MessageBubbleProps) {
  return (
    <div className="mb-3 flex flex-col gap-1" data-testid={`message-bubble-${role}`}>
      <div className="flex items-center gap-2">
        <Tag color={ROLE_TO_COLOR[role]} className="!m-0 !text-xs">
          {ROLE_TO_LABEL[role]}
        </Tag>
        {timestamp && (
          <Text type="secondary" className="!text-xs">
            {timestamp}
          </Text>
        )}
      </div>
      <Paragraph className="!mb-0 whitespace-pre-wrap !text-sm">{content}</Paragraph>
    </div>
  );
}
