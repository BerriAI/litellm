import React from "react";
import { Card, Badge, Tooltip, Button } from "antd";
import { CopyOutlined, KeyOutlined, WarningOutlined, DeleteOutlined } from "@ant-design/icons";
import { Agent, AgentKeyInfo } from "./types";

interface AgentCardProps {
  agent: Agent;
  keyInfo?: AgentKeyInfo;
  onAgentClick: (agentId: string) => void;
  onDeleteClick?: (agentId: string, agentName: string) => void;
  accessToken: string | null;
  isAdmin: boolean;
  onAgentUpdated: () => void;
}

const AgentCard: React.FC<AgentCardProps> = ({
  agent,
  keyInfo,
  onAgentClick,
  onDeleteClick,
  isAdmin,
}) => {
  const description =
    agent.agent_card_params?.description || "No description";
  const url = agent.agent_card_params?.url;
  const hasKey = keyInfo?.has_key ?? false;
  const statusBadge = hasKey ? (
    <Badge status="success" text="Active" />
  ) : (
    <Badge status="warning" text="Needs Setup" />
  );

  const copyToClipboard = (e: React.MouseEvent, text: string) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text);
  };

  return (
    <Card
      hoverable
      className="h-full flex flex-col"
      styles={{
        body: { flex: 1, display: "flex", flexDirection: "column" },
      }}
      onClick={() => onAgentClick(agent.agent_id)}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-gray-900 truncate">
              {agent.agent_name}
            </span>
            <Tooltip title="Copy Agent ID">
              <CopyOutlined
                onClick={(e) => copyToClipboard(e, agent.agent_id)}
                className="cursor-pointer text-gray-400 hover:text-blue-500 text-xs shrink-0"
              />
            </Tooltip>
          </div>
          <div className="mt-1">{statusBadge}</div>
        </div>
        {isAdmin && onDeleteClick && (
          <Tooltip title="Delete agent">
            <Button
              type="text"
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                onDeleteClick(agent.agent_id, agent.agent_name);
              }}
              className="shrink-0 -mr-1"
            />
          </Tooltip>
        )}
      </div>
      <p className="text-sm text-gray-600 line-clamp-2 flex-1 mb-3">
        {description}
      </p>
      {url && (
        <p className="text-xs text-gray-500 truncate mb-2" title={url}>
          {url}
        </p>
      )}
      <div className="mt-auto pt-3 border-t border-gray-100 text-xs">
        {hasKey ? (
          <div className="flex items-center gap-1.5 text-gray-600">
            <KeyOutlined />
            <span>{keyInfo?.key_alias || keyInfo?.token_prefix || "Key assigned"}</span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 text-amber-600">
            <WarningOutlined />
            <span>No key assigned</span>
          </div>
        )}
      </div>
    </Card>
  );
};

export default AgentCard;
