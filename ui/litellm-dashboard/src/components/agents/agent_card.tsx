import React from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { AlertTriangle, Copy, Key, Trash2 } from "lucide-react";
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

  const statusDot = hasKey ? (
    <span className="inline-flex items-center gap-1.5 text-sm">
      <span className="h-2 w-2 rounded-full bg-emerald-500" />
      <span className="text-emerald-700 dark:text-emerald-400">Active</span>
    </span>
  ) : (
    <span className="inline-flex items-center gap-1.5 text-sm">
      <span className="h-2 w-2 rounded-full bg-amber-500" />
      <span className="text-amber-700 dark:text-amber-400">Needs Setup</span>
    </span>
  );

  const copyToClipboard = (e: React.MouseEvent, text: string) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text);
  };

  return (
    <Card
      className="h-full flex flex-col cursor-pointer hover:shadow-md transition-shadow p-4"
      onClick={() => onAgentClick(agent.agent_id)}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-foreground truncate">
              {agent.agent_name}
            </span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={(e) => copyToClipboard(e, agent.agent_id)}
                    className="cursor-pointer text-muted-foreground hover:text-primary shrink-0"
                    aria-label="Copy Agent ID"
                  >
                    <Copy className="h-3 w-3" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>Copy Agent ID</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <div className="mt-1">{statusDot}</div>
        </div>
        {isAdmin && onDeleteClick && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Delete agent"
                  className="shrink-0 -mr-1 text-destructive hover:text-destructive hover:bg-destructive/10 h-7 w-7"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteClick(agent.agent_id, agent.agent_name);
                  }}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Delete agent</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
      <p className="text-sm text-muted-foreground line-clamp-2 flex-1 mb-3">
        {description}
      </p>
      {url && (
        <p className="text-xs text-muted-foreground truncate mb-2" title={url}>
          {url}
        </p>
      )}
      <div className="mt-auto pt-3 border-t border-border text-xs">
        {hasKey ? (
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <Key className="h-3 w-3" />
            <span>
              {keyInfo?.key_alias || keyInfo?.token_prefix || "Key assigned"}
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 text-amber-600 dark:text-amber-400">
            <AlertTriangle className="h-3 w-3" />
            <span>No key assigned</span>
          </div>
        )}
      </div>
    </Card>
  );
};

export default AgentCard;
