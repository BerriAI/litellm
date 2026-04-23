import React from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Clock,
  DollarSign,
  Hash,
  LogIn,
  LogOut,
  Lightbulb,
  Wrench,
} from "lucide-react";

export interface TokenUsage {
  completionTokens?: number;
  promptTokens?: number;
  totalTokens?: number;
  reasoningTokens?: number;
  cost?: number;
}

interface ResponseMetricsProps {
  timeToFirstToken?: number;
  totalLatency?: number;
  usage?: TokenUsage;
  toolName?: string;
}

const TooltipIconStat: React.FC<{
  tooltip: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}> = ({ tooltip, icon, children }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex items-center">
          {icon}
          <span>{children}</span>
        </div>
      </TooltipTrigger>
      <TooltipContent>{tooltip}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

const ResponseMetrics: React.FC<ResponseMetricsProps> = ({
  timeToFirstToken,
  totalLatency,
  usage,
  toolName,
}) => {
  if (!timeToFirstToken && !totalLatency && !usage) return null;

  return (
    <div className="response-metrics mt-2 pt-2 border-t border-border text-xs text-muted-foreground flex flex-wrap gap-3">
      {timeToFirstToken !== undefined && (
        <TooltipIconStat
          tooltip="Time to first token"
          icon={<Clock className="h-3 w-3 mr-1" />}
        >
          TTFT: {(timeToFirstToken / 1000).toFixed(2)}s
        </TooltipIconStat>
      )}

      {totalLatency !== undefined && (
        <TooltipIconStat
          tooltip="Total latency"
          icon={<Clock className="h-3 w-3 mr-1" />}
        >
          Total Latency: {(totalLatency / 1000).toFixed(2)}s
        </TooltipIconStat>
      )}

      {usage?.promptTokens !== undefined && (
        <TooltipIconStat
          tooltip="Prompt tokens"
          icon={<LogIn className="h-3 w-3 mr-1" />}
        >
          In: {usage.promptTokens}
        </TooltipIconStat>
      )}

      {usage?.completionTokens !== undefined && (
        <TooltipIconStat
          tooltip="Completion tokens"
          icon={<LogOut className="h-3 w-3 mr-1" />}
        >
          Out: {usage.completionTokens}
        </TooltipIconStat>
      )}

      {usage?.reasoningTokens !== undefined && (
        <TooltipIconStat
          tooltip="Reasoning tokens"
          icon={<Lightbulb className="h-3 w-3 mr-1" />}
        >
          Reasoning: {usage.reasoningTokens}
        </TooltipIconStat>
      )}

      {usage?.totalTokens !== undefined && (
        <TooltipIconStat
          tooltip="Total tokens"
          icon={<Hash className="h-3 w-3 mr-1" />}
        >
          Total: {usage.totalTokens}
        </TooltipIconStat>
      )}

      {usage?.cost !== undefined && (
        <TooltipIconStat
          tooltip="Cost"
          icon={<DollarSign className="h-3 w-3 mr-1" />}
        >
          ${usage.cost.toFixed(6)}
        </TooltipIconStat>
      )}

      {toolName && (
        <TooltipIconStat
          tooltip="Tool used"
          icon={<Wrench className="h-3 w-3 mr-1" />}
        >
          Tool: {toolName}
        </TooltipIconStat>
      )}
    </div>
  );
};

export default ResponseMetrics;
