import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  AlertCircle,
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Copy,
  FileText,
  Link as LinkIcon,
  Loader2,
} from "lucide-react";

export interface A2ATaskMetadata {
  taskId?: string;
  contextId?: string;
  status?: {
    state?: string;
    timestamp?: string;
    message?: string;
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metadata?: Record<string, any>;
}

interface A2AMetricsProps {
  a2aMetadata?: A2ATaskMetadata;
  timeToFirstToken?: number;
  totalLatency?: number;
}

const getStatusIcon = (state?: string) => {
  switch (state) {
    case "completed":
      return <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />;
    case "working":
    case "submitted":
      return <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />;
    case "failed":
    case "canceled":
      return <AlertCircle className="h-3.5 w-3.5 text-red-500" />;
    default:
      return <Clock className="h-3.5 w-3.5 text-muted-foreground" />;
  }
};

const getStatusColor = (state?: string) => {
  switch (state) {
    case "completed":
      return "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-300";
    case "working":
    case "submitted":
      return "bg-blue-100 text-blue-700 dark:bg-blue-950/30 dark:text-blue-300";
    case "failed":
    case "canceled":
      return "bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-300";
    default:
      return "bg-muted text-muted-foreground";
  }
};

const formatTimestamp = (timestamp?: string) => {
  if (!timestamp) return null;
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return timestamp;
  }
};

const truncateId = (id?: string, length = 8) => {
  if (!id) return null;
  return id.length > length ? `${id.substring(0, length)}…` : id;
};

const copyToClipboard = (text: string) => {
  navigator.clipboard.writeText(text);
};

const A2AMetrics: React.FC<A2AMetricsProps> = ({
  a2aMetadata,
  timeToFirstToken,
  totalLatency,
}) => {
  const [showDetails, setShowDetails] = useState(false);

  if (!a2aMetadata && !timeToFirstToken && !totalLatency) return null;

  const { taskId, contextId, status, metadata } = a2aMetadata || {};
  const formattedTime = formatTimestamp(status?.timestamp);

  return (
    <div className="a2a-metrics mt-3 pt-2 border-t border-border text-xs">
      <div className="flex items-center mb-2 text-muted-foreground">
        <Bot className="h-3.5 w-3.5 mr-1.5 text-blue-500" />
        <span className="font-medium text-foreground">A2A Metadata</span>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-muted-foreground ml-4">
        {status?.state && (
          <span
            className={cn(
              "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium gap-1",
              getStatusColor(status.state),
            )}
          >
            {getStatusIcon(status.state)}
            <span className="capitalize">{status.state}</span>
          </span>
        )}

        {formattedTime && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="flex items-center">
                  <Clock className="h-3.5 w-3.5 mr-1" />
                  {formattedTime}
                </span>
              </TooltipTrigger>
              <TooltipContent>{status?.timestamp}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}

        {totalLatency !== undefined && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="flex items-center text-blue-600">
                  <Clock className="h-3.5 w-3.5 mr-1" />
                  {(totalLatency / 1000).toFixed(2)}s
                </span>
              </TooltipTrigger>
              <TooltipContent>Total latency</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}

        {timeToFirstToken !== undefined && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="flex items-center text-emerald-600">
                  TTFT: {(timeToFirstToken / 1000).toFixed(2)}s
                </span>
              </TooltipTrigger>
              <TooltipContent>Time to first token</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-3 text-muted-foreground ml-4 mt-1.5">
        {taskId && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span
                  className="flex items-center cursor-pointer hover:text-foreground"
                  onClick={() => copyToClipboard(taskId)}
                >
                  <FileText className="h-3.5 w-3.5 mr-1" />
                  Task: {truncateId(taskId)}
                  <Copy className="h-3 w-3 ml-1 text-muted-foreground hover:text-foreground" />
                </span>
              </TooltipTrigger>
              <TooltipContent>Click to copy: {taskId}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}

        {contextId && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span
                  className="flex items-center cursor-pointer hover:text-foreground"
                  onClick={() => copyToClipboard(contextId)}
                >
                  <LinkIcon className="h-3.5 w-3.5 mr-1" />
                  Session: {truncateId(contextId)}
                  <Copy className="h-3 w-3 ml-1 text-muted-foreground hover:text-foreground" />
                </span>
              </TooltipTrigger>
              <TooltipContent>Click to copy: {contextId}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}

        {(metadata || status?.message) && (
          <Button
            variant="ghost"
            size="sm"
            className="text-xs text-blue-500 hover:text-blue-700 p-0 h-auto"
            onClick={() => setShowDetails(!showDetails)}
          >
            {showDetails ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
            <span className="ml-1">Details</span>
          </Button>
        )}
      </div>

      {showDetails && (
        <div className="mt-2 ml-4 p-3 bg-muted rounded-md text-muted-foreground border border-border">
          {status?.message && (
            <div className="mb-2">
              <span className="font-medium text-foreground">Status Message:</span>
              <span className="ml-2">{status.message}</span>
            </div>
          )}

          {taskId && (
            <div className="mb-1.5 flex items-center">
              <span className="font-medium text-foreground w-24">Task ID:</span>
              <code className="ml-2 px-2 py-1 bg-background border border-border rounded text-xs font-mono">
                {taskId}
              </code>
              <Copy
                className="ml-2 h-3.5 w-3.5 cursor-pointer text-muted-foreground hover:text-primary"
                onClick={() => copyToClipboard(taskId)}
              />
            </div>
          )}

          {contextId && (
            <div className="mb-1.5 flex items-center">
              <span className="font-medium text-foreground w-24">Session ID:</span>
              <code className="ml-2 px-2 py-1 bg-background border border-border rounded text-xs font-mono">
                {contextId}
              </code>
              <Copy
                className="ml-2 h-3.5 w-3.5 cursor-pointer text-muted-foreground hover:text-primary"
                onClick={() => copyToClipboard(contextId)}
              />
            </div>
          )}

          {metadata && Object.keys(metadata).length > 0 && (
            <div className="mt-3">
              <span className="font-medium text-foreground">Custom Metadata:</span>
              <pre className="mt-1.5 p-2 bg-background border border-border rounded text-xs font-mono overflow-x-auto whitespace-pre-wrap">
                {JSON.stringify(metadata, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default A2AMetrics;
