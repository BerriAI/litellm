import React, { useState } from "react";
import {
  Bot,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  Clock,
  Copy,
  FileText,
  Link,
  LoaderCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export interface A2ATaskMetadata {
  taskId?: string;
  contextId?: string;
  status?: {
    state?: string;
    timestamp?: string;
    message?: string;
  };
  metadata?: Record<string, unknown>;
}

interface A2AMetricsProps {
  a2aMetadata?: A2ATaskMetadata;
  timeToFirstToken?: number;
  totalLatency?: number;
}

const getStatusIcon = (state?: string) => {
  switch (state) {
    case "completed":
      return <CheckCircle className="size-3 text-green-500" />;
    case "working":
    case "submitted":
      return <LoaderCircle className="size-3 animate-spin text-blue-500" />;
    case "failed":
    case "canceled":
      return <CircleAlert className="size-3 text-red-500" />;
    default:
      return <Clock className="size-3 text-gray-500" />;
  }
};

const getStatusColor = (state?: string) => {
  switch (state) {
    case "completed":
      return "bg-green-100 text-green-700";
    case "working":
    case "submitted":
      return "bg-blue-100 text-blue-700";
    case "failed":
    case "canceled":
      return "bg-red-100 text-red-700";
    default:
      return "bg-gray-100 text-gray-700";
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

const A2AMetrics: React.FC<A2AMetricsProps> = ({ a2aMetadata, timeToFirstToken, totalLatency }) => {
  const [showDetails, setShowDetails] = useState(false);

  if (!a2aMetadata && !timeToFirstToken && !totalLatency) return null;

  const { taskId, contextId, status, metadata } = a2aMetadata || {};
  const formattedTime = formatTimestamp(status?.timestamp);

  return (
    <div className="a2a-metrics mt-3 pt-2 border-t border-gray-200 text-xs">
      {/* A2A Metadata Header */}
      <div className="flex items-center mb-2 text-gray-600">
        <Bot className="mr-1.5 size-4 text-blue-500" />
        <span className="font-medium text-gray-700">A2A Metadata</span>
      </div>

      {/* Main metrics row */}
      <div className="flex flex-wrap items-center gap-2 text-gray-500 ml-4">
        {/* Status badge */}
        {status?.state && (
          <span
            className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(status.state)}`}
          >
            {getStatusIcon(status.state)}
            <span className="ml-1 capitalize">{status.state}</span>
          </span>
        )}

        {/* Timestamp */}
        {formattedTime && (
          <Tooltip>
            <TooltipTrigger render={<span className="flex items-center" />}>
              <Clock className="mr-1 size-3" />
              {formattedTime}
            </TooltipTrigger>
            <TooltipContent>{status?.timestamp}</TooltipContent>
          </Tooltip>
        )}

        {/* Latency */}
        {totalLatency !== undefined && (
          <Tooltip>
            <TooltipTrigger render={<span className="flex items-center text-blue-600" />}>
              <Clock className="mr-1 size-3" />
              {(totalLatency / 1000).toFixed(2)}s
            </TooltipTrigger>
            <TooltipContent>Total latency</TooltipContent>
          </Tooltip>
        )}

        {/* Time to first token */}
        {timeToFirstToken !== undefined && (
          <Tooltip>
            <TooltipTrigger render={<span className="flex items-center text-green-600" />}>
              TTFT: {(timeToFirstToken / 1000).toFixed(2)}s
            </TooltipTrigger>
            <TooltipContent>Time to first token</TooltipContent>
          </Tooltip>
        )}
      </div>

      {/* IDs row */}
      <div className="flex flex-wrap items-center gap-3 text-gray-500 ml-4 mt-1.5">
        {/* Task ID */}
        {taskId && (
          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  type="button"
                  variant="ghost"
                  size="xs"
                  className="h-auto p-0 font-normal text-gray-500 hover:bg-transparent hover:text-gray-700"
                  onClick={() => copyToClipboard(taskId)}
                  aria-label={`Copy task ID ${taskId}`}
                />
              }
            >
              <FileText className="size-3" />
              Task: {truncateId(taskId)}
              <Copy className="size-3 text-gray-400" />
            </TooltipTrigger>
            <TooltipContent>Click to copy: {taskId}</TooltipContent>
          </Tooltip>
        )}

        {/* Context/Session ID */}
        {contextId && (
          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  type="button"
                  variant="ghost"
                  size="xs"
                  className="h-auto p-0 font-normal text-gray-500 hover:bg-transparent hover:text-gray-700"
                  onClick={() => copyToClipboard(contextId)}
                  aria-label={`Copy session ID ${contextId}`}
                />
              }
            >
              <Link className="size-3" />
              Session: {truncateId(contextId)}
              <Copy className="size-3 text-gray-400" />
            </TooltipTrigger>
            <TooltipContent>Click to copy: {contextId}</TooltipContent>
          </Tooltip>
        )}

        {/* Details toggle */}
        {(metadata || status?.message) && (
          <Collapsible open={showDetails} onOpenChange={setShowDetails}>
            <CollapsibleTrigger
              render={
                <Button
                  type="button"
                  variant="ghost"
                  size="xs"
                  className="h-auto p-0 text-xs text-blue-500 hover:bg-transparent hover:text-blue-700"
                />
              }
            >
              {showDetails ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
              Details
            </CollapsibleTrigger>
          </Collapsible>
        )}
      </div>

      {/* Expandable details panel */}
      <Collapsible open={showDetails} onOpenChange={setShowDetails}>
        <CollapsibleContent>
          <div className="mt-2 ml-4 p-3 bg-gray-50 rounded-md text-gray-600 border border-gray-200">
            {/* Status message */}
            {status?.message && (
              <div className="mb-2">
                <span className="font-medium text-gray-700">Status Message:</span>
                <span className="ml-2">{status.message}</span>
              </div>
            )}

            {/* Full IDs */}
            {taskId && (
              <div className="mb-1.5 flex items-center">
                <span className="font-medium text-gray-700 w-24">Task ID:</span>
                <code className="ml-2 px-2 py-1 bg-white border border-gray-200 rounded-sm text-xs font-mono">
                  {taskId}
                </code>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-xs"
                  className="ml-2 text-gray-400 hover:text-blue-500"
                  onClick={() => copyToClipboard(taskId)}
                  aria-label={`Copy task ID ${taskId}`}
                >
                  <Copy className="size-3" />
                </Button>
              </div>
            )}

            {contextId && (
              <div className="mb-1.5 flex items-center">
                <span className="font-medium text-gray-700 w-24">Session ID:</span>
                <code className="ml-2 px-2 py-1 bg-white border border-gray-200 rounded-sm text-xs font-mono">
                  {contextId}
                </code>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-xs"
                  className="ml-2 text-gray-400 hover:text-blue-500"
                  onClick={() => copyToClipboard(contextId)}
                  aria-label={`Copy session ID ${contextId}`}
                >
                  <Copy className="size-3" />
                </Button>
              </div>
            )}

            {/* Metadata fields */}
            {metadata && Object.keys(metadata).length > 0 && (
              <div className="mt-3">
                <span className="font-medium text-gray-700">Custom Metadata:</span>
                <pre className="mt-1.5 p-2 bg-white border border-gray-200 rounded-sm text-xs font-mono overflow-x-auto whitespace-pre-wrap">
                  {JSON.stringify(metadata, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
};

export default A2AMetrics;
