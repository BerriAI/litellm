import React, { useState } from "react";
import { Tooltip, Button } from "antd";
import {
  CheckCircleOutlined,
  ClockCircleOutlined,
  LoadingOutlined,
  ExclamationCircleOutlined,
  CopyOutlined,
  DownOutlined,
  RightOutlined,
  LinkOutlined,
  FileTextOutlined,
  RobotOutlined,
} from "@ant-design/icons";

export interface A2ATaskMetadata {
  taskId?: string;
  contextId?: string;
  status?: {
    state?: string;
    timestamp?: string;
    message?: string;
  };
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
      return <CheckCircleOutlined className="text-green-500" />;
    case "working":
    case "submitted":
      return <LoadingOutlined className="text-blue-500" />;
    case "failed":
    case "canceled":
      return <ExclamationCircleOutlined className="text-red-500" />;
    default:
      return <ClockCircleOutlined className="text-gray-500" />;
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
        <RobotOutlined className="mr-1.5 text-blue-500" />
        <span className="font-medium text-gray-700">A2A Metadata</span>
      </div>

      {/* Main metrics row */}
      <div className="flex flex-wrap items-center gap-2 text-gray-500 ml-4">
        {/* Status badge */}
        {status?.state && (
          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(status.state)}`}>
            {getStatusIcon(status.state)}
            <span className="ml-1 capitalize">{status.state}</span>
          </span>
        )}

        {/* Timestamp */}
        {formattedTime && (
          <Tooltip title={status?.timestamp}>
            <span className="flex items-center">
              <ClockCircleOutlined className="mr-1" />
              {formattedTime}
            </span>
          </Tooltip>
        )}

        {/* Latency */}
        {totalLatency !== undefined && (
          <Tooltip title="Total latency">
            <span className="flex items-center text-blue-600">
              <ClockCircleOutlined className="mr-1" />
              {(totalLatency / 1000).toFixed(2)}s
            </span>
          </Tooltip>
        )}

        {/* Time to first token */}
        {timeToFirstToken !== undefined && (
          <Tooltip title="Time to first token">
            <span className="flex items-center text-green-600">
              TTFT: {(timeToFirstToken / 1000).toFixed(2)}s
            </span>
          </Tooltip>
        )}
      </div>

      {/* IDs row */}
      <div className="flex flex-wrap items-center gap-3 text-gray-500 ml-4 mt-1.5">
        {/* Task ID */}
        {taskId && (
          <Tooltip title={`Click to copy: ${taskId}`}>
            <span
              className="flex items-center cursor-pointer hover:text-gray-700"
              onClick={() => copyToClipboard(taskId)}
            >
              <FileTextOutlined className="mr-1" />
              Task: {truncateId(taskId)}
              <CopyOutlined className="ml-1 text-gray-400 hover:text-gray-600" />
            </span>
          </Tooltip>
        )}

        {/* Context/Session ID */}
        {contextId && (
          <Tooltip title={`Click to copy: ${contextId}`}>
            <span
              className="flex items-center cursor-pointer hover:text-gray-700"
              onClick={() => copyToClipboard(contextId)}
            >
              <LinkOutlined className="mr-1" />
              Session: {truncateId(contextId)}
              <CopyOutlined className="ml-1 text-gray-400 hover:text-gray-600" />
            </span>
          </Tooltip>
        )}

        {/* Details toggle */}
        {(metadata || status?.message) && (
          <Button
            type="text"
            size="small"
            className="text-xs text-blue-500 hover:text-blue-700 p-0 h-auto"
            onClick={() => setShowDetails(!showDetails)}
          >
            {showDetails ? <DownOutlined /> : <RightOutlined />}
            <span className="ml-1">Details</span>
          </Button>
        )}
      </div>

      {/* Expandable details panel */}
      {showDetails && (
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
              <code className="ml-2 px-2 py-1 bg-white border border-gray-200 rounded text-xs font-mono">{taskId}</code>
              <CopyOutlined
                className="ml-2 cursor-pointer text-gray-400 hover:text-blue-500"
                onClick={() => copyToClipboard(taskId)}
              />
            </div>
          )}

          {contextId && (
            <div className="mb-1.5 flex items-center">
              <span className="font-medium text-gray-700 w-24">Session ID:</span>
              <code className="ml-2 px-2 py-1 bg-white border border-gray-200 rounded text-xs font-mono">{contextId}</code>
              <CopyOutlined
                className="ml-2 cursor-pointer text-gray-400 hover:text-blue-500"
                onClick={() => copyToClipboard(contextId)}
              />
            </div>
          )}

          {/* Metadata fields */}
          {metadata && Object.keys(metadata).length > 0 && (
            <div className="mt-3">
              <span className="font-medium text-gray-700">Custom Metadata:</span>
              <pre className="mt-1.5 p-2 bg-white border border-gray-200 rounded text-xs font-mono overflow-x-auto whitespace-pre-wrap">
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
