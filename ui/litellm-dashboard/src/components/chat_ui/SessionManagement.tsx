import React from "react";
import { Switch, Tooltip } from "antd";
import { InfoCircleOutlined, CopyOutlined } from "@ant-design/icons";
import { EndpointType } from "./mode_endpoint_mapping";
import NotificationsManager from "../molecules/notifications_manager";

interface SessionManagementProps {
  endpointType: string;
  responsesSessionId: string | null;
  useApiSessionManagement: boolean;
  onToggleSessionManagement: (useApi: boolean) => void;
}

const SessionManagement: React.FC<SessionManagementProps> = ({
  endpointType,
  responsesSessionId,
  useApiSessionManagement,
  onToggleSessionManagement,
}) => {
  if (endpointType !== EndpointType.RESPONSES) {
    return null;
  }

  const handleCopySessionId = () => {
    if (responsesSessionId) {
      navigator.clipboard.writeText(responsesSessionId);
      NotificationsManager.success("Response ID copied to clipboard!");
    }
  };

  const getSessionDisplay = () => {
    if (!responsesSessionId) {
      return useApiSessionManagement ? "API Session: Ready" : "UI Session: Ready";
    }

    const sessionPrefix = useApiSessionManagement ? "Response ID" : "UI Session";
    const truncatedId = responsesSessionId.slice(0, 10);
    return `${sessionPrefix}: ${truncatedId}...`;
  };

  const getSessionDescription = () => {
    if (!responsesSessionId) {
      return useApiSessionManagement
        ? "LiteLLM will manage session using previous_response_id"
        : "UI will manage session using chat history";
    }

    return useApiSessionManagement
      ? "LiteLLM API session active - context maintained server-side"
      : "UI session active - context maintained client-side";
  };

  return (
    <div className="mb-4">
      {/* Session Management Toggle */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">Session Management</span>
          <Tooltip title="Choose between LiteLLM API session management (using previous_response_id) or UI-based session management (using chat history)">
            <InfoCircleOutlined className="text-gray-400" style={{ fontSize: "12px" }} />
          </Tooltip>
        </div>
        <Switch
          checked={useApiSessionManagement}
          onChange={onToggleSessionManagement}
          checkedChildren="API"
          unCheckedChildren="UI"
          size="small"
        />
      </div>

      {/* Session Status Indicator */}
      <div
        className={`text-xs p-2 rounded-md ${
          responsesSessionId
            ? "bg-green-50 text-green-700 border border-green-200"
            : "bg-blue-50 text-blue-700 border border-blue-200"
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <InfoCircleOutlined style={{ fontSize: "12px" }} />
            {getSessionDisplay()}
          </div>
          {responsesSessionId && (
            <Tooltip
              title={
                <div className="text-xs">
                  <div className="mb-1">Copy response ID to continue session:</div>
                  <div className="bg-gray-800 text-gray-100 p-2 rounded font-mono text-xs whitespace-pre-wrap">
                    {`curl -X POST "your-proxy-url/v1/responses" \\
  -H "Authorization: Bearer your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "your-model",
    "input": [{"role": "user", "content": "your message", "type": "message"}],
    "previous_response_id": "${responsesSessionId}",
    "stream": true
  }'`}
                  </div>
                </div>
              }
              overlayStyle={{ maxWidth: "500px" }}
            >
              <button onClick={handleCopySessionId} className="ml-2 p-1 hover:bg-green-100 rounded transition-colors">
                <CopyOutlined style={{ fontSize: "12px" }} />
              </button>
            </Tooltip>
          )}
        </div>
        <div className="text-xs opacity-75 mt-1">{getSessionDescription()}</div>
      </div>
    </div>
  );
};

export default SessionManagement;
