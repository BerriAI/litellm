import React from "react";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Copy, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import { EndpointType } from "./mode_endpoint_mapping";
import NotificationsManager from "../../molecules/notifications_manager";

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
      return useApiSessionManagement
        ? "API Session: Ready"
        : "UI Session: Ready";
    }

    const sessionPrefix = useApiSessionManagement
      ? "Response ID"
      : "UI Session";
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
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-foreground">
            Session Management
          </span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-3 w-3 text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                Choose between LiteLLM API session management (using
                previous_response_id) or UI-based session management (using
                chat history)
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          <span
            className={cn(
              useApiSessionManagement
                ? "text-muted-foreground"
                : "text-foreground",
            )}
          >
            UI
          </span>
          <Switch
            checked={useApiSessionManagement}
            onCheckedChange={onToggleSessionManagement}
          />
          <span
            className={cn(
              useApiSessionManagement
                ? "text-foreground"
                : "text-muted-foreground",
            )}
          >
            API
          </span>
        </div>
      </div>

      <div
        className={cn(
          "text-xs p-2 rounded-md border",
          responsesSessionId
            ? "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900 dark:text-emerald-300"
            : "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/30 dark:border-blue-900 dark:text-blue-300",
        )}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <Info className="h-3 w-3" />
            {getSessionDisplay()}
          </div>
          {responsesSessionId && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={handleCopySessionId}
                    className="ml-2 p-1 hover:bg-emerald-100 dark:hover:bg-emerald-900 rounded transition-colors"
                    aria-label="Copy response ID"
                  >
                    <Copy className="h-3 w-3" />
                  </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-[500px]">
                  <div className="text-xs">
                    <div className="mb-1">
                      Copy response ID to continue session:
                    </div>
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
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
        <div className="text-xs opacity-75 mt-1">
          {getSessionDescription()}
        </div>
      </div>
    </div>
  );
};

export default SessionManagement;
