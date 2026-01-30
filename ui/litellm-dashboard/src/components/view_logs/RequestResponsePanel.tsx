import { LogEntry } from "./columns";
import NotificationsManager from "../molecules/notifications_manager";
import { JsonView, defaultStyles } from "react-json-view-lite";
import "react-json-view-lite/dist/index.css";

interface RequestResponsePanelProps {
  row: {
    original: LogEntry;
  };
  hasClientRequest: string | boolean;
  hasModelRequest: string | boolean;
  hasClientResponse: string | boolean;
  hasError: boolean;
  errorInfo: any;
  getClientRequest: () => any;
  getModelRequest: () => any;
  formattedResponse: () => any;
}

export function RequestResponsePanel({
  row: _row,
  hasClientRequest,
  hasModelRequest,
  hasClientResponse,
  hasError,
  errorInfo,
  getClientRequest,
  getModelRequest,
  formattedResponse,
}: RequestResponsePanelProps) {
  const copyToClipboard = async (text: string) => {
    try {
      // Try modern clipboard API first
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      } else {
        // Fallback for non-secure contexts (like 0.0.0.0)
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        const successful = document.execCommand("copy");
        document.body.removeChild(textArea);

        if (!successful) {
          throw new Error("execCommand failed");
        }
        return true;
      }
    } catch (error) {
      console.error("Copy failed:", error);
      return false;
    }
  };

  const buildCopyHandler =
    (getData: () => any, successMessage: string, errorMessage: string) => async () => {
      const success = await copyToClipboard(JSON.stringify(getData(), null, 2));
      if (success) {
        NotificationsManager.success(successMessage);
      } else {
        NotificationsManager.fromBackend(errorMessage);
      }
    };

  const panels = [
    {
      key: "client-request",
      title: "Request from client",
      hasData: hasClientRequest,
      getData: getClientRequest,
      copyTitle: "Copy request from client",
      successMessage: "Client request copied to clipboard",
      errorMessage: "Failed to copy client request",
      emptyText: "Request from client not available",
    },
    {
      key: "model-request",
      title: "Request to model/endpoint",
      hasData: hasModelRequest,
      getData: getModelRequest,
      copyTitle: "Copy request to model/endpoint",
      successMessage: "Request to model/endpoint copied to clipboard",
      errorMessage: "Failed to copy request to model/endpoint",
      emptyText: (
        <span className="block whitespace-normal break-words text-left max-w-prose mx-auto">
          Request not available. Enable{" "}
          <a
            className="text-blue-600 underline"
            href="https://docs.litellm.ai/docs/proxy/config_settings#store_prompts_in_spend_logs"
            target="_blank"
            rel="noreferrer"
          >
            <code>store_prompts_in_spend_logs</code>
          </a>{" "}
          to capture and display model requests. If content is truncated, raise{" "}
          <a
            className="text-blue-600 underline"
            href="https://docs.litellm.ai/docs/proxy/config_settings#store_prompts_in_spend_logs#MAX_STRING_LENGTH_PROMPT_IN_DB"
            target="_blank"
            rel="noreferrer"
          >
            <code>MAX_STRING_LENGTH_PROMPT_IN_DB</code>
          </a>.
        </span>
      ),
    },
    {
      key: "client-response",
      title: "Response to client",
      hasData: hasClientResponse,
      getData: formattedResponse,
      copyTitle: "Copy response to client",
      successMessage: "Response to client copied to clipboard",
      errorMessage: "Failed to copy response to client",
      emptyText: "Response to client not available",
    },
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 w-full max-w-full overflow-hidden box-border">
      {panels.map((panel) => {
        const hasData = Boolean(panel.hasData);
        const data = hasData ? panel.getData() : null;
        const handleCopy = buildCopyHandler(panel.getData, panel.successMessage, panel.errorMessage);

        return (
          <div key={panel.key} className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden">
            <div className="flex justify-between items-center p-4 border-b">
              <h3 className="text-lg font-medium">
                {panel.title}
                {panel.key === "client-response" && hasError && (
                  <span className="ml-2 text-sm text-red-600">• HTTP code {errorInfo?.error_code || 400}</span>
                )}
              </h3>
              <button
                onClick={handleCopy}
                className="p-1 hover:bg-gray-200 rounded"
                title={panel.copyTitle}
                disabled={!hasData}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
              </button>
            </div>
            <div className="p-4 overflow-auto max-h-96 w-full max-w-full box-border">
              {hasData ? (
                <div className="[&_[role='tree']]:bg-white [&_[role='tree']]:text-slate-900">
                  <JsonView data={data} style={defaultStyles} clickToExpandNode />
                </div>
              ) : (
                <div className="text-gray-500 text-sm italic text-center py-4">{panel.emptyText}</div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
