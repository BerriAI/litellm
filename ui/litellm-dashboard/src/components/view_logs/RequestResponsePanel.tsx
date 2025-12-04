import { useState } from "react";
import { LogEntry } from "./columns";
import NotificationsManager from "../molecules/notifications_manager";
import { CheckIcon, ClipboardIcon } from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";

interface RequestResponsePanelProps {
  row: {
    original: LogEntry;
  };
  hasMessages: string | boolean;
  hasResponse: string | boolean;
  hasError: boolean;
  errorInfo: any;
  getRawRequest: () => any;
  formattedResponse: () => any;
}

export function RequestResponsePanel({
  row,
  hasMessages,
  hasResponse,
  hasError,
  errorInfo,
  getRawRequest,
  formattedResponse,
}: RequestResponsePanelProps) {
  const [copiedRequest, setCopiedRequest] = useState(false);
  const [copiedResponse, setCopiedResponse] = useState(false);

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

  const handleCopyRequest = async () => {
    const success = await copyToClipboard(JSON.stringify(getRawRequest(), null, 2));
    if (success) {
      setCopiedRequest(true);
      setTimeout(() => setCopiedRequest(false), 2000);
      NotificationsManager.success("Request copied to clipboard");
    } else {
      NotificationsManager.fromBackend("Failed to copy request");
    }
  };

  const handleCopyResponse = async () => {
    const success = await copyToClipboard(JSON.stringify(formattedResponse(), null, 2));
    if (success) {
      setCopiedResponse(true);
      setTimeout(() => setCopiedResponse(false), 2000);
      NotificationsManager.success("Response copied to clipboard");
    } else {
      NotificationsManager.fromBackend("Failed to copy response");
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full max-w-full overflow-hidden box-border">
      {/* Request Side */}
      <div className="flex flex-col w-full max-w-full overflow-hidden">
        <div className="flex justify-between items-center mb-3">
          <h3 className="text-base font-semibold text-gray-900">Request</h3>
          <button
            onClick={handleCopyRequest}
            className="p-2 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-600 transition-colors"
            title="Copy request"
            aria-label="Copy request"
          >
            {copiedRequest ? <CheckIcon size={16} /> : <ClipboardIcon size={16} />}
          </button>
        </div>
        <div className="relative rounded-lg border border-gray-200 overflow-hidden w-full max-w-full">
          <div className="overflow-auto max-h-[500px] w-full max-w-full">
            <SyntaxHighlighter
              language="json"
              style={oneLight}
              customStyle={{
                margin: 0,
                padding: "1.25rem",
                borderRadius: "0.5rem",
                fontSize: "0.875rem",
                backgroundColor: "#fafafa",
                lineHeight: "1.6",
              }}
              showLineNumbers={false}
              wrapLines={true}
              wrapLongLines={true}
            >
              {JSON.stringify(getRawRequest(), null, 2)}
            </SyntaxHighlighter>
          </div>
        </div>
      </div>

      {/* Response Side */}
      <div className="flex flex-col w-full max-w-full overflow-hidden">
        <div className="flex justify-between items-center mb-3">
          <h3 className="text-base font-semibold text-gray-900">
            Response
            {hasError && (
              <span className="ml-2 text-sm font-normal text-red-600">
                • HTTP {errorInfo?.error_code || 400}
              </span>
            )}
          </h3>
          <button
            onClick={handleCopyResponse}
            className="p-2 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Copy response"
            aria-label="Copy response"
            disabled={!hasResponse}
          >
            {copiedResponse ? <CheckIcon size={16} /> : <ClipboardIcon size={16} />}
          </button>
        </div>
        <div className="relative rounded-lg border border-gray-200 overflow-hidden w-full max-w-full">
          {hasResponse ? (
            <div className="overflow-auto max-h-[500px] w-full max-w-full">
              <SyntaxHighlighter
                language="json"
                style={oneLight}
                customStyle={{
                  margin: 0,
                  padding: "1.25rem",
                  borderRadius: "0.5rem",
                  fontSize: "0.875rem",
                  backgroundColor: "#fafafa",
                  lineHeight: "1.6",
                }}
                showLineNumbers={false}
                wrapLines={true}
                wrapLongLines={true}
              >
                {JSON.stringify(formattedResponse(), null, 2)}
              </SyntaxHighlighter>
            </div>
          ) : (
            <div className="p-8 text-center bg-gray-50">
              <p className="text-gray-500 text-sm italic">Response data not available</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
