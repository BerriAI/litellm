import { useState } from "react";
import { LogEntry } from "./columns";
import NotificationsManager from "../molecules/notifications_manager";
import { CheckIcon, ClipboardIcon } from "lucide-react";
import { JsonView } from "react-json-view-lite";
import "react-json-view-lite/dist/index.css";

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

// Custom styles for JsonView matching API Reference aesthetic
const polishedJsonStyles = {
  container: "polished-json-container",
  basicChildStyle: "polished-json-basic",
  label: "polished-json-label",
  nullValue: "polished-json-null",
  undefinedValue: "polished-json-undefined",
  numberValue: "polished-json-number",
  stringValue: "polished-json-string",
  booleanValue: "polished-json-boolean",
  otherValue: "polished-json-other",
  punctuation: "polished-json-punctuation",
  collapseIcon: "polished-json-collapse-icon",
  expandIcon: "polished-json-expand-icon",
  collapsedContent: "polished-json-collapsed",
};

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
    <>
      <style>{`
        .polished-json-container {
          font-family: 'SF Mono', Monaco, 'Courier New', monospace;
          font-size: 0.875rem;
          line-height: 1.6;
          color: #24292f;
          background-color: #fafafa;
          padding: 1.25rem;
          border-radius: 0.5rem;
        }
        
        .polished-json-label {
          color: #116329;
          font-weight: 500;
        }
        
        .polished-json-string {
          color: #0a3069;
        }
        
        .polished-json-number {
          color: #0550ae;
        }
        
        .polished-json-boolean {
          color: #8250df;
        }
        
        .polished-json-null,
        .polished-json-undefined {
          color: #6e7781;
          font-style: italic;
        }
        
        .polished-json-punctuation {
          color: #24292f;
        }
        
        .polished-json-collapse-icon,
        .polished-json-expand-icon {
          cursor: pointer;
          user-select: none;
          color: #656d76;
          font-size: 0.75rem;
          margin-right: 0.25rem;
          transition: color 0.2s;
        }
        
        .polished-json-collapse-icon:hover,
        .polished-json-expand-icon:hover {
          color: #0969da;
        }
        
        .polished-json-collapsed {
          color: #656d76;
          font-style: italic;
          margin-left: 0.5rem;
        }
        
        .polished-json-basic {
          margin-left: 1rem;
        }
      `}</style>
      
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
          <div className="relative rounded-lg border border-gray-200 overflow-hidden w-full max-w-full bg-white">
            <div className="overflow-auto max-h-[500px] w-full max-w-full">
              <JsonView data={getRawRequest()} style={polishedJsonStyles} clickToExpandNode={true} />
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
          <div className="relative rounded-lg border border-gray-200 overflow-hidden w-full max-w-full bg-white">
            {hasResponse ? (
              <div className="overflow-auto max-h-[500px] w-full max-w-full">
                <JsonView data={formattedResponse()} style={polishedJsonStyles} clickToExpandNode={true} />
              </div>
            ) : (
              <div className="p-8 text-center bg-gray-50">
                <p className="text-gray-500 text-sm italic">Response data not available</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
