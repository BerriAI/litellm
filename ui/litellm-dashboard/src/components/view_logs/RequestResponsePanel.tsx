import { LogEntry } from "./columns";
import { message } from "antd";

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
  const handleCopyRequest = () => {
    try {
      navigator.clipboard.writeText(JSON.stringify(getRawRequest(), null, 2));
      message.success('Request copied to clipboard');
    } catch (error) {
      message.error('Failed to copy request');
    }
  };

  const handleCopyResponse = () => {
    navigator.clipboard.writeText(JSON.stringify(formattedResponse(), null, 2));
    message.success('Response copied to clipboard');
  };

  return (
    <div className="grid grid-cols-2 gap-4">
      {/* Request Side */}
      <div className="bg-white rounded-lg shadow">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="text-lg font-medium">Request</h3>
          <button 
            onClick={handleCopyRequest}
            className="p-1 hover:bg-gray-200 rounded"
            title="Copy request"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
          </button>
        </div>
        <div className="p-4 overflow-auto max-h-96">
          <pre className="text-xs font-mono whitespace-pre-wrap break-all">{JSON.stringify(getRawRequest(), null, 2)}</pre>
        </div>
      </div>

      {/* Response Side */}
      <div className="bg-white rounded-lg shadow">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="text-lg font-medium">
            Response
            {hasError && (
              <span className="ml-2 text-sm text-red-600">
                • HTTP code {errorInfo?.error_code || 400}
              </span>
            )}
          </h3>
          <button 
            onClick={handleCopyResponse}
            className="p-1 hover:bg-gray-200 rounded"
            title="Copy response"
            disabled={!hasResponse}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
          </button>
        </div>
        <div className="p-4 overflow-auto max-h-96 bg-gray-50">
          {hasResponse ? (
            <pre className="text-xs font-mono whitespace-pre-wrap break-all">{JSON.stringify(formattedResponse(), null, 2)}</pre>
          ) : (
            <div className="text-gray-500 text-sm italic text-center py-4">Response data not available</div>
          )}
        </div>
      </div>
    </div>
  );
} 