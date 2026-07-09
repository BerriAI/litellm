import React from "react";

interface ConfigInfoMessageProps {
  show: boolean;
  promptStorageEnabled?: boolean;
}

export const ConfigInfoMessage: React.FC<ConfigInfoMessageProps> = ({ show, promptStorageEnabled = false }) => {
  if (!show) return null;

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start">
      <div className="text-blue-500 mr-3 shrink-0 mt-0.5">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="16" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
      </div>
      <div>
        <h4 className="text-sm font-medium text-blue-800">Request/Response Data Not Available</h4>
        {promptStorageEnabled ? (
          <p className="text-sm text-blue-700 mt-1">
            Prompt storage is enabled, but no request or response payload was recorded for this request. This usually
            means the payload was never captured rather than that storage is misconfigured, for example when a request
            failed before the payload was stored (such as a streaming request whose upstream call errored), or when the
            payload was written through a different logging integration. Check the request status and any error details
            above to see what happened.
          </p>
        ) : (
          <>
            <p className="text-sm text-blue-700 mt-1">
              To view request and response details, enable prompt storage in your LiteLLM configuration by adding the
              following to your <code className="bg-blue-100 px-1 py-0.5 rounded-sm">proxy_config.yaml</code> file, or
              toggle the setting in <strong>Admin Settings → Logging Settings</strong>.
            </p>
            <pre className="mt-2 bg-white p-3 rounded-sm border border-blue-200 text-xs font-mono overflow-auto">
              {`general_settings:
  store_model_in_db: true
  store_prompts_in_spend_logs: true`}
            </pre>
            <p className="text-xs text-blue-700 mt-2">
              Note: This will only affect new requests after the configuration change.
            </p>
          </>
        )}
      </div>
    </div>
  );
};
