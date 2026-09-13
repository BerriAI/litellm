import React from "react";

interface ConfigInfoMessageProps {
  show: boolean;
}

export const ConfigInfoMessage: React.FC<ConfigInfoMessageProps> = ({ show }) => {
  if (!show) return null;

  return (
    <div className="bg-info/10 border border-info/20 rounded-lg p-4 flex items-start">
      <div className="text-info mr-3 shrink-0 mt-0.5">
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
        <h4 className="text-sm font-medium text-info">Request/Response Data Not Available</h4>
        <p className="text-sm text-info mt-1">
          To view request and response details, enable prompt storage in your LiteLLM configuration by adding the
          following to your <code className="bg-info/15 px-1 py-0.5 rounded-sm">proxy_config.yaml</code> file, or toggle
          the setting in <strong>Admin Settings → Logging Settings</strong>.
        </p>
        <pre className="mt-2 bg-card p-3 rounded-sm border border-info/20 text-xs font-mono overflow-auto">
          {`general_settings:
  store_model_in_db: true
  store_prompts_in_spend_logs: true`}
        </pre>
        <p className="text-xs text-info mt-2">
          Note: This will only affect new requests after the configuration change.
        </p>
      </div>
    </div>
  );
};
