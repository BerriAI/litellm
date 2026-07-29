import React from "react";

interface ResponseTimeIndicatorProps {
  responseTimeMs: number | null;
}

export const ResponseTimeIndicator: React.FC<ResponseTimeIndicatorProps> = ({ responseTimeMs }) => {
  if (responseTimeMs === null || responseTimeMs === undefined) return null;

  return (
    <div className="flex items-center space-x-1 text-xs text-gray-500 font-mono">
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M12 6V12L16 14M12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2Z"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span>{responseTimeMs.toFixed(0)}ms</span>
    </div>
  );
};
