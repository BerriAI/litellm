import React from "react";
import { Card, Text, Button } from "@tremor/react";
import { CheckCircleIcon, XCircleIcon } from "@heroicons/react/outline";
import { ResponseTimeIndicator } from "./response_time_indicator";

// Helper function to deep-parse a JSON string if possible
const deepParse = (input: any) => {
  let parsed = input;
  if (typeof parsed === "string") {
    try {
      parsed = JSON.parse(parsed);
    } catch {
      return parsed;
    }
  }
  return parsed;
};

// TableClickableErrorField component
const TableClickableErrorField: React.FC<{ label: string; value: string }> = ({
  label,
  value,
}) => {
  const [isExpanded, setIsExpanded] = React.useState(false);
  const truncated = value.length > 50 ? value.substring(0, 50) + "..." : value;

  return (
    <tr className="border-t first:border-t-0">
      <td className="px-6 py-4 align-top w-full" colSpan={2}>
        <div className="flex">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="mr-2 text-gray-500 hover:text-gray-700 focus:outline-none"
          >
            {isExpanded ? "▼" : "▶"}
          </button>
          <div className="flex-1">
            <div className="font-medium">{label}</div>
            <pre className="mt-1 text-sm text-gray-700 whitespace-pre-wrap">
              {isExpanded ? value : truncated}
            </pre>
          </div>
        </div>
      </td>
    </tr>
  );
};

// HealthCheckDetails component
export const HealthCheckDetails: React.FC<{ response: any }> = ({ response }) => {
  if (response.error) {
    let errorData = deepParse(response.error);
    if (errorData && typeof errorData.message === "string") {
      const innerParsed = deepParse(errorData.message);
      if (innerParsed && typeof innerParsed === "object") {
        errorData = innerParsed;
      }
    }

    return (
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="flex items-center space-x-2 text-red-600 p-4 border-b border-gray-200">
          <XCircleIcon className="h-5 w-5" />
          <h3 className="text-lg font-medium">Cache Health Check Failed</h3>
        </div>
        <table className="w-full">
          <tbody>
            {errorData.message && (
              <TableClickableErrorField 
                label="Error Message" 
                value={errorData.message} 
              />
            )}
            {errorData.type && (
              <TableClickableErrorField 
                label="Error Type" 
                value={errorData.type} 
              />
            )}
            {errorData.traceback && (
              <TableClickableErrorField 
                label="Traceback" 
                value={errorData.traceback} 
              />
            )}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="flex items-center space-x-2 text-green-600 p-4 border-b border-gray-200">
        <CheckCircleIcon className="h-5 w-5" />
        <h3 className="text-lg font-medium">Cache Health Check Passed</h3>
      </div>
      <table className="w-full">
        <tbody>
          <TableClickableErrorField 
            label="Status" 
            value={String(response.status || "-")} 
          />
          <TableClickableErrorField 
            label="Cache Type" 
            value={String(response.cache_type || "-")} 
          />
          <TableClickableErrorField 
            label="Ping Response" 
            value={String(response.ping_response || "-")} 
          />
          <TableClickableErrorField 
            label="Set Cache Response" 
            value={String(response.set_cache_response || "-")} 
          />
          {response.litellm_cache_params && (
            <TableClickableErrorField 
              label="LiteLLM Cache Parameters" 
              value={JSON.stringify(deepParse(response.litellm_cache_params), null, 2)} 
            />
          )}
          {response.specific_cache_params && (
            <TableClickableErrorField 
              label="Specific Cache Parameters" 
              value={JSON.stringify(deepParse(response.specific_cache_params), null, 2)} 
            />
          )}
        </tbody>
      </table>
    </div>
  );
};

export const CacheHealthTab: React.FC<{ 
  accessToken: string | null;
  healthCheckResponse: any;
  runCachingHealthCheck: () => void;
  responseTimeMs?: number | null;
}> = ({ accessToken, healthCheckResponse, runCachingHealthCheck, responseTimeMs }) => {
  const [localResponseTimeMs, setLocalResponseTimeMs] = React.useState<number | null>(null);

  const handleHealthCheck = async () => {
    const startTime = performance.now();
    await runCachingHealthCheck();
    const endTime = performance.now();
    setLocalResponseTimeMs(endTime - startTime);
  };

  return (
    <div className="p-4">
      <div className="flex items-center justify-between mb-4">
        <Text className="text-gray-600">
          Cache health will run a very small request through API /cache/ping configured on litellm
        </Text>
        <ResponseTimeIndicator responseTimeMs={localResponseTimeMs} />
      </div>

      <Button 
        onClick={handleHealthCheck}
        className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-md"
      >
        Run cache health
      </Button>
      
      {healthCheckResponse && (
        <div className="mt-4">
          <HealthCheckDetails response={healthCheckResponse} />
        </div>
      )}
    </div>
  );
}; 