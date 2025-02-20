import React from "react";
import { Card, Text, Button, TabGroup, TabList, Tab, TabPanel, TabPanels } from "@tremor/react";
import { CheckCircleIcon, XCircleIcon, ClipboardCopyIcon } from "@heroicons/react/outline";
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

// TableClickableErrorField component with copy-to-clipboard functionality
const TableClickableErrorField: React.FC<{ label: string; value: string }> = ({
  label,
  value,
}) => {
  const [isExpanded, setIsExpanded] = React.useState(false);
  const [copied, setCopied] = React.useState(false);
  const truncated = value.length > 50 ? value.substring(0, 50) + "..." : value;

  const handleCopy = () => {
    navigator.clipboard.writeText(value);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <tr className="hover:bg-gray-50">
      <td className="px-4 py-2 align-top" colSpan={2}>
        <div className="flex items-center justify-between group">
          <div className="flex items-center flex-1">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-gray-400 hover:text-gray-600 mr-2"
            >
              {isExpanded ? "▼" : "▶"}
            </button>
            <div>
              <div className="text-sm text-gray-600">{label}</div>
              <pre className="mt-1 text-sm font-mono text-gray-800 whitespace-pre-wrap">
                {isExpanded ? value : truncated}
              </pre>
            </div>
          </div>
          <button
            onClick={handleCopy}
            className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-gray-600"
          >
            <ClipboardCopyIcon className="h-4 w-4" />
          </button>
        </div>
      </td>
    </tr>
  );
};

// Add new interface for Redis details
interface RedisDetails {
  redis_host?: string;
  redis_port?: string;
  redis_version?: string;
  startup_nodes?: string;
}

// Add new interface for Error Details
interface ErrorDetails {
  message: string;
  traceback: string;
  litellm_params?: any;
  redis_cache_params?: any;
}

// Update HealthCheckDetails component to handle errors
const HealthCheckDetails: React.FC<{ response: any }> = ({ response }) => {
  // Parse error information if present
  let errorDetails: ErrorDetails | null = null;
  let parsedLitellmParams = null;
  let parsedRedisParams = null;

  if (response.error) {
    try {
      const errorMessage = JSON.parse(response.error.message);
      errorDetails = {
        message: errorMessage.message,
        traceback: errorMessage.traceback,
        litellm_params: errorMessage.litellm_cache_params,
        redis_cache_params: errorMessage.redis_cache_params 
      };
      
      // Parse the cache parameters from the error response
      parsedLitellmParams = deepParse(errorDetails.litellm_params);
      parsedRedisParams = deepParse(errorDetails.redis_cache_params);
    } catch (e) {
      console.error("Error parsing error details:", e);
    }
  } else {
    // Existing parsing for non-error responses
    parsedLitellmParams = deepParse(response.litellm_cache_params);
    parsedRedisParams = deepParse(response.redis_cache_params);
  }

  console.log("parsedLitellmParams", parsedLitellmParams);
  console.log("parsedRedisParams", parsedRedisParams);
  
  // Extract Redis details from parsed response, checking multiple possible paths
  const redisDetails: RedisDetails = {
    redis_host: parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs?.host ||
                parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs?.host ||
                parsedRedisParams?.connection_kwargs?.host ||
                "N/A",
    
    redis_port: parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs?.port ||
                parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs?.port ||
                parsedRedisParams?.connection_kwargs?.port ||
                "N/A",
    
    redis_version: parsedRedisParams?.redis_version || "N/A",
    
    startup_nodes: parsedRedisParams?.redis_kwargs?.startup_nodes ? 
                  JSON.stringify(parsedRedisParams.redis_kwargs.startup_nodes) :
                  JSON.stringify([{
                    host: parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs?.host ||
                          parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs?.host,
                    port: parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs?.port ||
                          parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs?.port
                  }])
  };

  return (
    <div className="bg-white rounded-lg shadow">
      <TabGroup>
        <TabList className="border-b border-gray-200 px-4">
          <Tab className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800">Summary</Tab>
          <Tab className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800">Raw Response</Tab>
        </TabList>

        <TabPanels>
          <TabPanel className="p-4">
            <div>
              <div className="flex items-center mb-6">
                {response.status === "healthy" ? (
                  <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2" />
                ) : (
                  <XCircleIcon className="h-5 w-5 text-red-500 mr-2" />
                )}
                <Text className={`text-sm font-medium ${response.status === "healthy" ? "text-green-500" : "text-red-500"}`}>
                  Cache Status: {response.status || "unhealthy"}
                </Text>
              </div>

              <table className="w-full border-collapse">
                <tbody>
                  {/* Show error message if present */}
                  {errorDetails && (
                    <>
                      <tr><td colSpan={2} className="pt-4 pb-2 font-semibold text-red-600">Error Details</td></tr>
                      <TableClickableErrorField
                        label="Error Message"
                        value={errorDetails.message}
                      />
                      <TableClickableErrorField
                        label="Traceback"
                        value={errorDetails.traceback}
                      />
                    </>
                  )}

                  {/* Always show cache details, regardless of error state */}
                  <tr><td colSpan={2} className="pt-4 pb-2 font-semibold">Cache Details</td></tr>
                  <TableClickableErrorField
                    label="Cache Type"
                        value={response.cache_type}
                      />
                      <TableClickableErrorField
                        label="Ping Response"
                        value={String(response.ping_response)}
                      />
                      <TableClickableErrorField
                        label="Set Cache Response"
                        value={response.set_cache_response || "N/A"}
                  />
                  
                  {/* Redis Details Section */}
                      {response.cache_type === "redis" && (
                        <>
                  <tr><td colSpan={2} className="pt-4 pb-2 font-semibold">Redis Details</td></tr>
                  <TableClickableErrorField
                    label="Redis Host"
                            value={redisDetails.redis_host || "N/A"}
                  />
                  <TableClickableErrorField
                    label="Redis Port"
                            value={redisDetails.redis_port || "N/A"}
                  />
                  <TableClickableErrorField
                    label="Redis Version"
                            value={redisDetails.redis_version || "N/A"}
                  />
                  <TableClickableErrorField
                    label="Startup Nodes"
                            value={redisDetails.startup_nodes || "N/A"}
                  />
                </>
              )}
                </tbody>
              </table>
            </div>
          </TabPanel>

          <TabPanel className="p-4">
            <div className="bg-gray-50 rounded-md p-4 font-mono text-sm">
              <pre className="overflow-auto max-h-[500px]">
                {JSON.stringify({
                  ...response,
                  litellm_cache_params: parsedLitellmParams,
                  redis_cache_params: parsedRedisParams
                }, null, 2)}
              </pre>
            </div>
          </TabPanel>
        </TabPanels>
      </TabGroup>
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
  const [isLoading, setIsLoading] = React.useState<boolean>(false);

  const handleHealthCheck = async () => {
    setIsLoading(true);
    const startTime = performance.now();
    await runCachingHealthCheck();
    const endTime = performance.now();
    setLocalResponseTimeMs(endTime - startTime);
    setIsLoading(false);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Button 
          onClick={handleHealthCheck}
          disabled={isLoading}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white text-sm px-4 py-2 rounded-md"
        >
          {isLoading ? "Running Health Check..." : "Run Health Check"}
        </Button>
        <ResponseTimeIndicator responseTimeMs={localResponseTimeMs} />
      </div>

      {healthCheckResponse && (
        <HealthCheckDetails response={healthCheckResponse} />
      )}
    </div>
  );
}; 