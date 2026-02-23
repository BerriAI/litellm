import React from "react";
import { Text, Button, TabGroup, TabList, Tab, TabPanel, TabPanels } from "@tremor/react";
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
const TableClickableErrorField: React.FC<{ label: string; value: string | null | undefined }> = ({ label, value }) => {
  const [isExpanded, setIsExpanded] = React.useState(false);
  const [copied, setCopied] = React.useState(false);
  const safeValue = value?.toString() || "N/A";
  const truncated = safeValue.length > 50 ? safeValue.substring(0, 50) + "..." : safeValue;

  const handleCopy = () => {
    navigator.clipboard.writeText(safeValue);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <tr className="hover:bg-gray-50">
      <td className="px-4 py-2 align-top" colSpan={2}>
        <div className="flex items-center justify-between group">
          <div className="flex items-center flex-1">
            <button onClick={() => setIsExpanded(!isExpanded)} className="text-gray-400 hover:text-gray-600 mr-2">
              {isExpanded ? "▼" : "▶"}
            </button>
            <div>
              <div className="text-sm text-gray-600">{label}</div>
              <pre className="mt-1 text-sm font-mono text-gray-800 whitespace-pre-wrap">
                {isExpanded ? safeValue : truncated}
              </pre>
            </div>
          </div>
          <button onClick={handleCopy} className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-gray-600">
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
  namespace?: string;
}

// Add new interface for Error Details
interface ErrorDetails {
  message: string;
  traceback: string;
  litellm_params?: any;
  health_check_cache_params?: any;
}

// Update HealthCheckDetails component to handle errors
const HealthCheckDetails: React.FC<{ response: any }> = ({ response }) => {
  // Initialize with safe default values
  let errorDetails: ErrorDetails | null = null;
  let parsedLitellmParams: any = {};
  let parsedRedisParams: any = {};

  try {
    if (response?.error) {
      try {
        const errorMessage =
          typeof response.error.message === "string" ? JSON.parse(response.error.message) : response.error.message;

        errorDetails = {
          message: errorMessage?.message || "Unknown error",
          traceback: errorMessage?.traceback || "No traceback available",
          litellm_params: errorMessage?.litellm_cache_params || {},
          health_check_cache_params: errorMessage?.health_check_cache_params || {},
        };

        parsedLitellmParams = deepParse(errorDetails.litellm_params) || {};
        parsedRedisParams = deepParse(errorDetails.health_check_cache_params) || {};
      } catch (e) {
        console.warn("Error parsing error details:", e);
        errorDetails = {
          message: String(response.error.message || "Unknown error"),
          traceback: "Error parsing details",
          litellm_params: {},
          health_check_cache_params: {},
        };
      }
    } else {
      parsedLitellmParams = deepParse(response?.litellm_cache_params) || {};
      parsedRedisParams = deepParse(response?.health_check_cache_params) || {};
    }
  } catch (e) {
    console.warn("Error in response parsing:", e);
    // Provide safe fallback values
    parsedLitellmParams = {};
    parsedRedisParams = {};
  }

  // Safely extract Redis details with fallbacks
  const redisDetails: RedisDetails = {
    redis_host:
      parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs?.host ||
      parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs?.host ||
      parsedRedisParams?.connection_kwargs?.host ||
      parsedRedisParams?.host ||
      "N/A",

    redis_port:
      parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs?.port ||
      parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs?.port ||
      parsedRedisParams?.connection_kwargs?.port ||
      parsedRedisParams?.port ||
      "N/A",

    redis_version: parsedRedisParams?.redis_version || "N/A",

    startup_nodes: (() => {
      try {
        if (parsedRedisParams?.redis_kwargs?.startup_nodes) {
          return JSON.stringify(parsedRedisParams.redis_kwargs.startup_nodes);
        }
        const host =
          parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs?.host ||
          parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs?.host;
        const port =
          parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs?.port ||
          parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs?.port;
        return host && port ? JSON.stringify([{ host, port }]) : "N/A";
      } catch (e) {
        return "N/A";
      }
    })(),

    namespace: parsedRedisParams?.namespace || "N/A",
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
                {response?.status === "healthy" ? (
                  <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2" />
                ) : (
                  <XCircleIcon className="h-5 w-5 text-red-500 mr-2" />
                )}
                <Text
                  className={`text-sm font-medium ${response?.status === "healthy" ? "text-green-500" : "text-red-500"}`}
                >
                  Cache Status: {response?.status || "unhealthy"}
                </Text>
              </div>

              <table className="w-full border-collapse">
                <tbody>
                  {/* Show error message if present */}
                  {errorDetails && (
                    <>
                      <tr>
                        <td colSpan={2} className="pt-4 pb-2 font-semibold text-red-600">
                          Error Details
                        </td>
                      </tr>
                      <TableClickableErrorField label="Error Message" value={errorDetails.message} />
                      <TableClickableErrorField label="Traceback" value={errorDetails.traceback} />
                    </>
                  )}

                  {/* Always show cache details, regardless of error state */}
                  <tr>
                    <td colSpan={2} className="pt-4 pb-2 font-semibold">
                      Cache Details
                    </td>
                  </tr>
                  <TableClickableErrorField label="Cache Configuration" value={String(parsedLitellmParams?.type)} />
                  <TableClickableErrorField label="Ping Response" value={String(response.ping_response)} />
                  <TableClickableErrorField label="Set Cache Response" value={response.set_cache_response || "N/A"} />
                  <TableClickableErrorField
                    label="litellm_settings.cache_params"
                    value={JSON.stringify(parsedLitellmParams, null, 2)}
                  />

                  {/* Redis Details Section */}
                  {parsedLitellmParams?.type === "redis" && (
                    <>
                      <tr>
                        <td colSpan={2} className="pt-4 pb-2 font-semibold">
                          Redis Details
                        </td>
                      </tr>
                      <TableClickableErrorField label="Redis Host" value={redisDetails.redis_host || "N/A"} />
                      <TableClickableErrorField label="Redis Port" value={redisDetails.redis_port || "N/A"} />
                      <TableClickableErrorField label="Redis Version" value={redisDetails.redis_version || "N/A"} />
                      <TableClickableErrorField label="Startup Nodes" value={redisDetails.startup_nodes || "N/A"} />
                      <TableClickableErrorField label="Namespace" value={redisDetails.namespace || "N/A"} />
                    </>
                  )}
                </tbody>
              </table>
            </div>
          </TabPanel>

          <TabPanel className="p-4">
            <div className="bg-gray-50 rounded-md p-4 font-mono text-sm">
              <pre className="whitespace-pre-wrap break-words overflow-auto max-h-[500px]">
                {(() => {
                  try {
                    const data = {
                      ...response,
                      litellm_cache_params: parsedLitellmParams,
                      health_check_cache_params: parsedRedisParams,
                    };
                    // First parse any string JSON values
                    const prettyData = JSON.parse(
                      JSON.stringify(data, (key, value) => {
                        if (typeof value === "string") {
                          try {
                            return JSON.parse(value);
                          } catch {
                            return value;
                          }
                        }
                        return value;
                      }),
                    );
                    // Then stringify with proper formatting
                    return JSON.stringify(prettyData, null, 2);
                  } catch (e) {
                    return "Error formatting JSON: " + (e as Error).message;
                  }
                })()}
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

      {healthCheckResponse && <HealthCheckDetails response={healthCheckResponse} />}
    </div>
  );
};
