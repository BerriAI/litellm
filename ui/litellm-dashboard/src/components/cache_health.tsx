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
    setTimeout(() => {
      setCopied(false);
    }, 2000);
  };

  return (
    <tr className="border-t first:border-t-0">
      <td className="px-6 py-4 align-top w-full" colSpan={2}>
        <div className="flex items-center">
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
          <button
            onClick={handleCopy}
            className="ml-2 text-gray-500 hover:text-gray-700 focus:outline-none"
            title="Copy to clipboard"
          >
            <ClipboardCopyIcon className="h-5 w-5" />
            <span className="sr-only">{copied ? "Copied" : "Copy"}</span>
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

// Update HealthCheckDetails component to include Redis info
const HealthCheckDetails: React.FC<{ response: any }> = ({ response }) => {
  // Parse the JSON strings in the response
  const parsedLitellmParams = deepParse(response.litellm_cache_params);
  const parsedRedisParams = deepParse(response.redis_cache_params);
  
  // Extract Redis details from parsed response, checking multiple possible paths
  const redisDetails: RedisDetails = {
    redis_host: parsedRedisParams?.redis_kwargs?.startup_nodes?.[0]?.host || 
                parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs?.host ||
                parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs?.host ||
                "N/A",
    
    redis_port: parsedRedisParams?.redis_kwargs?.startup_nodes?.[0]?.port ||
                parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs?.port ||
                parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs?.port ||
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
    <div>
      <TabGroup>
        <TabList className="mb-6">
          <Tab>Summary</Tab>
          <Tab>Raw Response</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <div>
              <div className="flex items-center mb-4">
                {response.status === "healthy" ? (
                  <CheckCircleIcon className="h-6 w-6 text-green-500 mr-2" />
                ) : (
                  <XCircleIcon className="h-6 w-6 text-red-500 mr-2" />
                )}
                <Text className={response.status === "healthy" ? "text-green-500" : "text-red-500"}>
                  Cache Status: {response.status}
                </Text>
              </div>

              <table className="min-w-full">
                <tbody>
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
                  
                  {/* Add Redis Details Section */}
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

          <TabPanel>
            <Card className="max-w-screen-lg">
              <pre className="bg-gray-100 p-4 rounded-md overflow-auto max-h-[500px]">
                {JSON.stringify({
                  ...response,
                  litellm_cache_params: parsedLitellmParams,
                  redis_cache_params: parsedRedisParams
                }, null, 2)}
              </pre>
            </Card>
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
    <div className="p-4">
      <Button 
        onClick={handleHealthCheck}
        disabled={isLoading}
        className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-md"
      >
        {isLoading ? "Running Cache Health..." : "Run cache health"}
      </Button>

      <div className="flex items-center justify-end">
        <ResponseTimeIndicator responseTimeMs={localResponseTimeMs} />
      </div>

      {healthCheckResponse && (
        <div className="mt-4">
          <HealthCheckDetails response={healthCheckResponse} />
        </div>
      )}
    </div>
  );
}; 