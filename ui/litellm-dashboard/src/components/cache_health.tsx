import React from "react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CheckCircle, ClipboardCopy, XCircle } from "lucide-react";
import { ResponseTimeIndicator } from "./response_time_indicator";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
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

const TableClickableErrorField: React.FC<{
  label: string;
  value: string | null | undefined;
}> = ({ label, value }) => {
  const [isExpanded, setIsExpanded] = React.useState(false);
  const [, setCopied] = React.useState(false);
  const safeValue = value?.toString() || "N/A";
  const truncated =
    safeValue.length > 50 ? safeValue.substring(0, 50) + "..." : safeValue;

  const handleCopy = () => {
    navigator.clipboard.writeText(safeValue);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <tr className="hover:bg-muted">
      <td className="px-4 py-2 align-top" colSpan={2}>
        <div className="flex items-center justify-between group">
          <div className="flex items-center flex-1">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-muted-foreground hover:text-foreground mr-2"
            >
              {isExpanded ? "▼" : "▶"}
            </button>
            <div>
              <div className="text-sm text-muted-foreground">{label}</div>
              <pre className="mt-1 text-sm font-mono text-foreground whitespace-pre-wrap">
                {isExpanded ? safeValue : truncated}
              </pre>
            </div>
          </div>
          <button
            onClick={handleCopy}
            className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground"
            aria-label="Copy"
          >
            <ClipboardCopy className="h-4 w-4" />
          </button>
        </div>
      </td>
    </tr>
  );
};

interface RedisDetails {
  redis_host?: string;
  redis_port?: string;
  redis_version?: string;
  startup_nodes?: string;
  namespace?: string;
}

interface ErrorDetails {
  message: string;
  traceback: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  litellm_params?: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  health_check_cache_params?: any;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const HealthCheckDetails: React.FC<{ response: any }> = ({ response }) => {
  let errorDetails: ErrorDetails | null = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let parsedLitellmParams: any = {};
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let parsedRedisParams: any = {};

  try {
    if (response?.error) {
      try {
        const errorMessage =
          typeof response.error.message === "string"
            ? JSON.parse(response.error.message)
            : response.error.message;

        errorDetails = {
          message: errorMessage?.message || "Unknown error",
          traceback: errorMessage?.traceback || "No traceback available",
          litellm_params: errorMessage?.litellm_cache_params || {},
          health_check_cache_params:
            errorMessage?.health_check_cache_params || {},
        };

        parsedLitellmParams = deepParse(errorDetails.litellm_params) || {};
        parsedRedisParams =
          deepParse(errorDetails.health_check_cache_params) || {};
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
    parsedLitellmParams = {};
    parsedRedisParams = {};
  }

  const redisDetails: RedisDetails = {
    redis_host:
      parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs
        ?.host ||
      parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs
        ?.host ||
      parsedRedisParams?.connection_kwargs?.host ||
      parsedRedisParams?.host ||
      "N/A",
    redis_port:
      parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs
        ?.port ||
      parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs
        ?.port ||
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
          parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs
            ?.host ||
          parsedRedisParams?.redis_async_client?.connection_pool
            ?.connection_kwargs?.host;
        const port =
          parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs
            ?.port ||
          parsedRedisParams?.redis_async_client?.connection_pool
            ?.connection_kwargs?.port;
        return host && port ? JSON.stringify([{ host, port }]) : "N/A";
      } catch {
        return "N/A";
      }
    })(),
    namespace: parsedRedisParams?.namespace || "N/A",
  };

  return (
    <div className="bg-background rounded-lg shadow border border-border">
      <Tabs defaultValue="summary">
        <div className="border-b border-border px-4">
          <TabsList>
            <TabsTrigger value="summary">Summary</TabsTrigger>
            <TabsTrigger value="raw">Raw Response</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="summary" className="p-4">
          <div>
            <div className="flex items-center mb-6">
              {response?.status === "healthy" ? (
                <CheckCircle className="h-5 w-5 text-emerald-500 mr-2" />
              ) : (
                <XCircle className="h-5 w-5 text-red-500 mr-2" />
              )}
              <span
                className={`text-sm font-medium ${
                  response?.status === "healthy"
                    ? "text-emerald-500"
                    : "text-red-500"
                }`}
              >
                Cache Status: {response?.status || "unhealthy"}
              </span>
            </div>

            <table className="w-full border-collapse">
              <tbody>
                {errorDetails && (
                  <>
                    <tr>
                      <td
                        colSpan={2}
                        className="pt-4 pb-2 font-semibold text-destructive"
                      >
                        Error Details
                      </td>
                    </tr>
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

                <tr>
                  <td colSpan={2} className="pt-4 pb-2 font-semibold">
                    Cache Details
                  </td>
                </tr>
                <TableClickableErrorField
                  label="Cache Configuration"
                  value={String(parsedLitellmParams?.type)}
                />
                <TableClickableErrorField
                  label="Ping Response"
                  value={String(response.ping_response)}
                />
                <TableClickableErrorField
                  label="Set Cache Response"
                  value={response.set_cache_response || "N/A"}
                />
                <TableClickableErrorField
                  label="litellm_settings.cache_params"
                  value={JSON.stringify(parsedLitellmParams, null, 2)}
                />

                {parsedLitellmParams?.type === "redis" && (
                  <>
                    <tr>
                      <td colSpan={2} className="pt-4 pb-2 font-semibold">
                        Redis Details
                      </td>
                    </tr>
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
                    <TableClickableErrorField
                      label="Namespace"
                      value={redisDetails.namespace || "N/A"}
                    />
                  </>
                )}
              </tbody>
            </table>
          </div>
        </TabsContent>

        <TabsContent value="raw" className="p-4">
          <div className="bg-muted rounded-md p-4 font-mono text-sm">
            <pre className="whitespace-pre-wrap break-words overflow-auto max-h-[500px]">
              {(() => {
                try {
                  const data = {
                    ...response,
                    litellm_cache_params: parsedLitellmParams,
                    health_check_cache_params: parsedRedisParams,
                  };
                  const prettyData = JSON.parse(
                    JSON.stringify(data, (_key, value) => {
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
                  return JSON.stringify(prettyData, null, 2);
                } catch (e) {
                  return "Error formatting JSON: " + (e as Error).message;
                }
              })()}
            </pre>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export const CacheHealthTab: React.FC<{
  accessToken: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  healthCheckResponse: any;
  runCachingHealthCheck: () => void;
  responseTimeMs?: number | null;
}> = ({ healthCheckResponse, runCachingHealthCheck }) => {
  const [localResponseTimeMs, setLocalResponseTimeMs] = React.useState<
    number | null
  >(null);
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
        <Button onClick={handleHealthCheck} disabled={isLoading}>
          {isLoading ? "Running Health Check..." : "Run Health Check"}
        </Button>
        <ResponseTimeIndicator responseTimeMs={localResponseTimeMs} />
      </div>

      {healthCheckResponse && <HealthCheckDetails response={healthCheckResponse} />}
    </div>
  );
};
