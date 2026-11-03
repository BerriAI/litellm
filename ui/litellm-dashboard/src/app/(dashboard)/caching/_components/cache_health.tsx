import React from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CheckCircle2, XCircle, ClipboardCopy } from "lucide-react";
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
  const { t } = useTranslation();
  const [isExpanded, setIsExpanded] = React.useState(false);
  const safeValue = value?.toString() || t("caching.health.notAvailable");
  const truncated = safeValue.length > 50 ? safeValue.substring(0, 50) + "..." : safeValue;

  const handleCopy = () => {
    navigator.clipboard.writeText(safeValue);
  };

  return (
    <tr className="hover:bg-muted/50">
      <td className="px-4 py-2 align-top" colSpan={2}>
        <div className="group flex items-center justify-between">
          <div className="flex flex-1 items-center">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="mr-2 text-muted-foreground hover:text-foreground"
            >
              {isExpanded ? "▼" : "▶"}
            </button>
            <div>
              <div className="text-sm text-muted-foreground">{label}</div>
              <pre className="mt-1 font-mono text-sm whitespace-pre-wrap">{isExpanded ? safeValue : truncated}</pre>
            </div>
          </div>
          <button
            onClick={handleCopy}
            className="text-muted-foreground opacity-0 group-hover:opacity-100 hover:text-foreground"
          >
            <ClipboardCopy className="size-4" />
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
  const { t } = useTranslation();
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
          message: errorMessage?.message || t("caching.health.unknownError"),
          traceback: errorMessage?.traceback || t("caching.health.noTraceback"),
          litellm_params: errorMessage?.litellm_cache_params || {},
          health_check_cache_params: errorMessage?.health_check_cache_params || {},
        };

        parsedLitellmParams = deepParse(errorDetails.litellm_params) || {};
        parsedRedisParams = deepParse(errorDetails.health_check_cache_params) || {};
      } catch (e) {
        console.warn("Error parsing error details:", e);
        errorDetails = {
          message: String(response.error.message || t("caching.health.unknownError")),
          traceback: t("caching.health.parseFailed"),
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
      t("caching.health.notAvailable"),

    redis_port:
      parsedRedisParams?.redis_client?.connection_pool?.connection_kwargs?.port ||
      parsedRedisParams?.redis_async_client?.connection_pool?.connection_kwargs?.port ||
      parsedRedisParams?.connection_kwargs?.port ||
      parsedRedisParams?.port ||
      t("caching.health.notAvailable"),

    redis_version: parsedRedisParams?.redis_version || t("caching.health.notAvailable"),

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
        return host && port ? JSON.stringify([{ host, port }]) : t("caching.health.notAvailable");
      } catch (e) {
        return t("caching.health.notAvailable");
      }
    })(),

    namespace: parsedRedisParams?.namespace || t("caching.health.notAvailable"),
  };

  return (
    <div className="rounded-lg bg-card shadow-sm">
      <Tabs defaultValue="summary">
        <TabsList className="border-b border-border px-4">
          <TabsTrigger value="summary" className="flex-none">
            {t("caching.health.tabs.summary")}
          </TabsTrigger>
          <TabsTrigger value="raw" className="flex-none">
            {t("caching.health.tabs.raw")}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="summary" className="p-4" keepMounted>
          <div>
            <div className="mb-6 flex items-center">
              {response?.status === "healthy" ? (
                <CheckCircle2 className="mr-2 size-5 text-success" />
              ) : (
                <XCircle className="mr-2 size-5 text-destructive" />
              )}
              <p
                className={`text-sm font-medium ${response?.status === "healthy" ? "text-success" : "text-destructive"}`}
              >
                {t("caching.health.cacheStatus")}
                {response?.status || "unhealthy"}
              </p>
            </div>

            <table className="w-full border-collapse">
              <tbody>
                {/* Show error message if present */}
                {errorDetails && (
                  <>
                    <tr>
                      <td colSpan={2} className="pt-4 pb-2 font-semibold text-destructive">
                        {t("caching.health.errorDetails")}
                      </td>
                    </tr>
                    <TableClickableErrorField label={t("caching.health.errorMessage")} value={errorDetails.message} />
                    <TableClickableErrorField label={t("caching.health.traceback")} value={errorDetails.traceback} />
                  </>
                )}

                {/* Always show cache details, regardless of error state */}
                <tr>
                  <td colSpan={2} className="pt-4 pb-2 font-semibold">
                    {t("caching.health.cacheDetails")}
                  </td>
                </tr>
                <TableClickableErrorField
                  label={t("caching.health.cacheConfiguration")}
                  value={String(parsedLitellmParams?.type)}
                />
                <TableClickableErrorField
                  label={t("caching.health.pingResponse")}
                  value={String(response.ping_response)}
                />
                <TableClickableErrorField
                  label={t("caching.health.setCacheResponse")}
                  value={response.set_cache_response || t("caching.health.notAvailable")}
                />
                <TableClickableErrorField
                  label="litellm_settings.cache_params"
                  value={JSON.stringify(parsedLitellmParams, null, 2)}
                />

                {/* Redis Details Section */}
                {parsedLitellmParams?.type === "redis" && (
                  <>
                    <tr>
                      <td colSpan={2} className="pt-4 pb-2 font-semibold">
                        {t("caching.health.redisDetails")}
                      </td>
                    </tr>
                    <TableClickableErrorField
                      label={t("caching.health.redisHost")}
                      value={redisDetails.redis_host || t("caching.health.notAvailable")}
                    />
                    <TableClickableErrorField
                      label={t("caching.health.redisPort")}
                      value={redisDetails.redis_port || t("caching.health.notAvailable")}
                    />
                    <TableClickableErrorField
                      label={t("caching.health.redisVersion")}
                      value={redisDetails.redis_version || t("caching.health.notAvailable")}
                    />
                    <TableClickableErrorField
                      label={t("caching.health.startupNodes")}
                      value={redisDetails.startup_nodes || t("caching.health.notAvailable")}
                    />
                    <TableClickableErrorField
                      label={t("caching.health.namespace")}
                      value={redisDetails.namespace || t("caching.health.notAvailable")}
                    />
                  </>
                )}
              </tbody>
            </table>
          </div>
        </TabsContent>

        <TabsContent value="raw" className="p-4" keepMounted>
          <div className="rounded-md bg-muted p-4 font-mono text-sm">
            <pre className="whitespace-pre-wrap wrap-break-word overflow-auto max-h-[500px]">
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
                  return t("caching.health.formatFailed", { message: (e as Error).message });
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
  healthCheckResponse: any;
  runCachingHealthCheck: () => void;
  responseTimeMs?: number | null;
}> = ({ accessToken, healthCheckResponse, runCachingHealthCheck, responseTimeMs }) => {
  const { t } = useTranslation();
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
        <Button onClick={handleHealthCheck} disabled={isLoading}>
          {isLoading ? t("caching.health.running") : t("caching.health.runButton")}
        </Button>
        <ResponseTimeIndicator responseTimeMs={localResponseTimeMs} />
      </div>

      {healthCheckResponse && <HealthCheckDetails response={healthCheckResponse} />}
    </div>
  );
};
