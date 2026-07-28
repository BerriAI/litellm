import { DateRangePickerValue } from "@tremor/react";
import React, { useEffect, useState } from "react";
import NotificationsManager from "@/components/molecules/notifications_manager";
import UsageDatePicker from "@/components/shared/usage_date_picker";
import { BarChart } from "@/components/shared/charts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
} from "@/components/ui/combobox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { RefreshCw } from "lucide-react";
import { cachingHealthCheckCall } from "@/components/networking";
import { useCacheActivity, type CacheActivityGroup } from "@/app/(dashboard)/hooks/caching/useCacheActivity";
import { migratedHref } from "@/utils/migratedPages";

// Import the new component
import { CacheHealthTab } from "./cache_health";
import CacheSettings from "./cache_settings";
import CoordinationRedisSettings from "./coordination_redis_settings";

const REQUEST_SERIES = {
  apiRequests: "LLM API requests",
  cacheHits: "Cache hit",
  failed: "Failed requests",
} as const;

const toChartDatum = (group: CacheActivityGroup) => ({
  name: group.call_type,
  [REQUEST_SERIES.apiRequests]: group.api_requests,
  [REQUEST_SERIES.cacheHits]: group.cache_hits,
  [REQUEST_SERIES.failed]: group.failed_requests,
  "Cached Completion Tokens": group.cached_completion_tokens,
  "Generated Completion Tokens": group.generated_completion_tokens,
});

const UNKNOWN_CALL_TYPE = "Unknown";

export const buildLogsDrilldownUrl = (
  clicked: { name: string; categoryClicked: string },
  range: { startDate: string | undefined; endDate: string | undefined },
): string => {
  const params = new URLSearchParams();
  if (clicked.name !== UNKNOWN_CALL_TYPE) {
    params.set("call_type", clicked.name);
  }
  if (clicked.categoryClicked === REQUEST_SERIES.failed) {
    params.set("status", "failure");
  }
  if (clicked.categoryClicked === REQUEST_SERIES.cacheHits) {
    params.set("cache_hit", "true");
  }
  if (clicked.categoryClicked === REQUEST_SERIES.apiRequests) {
    params.set("status", "success");
    params.set("cache_hit", "false");
  }
  if (range.startDate) params.set("start_time", `${range.startDate}T00:00`);
  if (range.endDate) params.set("end_time", `${range.endDate}T23:59`);
  return `${migratedHref("logs")}?${params.toString()}`;
};

const formatDateWithoutTZ = (date: Date | undefined) => {
  if (!date) return undefined;
  return date.toISOString().split("T")[0];
};

function valueFormatterNumbers(number: number) {
  const formatter = new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
    notation: "compact",
    compactDisplay: "short",
  });

  return formatter.format(number);
}

interface CachePageProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  premiumUser: boolean;
}

interface CacheHealthResponse {
  status?: string;
  cache_type?: string;
  ping_response?: boolean;
  set_cache_response?: string;
  litellm_cache_params?: string;
  error?: {
    message: string;
    type: string;
    param: string;
    code: string;
  };
}

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

const CacheDashboard: React.FC<CachePageProps> = ({ accessToken, token, userRole, userID, premiumUser }) => {
  const [selectedApiKeys, setSelectedApiKeys] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);

  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });

  const [lastRefreshed, setLastRefreshed] = useState("");
  const [healthCheckResponse, setHealthCheckResponse] = useState<any>("");

  const { data: activity, refetch } = useCacheActivity({
    startDate: formatDateWithoutTZ(dateValue.from),
    endDate: formatDateWithoutTZ(dateValue.to),
    keyAliases: selectedApiKeys,
    models: selectedModels,
  });

  useEffect(() => {
    setLastRefreshed(new Date().toLocaleString());
  }, []);

  const uniqueApiKeys = activity?.filter_options.key_aliases ?? [];
  const uniqueModels = activity?.filter_options.models ?? [];
  const chartData = (activity?.groups ?? []).map(toChartDatum);

  const handleRefreshClick = () => {
    refetch();
    setLastRefreshed(new Date().toLocaleString());
  };

  const runCachingHealthCheck = async () => {
    try {
      NotificationsManager.info("Running cache health check...");
      setHealthCheckResponse("");
      const response = await cachingHealthCheckCall(accessToken !== null ? accessToken : "");
      setHealthCheckResponse(response);
    } catch (error: any) {
      console.error("Error running health check:", error);
      let errorData;
      if (error && error.message) {
        try {
          // Parse the error message which may contain a nested error layer.
          let parsedData = JSON.parse(error.message);
          // If the parsed object is wrapped (e.g. { error: { ... } }), unwrap it.
          if (parsedData.error) {
            parsedData = parsedData.error;
          }
          errorData = parsedData;
        } catch (e) {
          errorData = { message: error.message };
        }
      } else {
        errorData = { message: "Unknown error occurred" };
      }
      setHealthCheckResponse({ error: errorData });
    }
  };

  const totals = activity?.totals;
  const hasRequests = totals != null && totals.api_requests + totals.cache_hits + totals.failed_requests > 0;
  const statCards = [
    { label: "Cache Hit Ratio", value: `${hasRequests ? totals.cache_hit_ratio.toFixed(2) : "0"}%` },
    { label: "Cache Hits", value: valueFormatterNumbers(totals?.cache_hits ?? 0) },
    { label: "Cached Completion Tokens", value: valueFormatterNumbers(totals?.cached_completion_tokens ?? 0) },
  ];

  return (
    <Tabs defaultValue="analytics" className="mt-2 mb-8 w-full gap-2 p-8">
      <div className="mt-2 flex w-full items-center justify-between">
        <TabsList>
          <TabsTrigger value="analytics" className="flex-none">
            Cache Analytics
          </TabsTrigger>
          <TabsTrigger value="health" className="flex-none">
            Cache Health
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex-none">
            Cache Settings
          </TabsTrigger>
          <TabsTrigger value="coordination" className="flex-none">
            Coordination Redis
          </TabsTrigger>
        </TabsList>

        <div className="flex items-center space-x-2">
          {lastRefreshed && <p className="text-sm text-muted-foreground">Last Refreshed: {lastRefreshed}</p>}
          <Button variant="outline" size="icon-sm" onClick={handleRefreshClick} aria-label="Refresh">
            <RefreshCw />
          </Button>
        </div>
      </div>

      <TabsContent value="analytics">
        <Card>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Analytics for LiteLLM&apos;s{" "}
              <a
                href="https://docs.litellm.ai/docs/proxy/caching"
                target="_blank"
                rel="noreferrer"
                className="underline"
              >
                response cache
              </a>{" "}
              (e.g. Redis / in-memory): requests answered from cache without calling the LLM provider. Provider-side{" "}
              <a
                href="https://docs.litellm.ai/docs/completion/prompt_caching"
                target="_blank"
                rel="noreferrer"
                className="underline"
              >
                prompt caching
              </a>{" "}
              (cached input tokens from Anthropic, OpenAI, etc.) is not shown here; see &quot;Prompt Caching
              Metrics&quot; on the Usage page or individual requests in the Logs page.
            </p>

            <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
              <Combobox
                multiple
                items={uniqueApiKeys}
                value={selectedApiKeys}
                onValueChange={(keys: string[]) => setSelectedApiKeys(keys)}
              >
                <ComboboxChips>
                  <ComboboxValue>
                    {(keys: string[]) =>
                      keys.map((key) => (
                        <ComboboxChip key={key} aria-label={key}>
                          {key}
                        </ComboboxChip>
                      ))
                    }
                  </ComboboxValue>
                  <ComboboxChipsInput placeholder="Select Virtual Keys" className="border-0 bg-transparent" />
                </ComboboxChips>
                <ComboboxContent>
                  <ComboboxEmpty>No virtual keys found</ComboboxEmpty>
                  <ComboboxList>
                    {(key: string) => (
                      <ComboboxItem key={key} value={key}>
                        {key}
                      </ComboboxItem>
                    )}
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>

              <Combobox
                multiple
                items={uniqueModels}
                value={selectedModels}
                onValueChange={(models: string[]) => setSelectedModels(models)}
              >
                <ComboboxChips>
                  <ComboboxValue>
                    {(models: string[]) =>
                      models.map((model) => (
                        <ComboboxChip key={model} aria-label={model}>
                          {model}
                        </ComboboxChip>
                      ))
                    }
                  </ComboboxValue>
                  <ComboboxChipsInput placeholder="Select Models" className="border-0 bg-transparent" />
                </ComboboxChips>
                <ComboboxContent>
                  <ComboboxEmpty>No models found</ComboboxEmpty>
                  <ComboboxList>
                    {(model: string) => (
                      <ComboboxItem key={model} value={model}>
                        {model}
                      </ComboboxItem>
                    )}
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>

              <UsageDatePicker
                value={dateValue}
                onValueChange={(value) => {
                  setDateValue(value);
                }}
              />
            </div>

            <div className="mt-4 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
              {statCards.map((stat) => (
                <Card key={stat.label}>
                  <CardContent>
                    <p className="text-sm font-medium text-muted-foreground">{stat.label}</p>
                    <div className="mt-2 flex items-baseline space-x-2.5">
                      <p className="text-3xl font-semibold">{stat.value}</p>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <Card className="mt-4">
              <CardHeader>
                <CardTitle className="text-base font-semibold">Cache Hits vs API Requests</CardTitle>
              </CardHeader>
              <CardContent>
                <BarChart
                  data={chartData}
                  stack={true}
                  index="name"
                  valueFormatter={valueFormatterNumbers}
                  categories={[REQUEST_SERIES.apiRequests, REQUEST_SERIES.cacheHits, REQUEST_SERIES.failed]}
                  colors={["sky", "teal", "red"]}
                  yAxisWidth={48}
                  className="cursor-pointer"
                  onValueChange={(clicked) =>
                    window.location.assign(
                      buildLogsDrilldownUrl(clicked, {
                        startDate: formatDateWithoutTZ(dateValue.from),
                        endDate: formatDateWithoutTZ(dateValue.to),
                      }),
                    )
                  }
                />
              </CardContent>
            </Card>

            <Card className="mt-6">
              <CardHeader>
                <CardTitle className="text-base font-semibold">
                  Cached Completion Tokens vs Generated Completion Tokens
                </CardTitle>
              </CardHeader>
              <CardContent>
                <BarChart
                  data={chartData}
                  stack={true}
                  index="name"
                  valueFormatter={valueFormatterNumbers}
                  categories={["Generated Completion Tokens", "Cached Completion Tokens"]}
                  colors={["sky", "teal"]}
                  yAxisWidth={48}
                />
              </CardContent>
            </Card>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="health">
        <CacheHealthTab
          accessToken={accessToken}
          healthCheckResponse={healthCheckResponse}
          runCachingHealthCheck={runCachingHealthCheck}
        />
      </TabsContent>

      <TabsContent value="settings">
        <CacheSettings accessToken={accessToken} userRole={userRole} userID={userID} />
      </TabsContent>

      <TabsContent value="coordination">
        <CoordinationRedisSettings />
      </TabsContent>
    </Tabs>
  );
};

export default CacheDashboard;
