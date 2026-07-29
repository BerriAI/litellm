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
import { adminGlobalCacheActivity, cachingHealthCheckCall } from "@/components/networking";

// Import the new component
import { CacheHealthTab } from "./cache_health";
import CacheSettings from "./cache_settings";
import CoordinationRedisSettings from "./coordination_redis_settings";

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

interface cacheDataItem {
  api_key: string;
  model: string;
  cache_hit_true_rows: number;
  cached_completion_tokens: number;
  total_rows: number;
  generated_completion_tokens: number;
  call_type: string;

  // Add other properties as needed
}

type uiData = {
  name: string;
  "LLM API requests": number;
  "Cache hit": number;
  "Cached Completion Tokens": number;
  "Generated Completion Tokens": number;
};

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
  const [filteredData, setFilteredData] = useState<uiData[]>([]);
  const [selectedApiKeys, setSelectedApiKeys] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [data, setData] = useState<cacheDataItem[]>([]);
  const [cachedResponses, setCachedResponses] = useState("0");
  const [cachedTokens, setCachedTokens] = useState("0");
  const [cacheHitRatio, setCacheHitRatio] = useState("0");

  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });

  const [lastRefreshed, setLastRefreshed] = useState("");
  const [healthCheckResponse, setHealthCheckResponse] = useState<any>("");

  useEffect(() => {
    if (!accessToken || !dateValue) {
      return;
    }
    const fetchData = async () => {
      const response = await adminGlobalCacheActivity(
        accessToken,
        formatDateWithoutTZ(dateValue.from),
        formatDateWithoutTZ(dateValue.to),
      );
      setData(response);
    };
    fetchData();

    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleString());
  }, [accessToken]);

  const uniqueApiKeys = Array.from(new Set(data.map((item) => item?.api_key ?? "")));
  const uniqueModels = Array.from(new Set(data.map((item) => item?.model ?? "")));
  const uniqueCallTypes = Array.from(new Set(data.map((item) => item?.call_type ?? "")));

  const updateCachingData = async (startTime: Date | undefined, endTime: Date | undefined) => {
    if (!startTime || !endTime || !accessToken) {
      return;
    }

    let new_cache_data = await adminGlobalCacheActivity(
      accessToken,
      formatDateWithoutTZ(startTime),
      formatDateWithoutTZ(endTime),
    );

    setData(new_cache_data);
  };

  useEffect(() => {
    let newData: cacheDataItem[] = data;
    if (selectedApiKeys.length > 0) {
      newData = newData.filter((item) => selectedApiKeys.includes(item.api_key));
    }

    if (selectedModels.length > 0) {
      newData = newData.filter((item) => selectedModels.includes(item.model));
    }

    /* 
    Data looks like this 
    [{"api_key":"sk-test-mock-key-001","call_type":"acompletion","model":"llama3-8b-8192","total_rows":13,"cache_hit_true_rows":0},
    {"api_key":"sk-test-mock-key-002","call_type":"None","model":"chatgpt-v-2","total_rows":1,"cache_hit_true_rows":0},
    {"api_key":"sk-test-mock-key-123","call_type":"acompletion","model":"gpt-3.5-turbo","total_rows":19,"cache_hit_true_rows":0},
    {"api_key":"sk-test-mock-key-123","call_type":"aimage_generation","model":"","total_rows":3,"cache_hit_true_rows":0},
    {"api_key":"sk-test-mock-key-003","call_type":"None","model":"chatgpt-v-2","total_rows":1,"cache_hit_true_rows":0},
    {"api_key":"sk-test-mock-key-004","call_type":"","model":"chatgpt-v-2","total_rows":1,"cache_hit_true_rows":0},
    {"api_key":"sk-test-mock-key-005","call_type":"","model":"chatgpt-v-2","total_rows":1,"cache_hit_true_rows":0},
    */

    // What data we need for bar chat
    // ui_data = [
    //     {
    //         name: "Call Type",
    //         Cache hit: 20,
    //         LLM API requests: 10,
    //     }
    // ]

    let llm_api_requests = 0;
    let cache_hits = 0;
    let cached_tokens = 0;
    const processedData = newData.reduce((acc: uiData[], item) => {
      if (!item.call_type) {
        item.call_type = "Unknown";
      }

      llm_api_requests += (item.total_rows || 0) - (item.cache_hit_true_rows || 0);
      cache_hits += item.cache_hit_true_rows || 0;
      cached_tokens += item.cached_completion_tokens || 0;

      const existingItem = acc.find((i) => i.name === item.call_type);
      if (existingItem) {
        existingItem["LLM API requests"] += (item.total_rows || 0) - (item.cache_hit_true_rows || 0);
        existingItem["Cache hit"] += item.cache_hit_true_rows || 0;
        existingItem["Cached Completion Tokens"] += item.cached_completion_tokens || 0;
        existingItem["Generated Completion Tokens"] += item.generated_completion_tokens || 0;
      } else {
        acc.push({
          name: item.call_type,
          "LLM API requests": (item.total_rows || 0) - (item.cache_hit_true_rows || 0),
          "Cache hit": item.cache_hit_true_rows || 0,
          "Cached Completion Tokens": item.cached_completion_tokens || 0,
          "Generated Completion Tokens": item.generated_completion_tokens || 0,
        });
      }
      return acc;
    }, []);

    // set header cache statistics
    setCachedResponses(valueFormatterNumbers(cache_hits));
    setCachedTokens(valueFormatterNumbers(cached_tokens));
    let allRequests = cache_hits + llm_api_requests;
    if (allRequests > 0) {
      let cache_hit_ratio = ((cache_hits / allRequests) * 100).toFixed(2);
      setCacheHitRatio(cache_hit_ratio);
    } else {
      setCacheHitRatio("0");
    }

    setFilteredData(processedData);
  }, [selectedApiKeys, selectedModels, dateValue, data]);

  const handleRefreshClick = () => {
    // Update the 'lastRefreshed' state to the current date and time
    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleString());
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

  const statCards = [
    { label: "Cache Hit Ratio", value: `${cacheHitRatio}%` },
    { label: "Cache Hits", value: cachedResponses },
    { label: "Cached Completion Tokens", value: cachedTokens },
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
                  updateCachingData(value.from, value.to);
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
                  data={filteredData}
                  stack={true}
                  index="name"
                  valueFormatter={valueFormatterNumbers}
                  categories={["LLM API requests", "Cache hit"]}
                  colors={["sky", "teal"]}
                  yAxisWidth={48}
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
                  data={filteredData}
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
