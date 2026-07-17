import {
  Card,
  Col,
  DateRangePickerValue,
  Grid,
  Icon,
  MultiSelect,
  MultiSelectItem,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Text,
} from "@tremor/react";
import React, { useEffect, useMemo, useState } from "react";
import NotificationsManager from "@/components/molecules/notifications_manager";
import UsageDatePicker from "@/components/shared/usage_date_picker";
import { BarChart } from "@/components/shared/charts";
import { Card as ChartCard, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

import { RefreshIcon } from "@heroicons/react/outline";
import {
  adminGlobalCacheActivity,
  adminGlobalPromptCacheActivity,
  cachingHealthCheckCall,
  PromptCacheActivityItem,
} from "@/components/networking";

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

type promptCacheUiData = {
  name: string;
  "Uncached Input Tokens": number;
  "Cache Read Input Tokens": number;
  "Cache Creation Input Tokens": number;
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
  const [promptCacheData, setPromptCacheData] = useState<PromptCacheActivityItem[]>([]);
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
      const promptCacheResponse = await adminGlobalPromptCacheActivity(
        accessToken,
        formatDateWithoutTZ(dateValue.from),
        formatDateWithoutTZ(dateValue.to),
      );
      setPromptCacheData(promptCacheResponse);
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

    let new_prompt_cache_data = await adminGlobalPromptCacheActivity(
      accessToken,
      formatDateWithoutTZ(startTime),
      formatDateWithoutTZ(endTime),
    );

    setPromptCacheData(new_prompt_cache_data);
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

  const { promptCacheChart, promptCacheReadTokens, promptCacheCreationTokens } = useMemo(() => {
    const rows = promptCacheData
      .filter((item) => selectedApiKeys.length === 0 || selectedApiKeys.includes(item.api_key))
      .filter((item) => selectedModels.length === 0 || (item.model !== null && selectedModels.includes(item.model)));

    const byModel = new Map<string, promptCacheUiData>();
    for (const item of rows) {
      const cacheRead = item.cache_read_input_tokens || 0;
      const cacheCreation = item.cache_creation_input_tokens || 0;
      const uncached = Math.max((item.prompt_tokens || 0) - cacheRead - cacheCreation, 0);
      const name = item.model || "Unknown";
      const existing = byModel.get(name);
      byModel.set(name, {
        name,
        "Uncached Input Tokens": (existing?.["Uncached Input Tokens"] ?? 0) + uncached,
        "Cache Read Input Tokens": (existing?.["Cache Read Input Tokens"] ?? 0) + cacheRead,
        "Cache Creation Input Tokens": (existing?.["Cache Creation Input Tokens"] ?? 0) + cacheCreation,
      });
    }

    const sum = (pick: (item: PromptCacheActivityItem) => number) =>
      rows.reduce((total, item) => total + (pick(item) || 0), 0);

    return {
      promptCacheChart: Array.from(byModel.values()),
      promptCacheReadTokens: valueFormatterNumbers(sum((item) => item.cache_read_input_tokens)),
      promptCacheCreationTokens: valueFormatterNumbers(sum((item) => item.cache_creation_input_tokens)),
    };
  }, [selectedApiKeys, selectedModels, promptCacheData]);

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

  return (
    <TabGroup className="gap-2 p-8 h-full w-full mt-2 mb-8">
      <TabList className="flex justify-between mt-2 w-full items-center">
        <div className="flex">
          <Tab>Cache Analytics</Tab>
          <Tab>Cache Health</Tab>
          <Tab>Cache Settings</Tab>
          <Tab>Coordination Redis</Tab>
        </div>

        <div className="flex items-center space-x-2">
          {lastRefreshed && <Text>Last Refreshed: {lastRefreshed}</Text>}
          <Icon
            icon={RefreshIcon} // Modify as necessary for correct icon name
            variant="shadow"
            size="xs"
            className="self-center"
            onClick={handleRefreshClick}
          />
        </div>
      </TabList>
      <TabPanels>
        <TabPanel>
          <Card>
            <Grid numItems={3} className="gap-4 mt-4">
              <Col>
                <MultiSelect
                  placeholder="Select Virtual Keys"
                  value={selectedApiKeys}
                  onValueChange={setSelectedApiKeys}
                >
                  {uniqueApiKeys.map((key) => (
                    <MultiSelectItem key={key} value={key}>
                      {key}
                    </MultiSelectItem>
                  ))}
                </MultiSelect>
              </Col>
              <Col>
                <MultiSelect placeholder="Select Models" value={selectedModels} onValueChange={setSelectedModels}>
                  {uniqueModels.map((model) => (
                    <MultiSelectItem key={model} value={model}>
                      {model}
                    </MultiSelectItem>
                  ))}
                </MultiSelect>
              </Col>
              <Col>
                <UsageDatePicker
                  value={dateValue}
                  onValueChange={(value) => {
                    setDateValue(value);
                    updateCachingData(value.from, value.to);
                  }}
                />
              </Col>
            </Grid>

            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 mt-4">
              <Card>
                <p className="text-tremor-default font-medium text-tremor-content dark:text-dark-tremor-content">
                  Cache Hit Ratio
                </p>
                <div className="mt-2 flex items-baseline space-x-2.5">
                  <p className="text-tremor-metric font-semibold text-tremor-content-strong dark:text-dark-tremor-content-strong">
                    {cacheHitRatio}%
                  </p>
                </div>
              </Card>
              <Card>
                <p className="text-tremor-default font-medium text-tremor-content dark:text-dark-tremor-content">
                  Cache Hits
                </p>
                <div className="mt-2 flex items-baseline space-x-2.5">
                  <p className="text-tremor-metric font-semibold text-tremor-content-strong dark:text-dark-tremor-content-strong">
                    {cachedResponses}
                  </p>
                </div>
              </Card>

              <Card>
                <p className="text-tremor-default font-medium text-tremor-content dark:text-dark-tremor-content">
                  Cached Tokens
                </p>
                <div className="mt-2 flex items-baseline space-x-2.5">
                  <p className="text-tremor-metric font-semibold text-tremor-content-strong dark:text-dark-tremor-content-strong">
                    {cachedTokens}
                  </p>
                </div>
              </Card>
            </div>

            <ChartCard className="mt-4">
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
            </ChartCard>

            <ChartCard className="mt-6">
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
            </ChartCard>

            <div className="mt-8">
              <p className="text-tremor-title font-semibold text-tremor-content-strong dark:text-dark-tremor-content-strong">
                Provider Prompt Caching
              </p>
              <p className="text-tremor-default text-tremor-content dark:text-dark-tremor-content mt-1">
                Input tokens cached by the provider (e.g. Anthropic prompt caching). Tracked separately from
                LiteLLM&apos;s response cache above.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 mt-4">
              <Card>
                <p className="text-tremor-default font-medium text-tremor-content dark:text-dark-tremor-content">
                  Cache Read Input Tokens
                </p>
                <div className="mt-2 flex items-baseline space-x-2.5">
                  <p className="text-tremor-metric font-semibold text-tremor-content-strong dark:text-dark-tremor-content-strong">
                    {promptCacheReadTokens}
                  </p>
                </div>
              </Card>
              <Card>
                <p className="text-tremor-default font-medium text-tremor-content dark:text-dark-tremor-content">
                  Cache Creation Input Tokens
                </p>
                <div className="mt-2 flex items-baseline space-x-2.5">
                  <p className="text-tremor-metric font-semibold text-tremor-content-strong dark:text-dark-tremor-content-strong">
                    {promptCacheCreationTokens}
                  </p>
                </div>
              </Card>
            </div>

            <ChartCard className="mt-4">
              <CardHeader>
                <CardTitle className="text-base font-semibold">Prompt Cache Input Tokens by Model</CardTitle>
              </CardHeader>
              <CardContent>
                <BarChart
                  data={promptCacheChart}
                  stack={true}
                  index="name"
                  valueFormatter={valueFormatterNumbers}
                  categories={["Uncached Input Tokens", "Cache Read Input Tokens", "Cache Creation Input Tokens"]}
                  colors={["sky", "teal", "indigo"]}
                  yAxisWidth={48}
                />
              </CardContent>
            </ChartCard>
          </Card>
        </TabPanel>
        <TabPanel>
          <CacheHealthTab
            accessToken={accessToken}
            healthCheckResponse={healthCheckResponse}
            runCachingHealthCheck={runCachingHealthCheck}
          />
        </TabPanel>
        <TabPanel>
          <CacheSettings accessToken={accessToken} userRole={userRole} userID={userID} />
        </TabPanel>
        <TabPanel>
          <CoordinationRedisSettings />
        </TabPanel>
      </TabPanels>
    </TabGroup>
  );
};

export default CacheDashboard;
