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
import React, { useEffect, useState } from "react";
import NotificationsManager from "@/components/molecules/notifications_manager";
import UsageDatePicker from "@/components/shared/usage_date_picker";
import { BarChart } from "@/components/shared/charts";
import { Card as ChartCard, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

import { RefreshIcon } from "@heroicons/react/outline";
import { adminGlobalCacheActivity, cachingHealthCheckCall } from "@/components/networking";

// Import the new component
import { CacheHealthTab } from "./cache_health";
import CacheSettings from "./cache_settings";
import CoordinationRedisSettings from "./coordination_redis_settings";
import { buildCacheDashboardMetrics, type CacheChartData, type CacheDataItem } from "./cacheDashboardMetrics";

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
  const [filteredData, setFilteredData] = useState<CacheChartData[]>([]);
  const [selectedApiKeys, setSelectedApiKeys] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [data, setData] = useState<CacheDataItem[]>([]);
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
    let newData: CacheDataItem[] = data;
    if (selectedApiKeys.length > 0) {
      newData = newData.filter((item) => selectedApiKeys.includes(item.api_key ?? ""));
    }

    if (selectedModels.length > 0) {
      newData = newData.filter((item) => selectedModels.includes(item.model ?? ""));
    }

    const metrics = buildCacheDashboardMetrics(newData);

    setCachedResponses(valueFormatterNumbers(metrics.cacheHits));
    setCachedTokens(valueFormatterNumbers(metrics.cachedTokens));
    setCacheHitRatio(metrics.cacheHitRatio);
    setFilteredData(metrics.chartData);
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
            <Text className="text-tremor-content dark:text-dark-tremor-content">
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
            </Text>
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
                  Cached Completion Tokens
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
                <CardTitle className="text-base font-semibold">Cached Tokens vs Generated Completion Tokens</CardTitle>
              </CardHeader>
              <CardContent>
                <BarChart
                  data={filteredData}
                  stack={true}
                  index="name"
                  valueFormatter={valueFormatterNumbers}
                  categories={[
                    "Generated Completion Tokens",
                    "Cached Completion Tokens",
                    "Provider Prompt Cache Tokens",
                  ]}
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
