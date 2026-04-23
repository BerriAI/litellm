import { BarChart } from "@tremor/react";
import type { DateRangePickerValue } from "@tremor/react";
import React, { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, X } from "lucide-react";
import NotificationsManager from "./molecules/notifications_manager";
import UsageDatePicker from "./shared/usage_date_picker";

import { adminGlobalCacheActivity, cachingHealthCheckCall } from "./networking";

import { CacheHealthTab } from "./cache_health";
import CacheSettings from "./cache_settings";

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
}

interface uiData {
  name: string;
  "LLM API requests": number;
  "Cache hit": number;
  "Cached Completion Tokens": number;
  "Generated Completion Tokens": number;
}

/**
 * Multi-select chips wrapper around shadcn Select. Used for the API-key /
 * model filter rows.
 */
function ChipMultiSelect({
  placeholder,
  options,
  value,
  onChange,
}: {
  placeholder: string;
  options: string[];
  value: string[];
  onChange: (next: string[]) => void;
}) {
  const remaining = options.filter((o) => !value.includes(o));
  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...value, v]);
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              No more options
            </div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt} value={opt}>
                {opt}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {value.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {value.map((v) => (
            <Badge key={v} variant="secondary" className="gap-1">
              {v}
              <button
                type="button"
                onClick={() => onChange(value.filter((s) => s !== v))}
                aria-label={`Remove ${v}`}
              >
                <X size={10} />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

const CacheDashboard: React.FC<CachePageProps> = ({
  accessToken,
  userRole,
  userID,
}) => {
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [healthCheckResponse, setHealthCheckResponse] = useState<any>("");

  useEffect(() => {
    if (!accessToken || !dateValue) return;
    const fetchData = async () => {
      const response = await adminGlobalCacheActivity(
        accessToken,
        formatDateWithoutTZ(dateValue.from),
        formatDateWithoutTZ(dateValue.to),
      );
      setData(response);
    };
    fetchData();
    setLastRefreshed(new Date().toLocaleString());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  const uniqueApiKeys = Array.from(
    new Set(data.map((item) => item?.api_key ?? "")),
  );
  const uniqueModels = Array.from(
    new Set(data.map((item) => item?.model ?? "")),
  );

  const updateCachingData = async (
    startTime: Date | undefined,
    endTime: Date | undefined,
  ) => {
    if (!startTime || !endTime || !accessToken) return;
    const new_cache_data = await adminGlobalCacheActivity(
      accessToken,
      formatDateWithoutTZ(startTime),
      formatDateWithoutTZ(endTime),
    );
    setData(new_cache_data);
  };

  useEffect(() => {
    let newData: cacheDataItem[] = data;
    if (selectedApiKeys.length > 0) {
      newData = newData.filter((item) =>
        selectedApiKeys.includes(item.api_key),
      );
    }
    if (selectedModels.length > 0) {
      newData = newData.filter((item) => selectedModels.includes(item.model));
    }

    let llm_api_requests = 0;
    let cache_hits = 0;
    let cached_tokens = 0;
    const processedData = newData.reduce((acc: uiData[], item) => {
      if (!item.call_type) item.call_type = "Unknown";
      llm_api_requests +=
        (item.total_rows || 0) - (item.cache_hit_true_rows || 0);
      cache_hits += item.cache_hit_true_rows || 0;
      cached_tokens += item.cached_completion_tokens || 0;
      const existingItem = acc.find((i) => i.name === item.call_type);
      if (existingItem) {
        existingItem["LLM API requests"] +=
          (item.total_rows || 0) - (item.cache_hit_true_rows || 0);
        existingItem["Cache hit"] += item.cache_hit_true_rows || 0;
        existingItem["Cached Completion Tokens"] +=
          item.cached_completion_tokens || 0;
        existingItem["Generated Completion Tokens"] +=
          item.generated_completion_tokens || 0;
      } else {
        acc.push({
          name: item.call_type,
          "LLM API requests":
            (item.total_rows || 0) - (item.cache_hit_true_rows || 0),
          "Cache hit": item.cache_hit_true_rows || 0,
          "Cached Completion Tokens": item.cached_completion_tokens || 0,
          "Generated Completion Tokens": item.generated_completion_tokens || 0,
        });
      }
      return acc;
    }, []);

    setCachedResponses(valueFormatterNumbers(cache_hits));
    setCachedTokens(valueFormatterNumbers(cached_tokens));
    const allRequests = cache_hits + llm_api_requests;
    setCacheHitRatio(
      allRequests > 0 ? ((cache_hits / allRequests) * 100).toFixed(2) : "0",
    );
    setFilteredData(processedData);
  }, [selectedApiKeys, selectedModels, dateValue, data]);

  const handleRefreshClick = () => {
    setLastRefreshed(new Date().toLocaleString());
  };

  const runCachingHealthCheck = async () => {
    try {
      NotificationsManager.info("Running cache health check...");
      setHealthCheckResponse("");
      const response = await cachingHealthCheckCall(
        accessToken !== null ? accessToken : "",
      );
      setHealthCheckResponse(response);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
      console.error("Error running health check:", error);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let errorData: any;
      if (error && error.message) {
        try {
          let parsedData = JSON.parse(error.message);
          if (parsedData.error) parsedData = parsedData.error;
          errorData = parsedData;
        } catch {
          errorData = { message: error.message };
        }
      } else {
        errorData = { message: "Unknown error occurred" };
      }
      setHealthCheckResponse({ error: errorData });
    }
  };

  return (
    <Tabs
      defaultValue="analytics"
      className="gap-2 p-8 h-full w-full mt-2 mb-8"
    >
      <div className="flex justify-between mt-2 w-full items-center">
        <TabsList>
          <TabsTrigger value="analytics">Cache Analytics</TabsTrigger>
          <TabsTrigger value="health">Cache Health</TabsTrigger>
          <TabsTrigger value="settings">Cache Settings</TabsTrigger>
        </TabsList>

        <div className="flex items-center space-x-2">
          {lastRefreshed && (
            <span className="text-sm text-muted-foreground">
              Last Refreshed: {lastRefreshed}
            </span>
          )}
          <Button
            variant="outline"
            size="icon"
            onClick={handleRefreshClick}
            aria-label="Refresh"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <TabsContent value="analytics">
        <Card className="p-6">
          <div className="grid grid-cols-3 gap-4 mt-4">
            <div>
              <ChipMultiSelect
                placeholder="Select Virtual Keys"
                options={uniqueApiKeys}
                value={selectedApiKeys}
                onChange={setSelectedApiKeys}
              />
            </div>
            <div>
              <ChipMultiSelect
                placeholder="Select Models"
                options={uniqueModels}
                value={selectedModels}
                onChange={setSelectedModels}
              />
            </div>
            <div>
              <UsageDatePicker
                value={dateValue}
                onValueChange={(value) => {
                  setDateValue(value);
                  updateCachingData(value.from, value.to);
                }}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 mt-4">
            <Card className="p-4">
              <p className="text-sm font-medium text-muted-foreground">
                Cache Hit Ratio
              </p>
              <div className="mt-2 flex items-baseline space-x-2.5">
                <p className="text-3xl font-semibold text-foreground">
                  {cacheHitRatio}%
                </p>
              </div>
            </Card>
            <Card className="p-4">
              <p className="text-sm font-medium text-muted-foreground">
                Cache Hits
              </p>
              <div className="mt-2 flex items-baseline space-x-2.5">
                <p className="text-3xl font-semibold text-foreground">
                  {cachedResponses}
                </p>
              </div>
            </Card>
            <Card className="p-4">
              <p className="text-sm font-medium text-muted-foreground">
                Cached Tokens
              </p>
              <div className="mt-2 flex items-baseline space-x-2.5">
                <p className="text-3xl font-semibold text-foreground">
                  {cachedTokens}
                </p>
              </div>
            </Card>
          </div>

          <p className="mt-4 text-sm text-muted-foreground">
            Cache Hits vs API Requests
          </p>
          <BarChart
            title="Cache Hits vs API Requests"
            data={filteredData}
            stack={true}
            index="name"
            valueFormatter={valueFormatterNumbers}
            categories={["LLM API requests", "Cache hit"]}
            colors={["sky", "teal"]}
            yAxisWidth={48}
          />

          <p className="mt-4 text-sm text-muted-foreground">
            Cached Completion Tokens vs Generated Completion Tokens
          </p>
          <BarChart
            className="mt-6"
            data={filteredData}
            stack={true}
            index="name"
            valueFormatter={valueFormatterNumbers}
            categories={[
              "Generated Completion Tokens",
              "Cached Completion Tokens",
            ]}
            colors={["sky", "teal"]}
            yAxisWidth={48}
          />
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
        <CacheSettings
          accessToken={accessToken}
          userRole={userRole}
          userID={userID}
        />
      </TabsContent>
    </Tabs>
  );
};

export default CacheDashboard;
