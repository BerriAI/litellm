import { Card, Col, DateRangePickerValue, Grid, MultiSelect, MultiSelectItem, Text } from "@tremor/react";
import React, { useEffect, useMemo, useState } from "react";
import UsageDatePicker from "@/components/shared/usage_date_picker";
import { BarChart } from "@/components/shared/charts";
import { Card as ChartCard, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { adminGlobalCacheActivity } from "@/components/networking";

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

interface cacheDataItem {
  api_key: string;
  model: string;
  cache_hit_true_rows: number;
  cached_completion_tokens: number;
  total_rows: number;
  generated_completion_tokens: number;
  call_type: string;
}

type uiData = {
  name: string;
  "LLM API requests": number;
  "Cache hit": number;
  "Cached Completion Tokens": number;
  "Generated Completion Tokens": number;
};

const aggregateCacheData = (rows: cacheDataItem[]) => {
  const chartData = rows.reduce<uiData[]>((acc, item) => {
    const name = item.call_type || "Unknown";
    const apiRequests = (item.total_rows || 0) - (item.cache_hit_true_rows || 0);
    const existing = acc.find((row) => row.name === name);
    if (existing) {
      existing["LLM API requests"] += apiRequests;
      existing["Cache hit"] += item.cache_hit_true_rows || 0;
      existing["Cached Completion Tokens"] += item.cached_completion_tokens || 0;
      existing["Generated Completion Tokens"] += item.generated_completion_tokens || 0;
      return acc;
    }
    return [
      ...acc,
      {
        name,
        "LLM API requests": apiRequests,
        "Cache hit": item.cache_hit_true_rows || 0,
        "Cached Completion Tokens": item.cached_completion_tokens || 0,
        "Generated Completion Tokens": item.generated_completion_tokens || 0,
      },
    ];
  }, []);

  const totals = rows.reduce(
    (acc, item) => ({
      cacheHits: acc.cacheHits + (item.cache_hit_true_rows || 0),
      cachedTokens: acc.cachedTokens + (item.cached_completion_tokens || 0),
      apiRequests: acc.apiRequests + ((item.total_rows || 0) - (item.cache_hit_true_rows || 0)),
    }),
    { cacheHits: 0, cachedTokens: 0, apiRequests: 0 },
  );

  const allRequests = totals.cacheHits + totals.apiRequests;
  const cacheHitRatio = allRequests > 0 ? ((totals.cacheHits / allRequests) * 100).toFixed(2) : "0";

  return {
    chartData,
    cacheHitRatio,
    cachedResponses: valueFormatterNumbers(totals.cacheHits),
    cachedTokens: valueFormatterNumbers(totals.cachedTokens),
  };
};

const CacheAnalyticsTab: React.FC<{ accessToken: string | null }> = ({ accessToken }) => {
  const [selectedApiKeys, setSelectedApiKeys] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [data, setData] = useState<cacheDataItem[]>([]);

  const [dateValue, setDateValue] = useState<DateRangePickerValue>(() => {
    const now = new Date();
    return { from: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000), to: now };
  });

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
  }, [accessToken]);

  const uniqueApiKeys = Array.from(new Set(data.map((item) => item?.api_key ?? "")));
  const uniqueModels = Array.from(new Set(data.map((item) => item?.model ?? "")));

  const updateCachingData = async (startTime: Date | undefined, endTime: Date | undefined) => {
    if (!startTime || !endTime || !accessToken) {
      return;
    }

    const newCacheData = await adminGlobalCacheActivity(
      accessToken,
      formatDateWithoutTZ(startTime),
      formatDateWithoutTZ(endTime),
    );

    setData(newCacheData);
  };

  const { chartData, cacheHitRatio, cachedResponses, cachedTokens } = useMemo(() => {
    const matchesFilters = (item: cacheDataItem) => {
      const keyMatch = selectedApiKeys.length === 0 || selectedApiKeys.includes(item.api_key);
      const modelMatch = selectedModels.length === 0 || selectedModels.includes(item.model);
      return keyMatch && modelMatch;
    };
    return aggregateCacheData(data.filter(matchesFilters));
  }, [data, selectedApiKeys, selectedModels]);

  return (
    <Card>
      <Text className="text-tremor-content dark:text-dark-tremor-content">
        Analytics for LiteLLM&apos;s{" "}
        <a href="https://docs.litellm.ai/docs/proxy/caching" target="_blank" rel="noreferrer" className="underline">
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
        (cached input tokens from Anthropic, OpenAI, etc.) is not shown here; see &quot;Prompt Caching Metrics&quot; on
        the Usage page or individual requests in the Logs page.
      </Text>
      <Grid numItems={3} className="gap-4 mt-4">
        <Col>
          <MultiSelect placeholder="Select Virtual Keys" value={selectedApiKeys} onValueChange={setSelectedApiKeys}>
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
            data={chartData}
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
            data={chartData}
            stack={true}
            index="name"
            valueFormatter={valueFormatterNumbers}
            categories={["Generated Completion Tokens", "Cached Completion Tokens"]}
            colors={["sky", "teal"]}
            yAxisWidth={48}
          />
        </CardContent>
      </ChartCard>
    </Card>
  );
};

export default CacheAnalyticsTab;
