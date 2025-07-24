import React from 'react';
import { Card, Grid, Text, Title, Accordion, AccordionHeader, AccordionBody } from '@tremor/react';
import { AreaChart, BarChart } from '@tremor/react';
import { SpendMetrics, DailyData, ModelActivityData, MetricWithMetadata, KeyMetricWithMetadata, TopApiKeyData } from './usage/types';
import { Collapse } from 'antd';
import { formatNumberWithCommas } from '@/utils/dataUtils';
import type { CustomTooltipProps } from "@tremor/react";
import {
  valueFormatter,
  valueFormatterSpend,
} from "../components/usage/utils/value_formatters";

interface ActivityMetricsProps {
  modelMetrics: Record<string, ModelActivityData>;
}

interface ChartDataPoint {
  date: string;
  metrics: SpendMetrics;
}

const colorNameToHex: { [key: string]: string } = {
  blue: "#3b82f6",
  cyan: "#06b6d4",
  indigo: "#6366f1",
  green: "#22c55e",
  red: "#ef4444",
  purple: "#8b5cf6",
};

export const CustomTooltip = ({
  active,
  payload,
  label,
}: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    const formatCategoryName = (name: string): string => {
      return name
        .replace("metrics.", "")
        .replace(/_/g, " ")
        .split(" ")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
    };

    const getRawValue = (
      dataPoint: ChartDataPoint,
      key: string
    ): number | undefined => {
      // key is like "metrics.total_tokens"
      const metricKey = key.substring(
        key.indexOf(".") + 1
      ) as keyof SpendMetrics;
      if (dataPoint.metrics && metricKey in dataPoint.metrics) {
        return dataPoint.metrics[metricKey];
      }
      return undefined;
    };

    return (
      <div className="w-56 rounded-tremor-default border border-tremor-border bg-tremor-background p-2 text-tremor-default shadow-tremor-dropdown">
        <p className="text-tremor-content-strong">{label}</p>
        {payload.map((item) => {
          const dataKey = item.dataKey?.toString();
          if (!dataKey || !item.payload) return null;

          const rawValue = getRawValue(item.payload, dataKey);
          const isSpend = dataKey.includes("spend");
          const formattedValue =
            rawValue !== undefined
              ? isSpend
                ? `$${rawValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                : rawValue.toLocaleString()
              : "N/A";

          const colorName = item.color as keyof typeof colorNameToHex;
          const hexColor = colorNameToHex[colorName] || item.color;
          return (
            <div
              key={dataKey}
              className="flex items-center justify-between space-x-4"
            >
              <div className="flex items-center space-x-2">
                <span
                  className={`h-2 w-2 shrink-0 rounded-full ring-2 ring-white drop-shadow-md`}
                  style={{ backgroundColor: hexColor }}
                />
                <p className="font-medium text-tremor-content dark:text-dark-tremor-content">
                  {formatCategoryName(dataKey)}
                </p>
              </div>
              <p className="font-medium text-tremor-content-emphasis dark:text-dark-tremor-content-emphasis">
                {formattedValue}
              </p>
            </div>
          );
        })}
      </div>
    );
  }
  return null;
};

const CustomLegend = ({
  categories,
  colors,
}: {
  categories: string[];
  colors: string[];
}) => {
  const formatCategoryName = (name: string): string => {
    return name
      .replace("metrics.", "")
      .replace(/_/g, " ")
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  return (
    <div className="flex items-center justify-end space-x-4">
      {categories.map((category, idx) => {
        const colorName = colors[idx] as keyof typeof colorNameToHex;
        const hexColor = colorNameToHex[colorName] || colors[idx];
        return (
          <div key={category} className="flex items-center space-x-2">
            <span
              className={`h-2 w-2 shrink-0 rounded-full ring-4 ring-white`}
              style={{ backgroundColor: hexColor }}
            />
            <p className="text-sm text-tremor-content dark:text-dark-tremor-content">
              {formatCategoryName(category)}
            </p>
          </div>
        );
      })}
    </div>
  );
};

const ModelSection = ({
  modelName,
  metrics,
}: {
  modelName: string;
  metrics: ModelActivityData;
}) => {
  return (
    <div className="space-y-2">
      {/* Summary Cards */}
      <Grid numItems={4} className="gap-4">
        <Card>
          <Text>Total Requests</Text>
          <Title>{metrics.total_requests.toLocaleString()}</Title>
        </Card>
        <Card>
          <Text>Total Successful Requests</Text>
          <Title>{metrics.total_successful_requests.toLocaleString()}</Title>
        </Card>
        <Card>
          <Text>Total Tokens</Text>
          <Title>{metrics.total_tokens.toLocaleString()}</Title>
          <Text>
            {Math.round(
              metrics.total_tokens / metrics.total_successful_requests
            )}{" "}
            avg per successful request
          </Text>
        </Card>
        <Card>
          <Text>Total Spend</Text>
          <Title>${formatNumberWithCommas(metrics.total_spend, 2)}</Title>
          <Text>
            $
            {formatNumberWithCommas(
              metrics.total_spend / metrics.total_successful_requests,
              3
            )}{" "}
            per successful request
          </Text>
        </Card>
      </Grid>

      {/* Top API Keys Section */}
      {metrics.top_api_keys && metrics.top_api_keys.length > 0 && (
        <Card className="mt-4">
          <Title>Top API Keys by Spend</Title>
          <div className="mt-3">
            <div className="grid grid-cols-1 gap-2">
              {metrics.top_api_keys.map((keyData, index) => (
                <div key={keyData.api_key} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                  <div>
                    <Text className="font-medium">
                      {keyData.key_alias || `${keyData.api_key.substring(0, 10)}...`}
                    </Text>
                    {keyData.team_id && (
                      <Text className="text-xs text-gray-500">Team: {keyData.team_id}</Text>
                    )}
                  </div>
                  <div className="text-right">
                    <Text className="font-medium">${formatNumberWithCommas(keyData.spend, 2)}</Text>
                    <Text className="text-xs text-gray-500">
                      {keyData.requests.toLocaleString()} requests | {keyData.tokens.toLocaleString()} tokens
                    </Text>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      )}

      {/* Charts */}
      <Grid numItems={2} className="gap-4">
        <Card>
          <div className="flex justify-between items-center">
            <Title>Total Tokens</Title>
            <CustomLegend
              categories={[
                "metrics.prompt_tokens",
                "metrics.completion_tokens",
                "metrics.total_tokens",
              ]}
              colors={["blue", "cyan", "indigo"]}
            />
          </div>
          <AreaChart
            className="mt-4"
            data={metrics.daily_data}
            index="date"
            categories={[
              "metrics.prompt_tokens",
              "metrics.completion_tokens",
              "metrics.total_tokens",
            ]}
            colors={["blue", "cyan", "indigo"]}
            valueFormatter={valueFormatter}
            customTooltip={CustomTooltip}
            showLegend={false}
          />
        </Card>

        <Card>
          <div className="flex justify-between items-center">
            <Title>Requests per day</Title>
            <CustomLegend
              categories={["metrics.api_requests"]}
              colors={["blue"]}
            />
          </div>
          <BarChart
            className="mt-4"
            data={metrics.daily_data}
            index="date"
            categories={["metrics.api_requests"]}
            colors={["blue"]}
            valueFormatter={valueFormatter}
            customTooltip={CustomTooltip}
            showLegend={false}
          />
        </Card>

        <Card>
          <div className="flex justify-between items-center">
            <Title>Spend per day</Title>
            <CustomLegend categories={["metrics.spend"]} colors={["green"]} />
          </div>
          <BarChart
            className="mt-4"
            data={metrics.daily_data}
            index="date"
            categories={["metrics.spend"]}
            colors={["green"]}
            valueFormatter={valueFormatterSpend}
            customTooltip={CustomTooltip}
            showLegend={false}
          />
        </Card>

        <Card>
          <div className="flex justify-between items-center">
            <Title>Success vs Failed Requests</Title>
            <CustomLegend
              categories={[
                "metrics.successful_requests",
                "metrics.failed_requests",
              ]}
              colors={["green", "red"]}
            />
          </div>
          <AreaChart
            className="mt-4"
            data={metrics.daily_data}
            index="date"
            categories={[
              "metrics.successful_requests",
              "metrics.failed_requests",
            ]}
            colors={["green", "red"]}
            valueFormatter={valueFormatter}
            stack
            customTooltip={CustomTooltip}
            showLegend={false}
          />
        </Card>

        <Card>
          <div className="flex justify-between items-center">
            <Title>Prompt Caching Metrics</Title>
            <CustomLegend
              categories={[
                "metrics.cache_read_input_tokens",
                "metrics.cache_creation_input_tokens",
              ]}
              colors={["cyan", "purple"]}
            />
          </div>
          <div className="mb-2">
            <Text>
              Cache Read:{" "}
              {metrics.total_cache_read_input_tokens?.toLocaleString() || 0}{" "}
              tokens
            </Text>
            <Text>
              Cache Creation:{" "}
              {metrics.total_cache_creation_input_tokens?.toLocaleString() || 0}{" "}
              tokens
            </Text>
          </div>
          <AreaChart
            className="mt-4"
            data={metrics.daily_data}
            index="date"
            categories={[
              "metrics.cache_read_input_tokens",
              "metrics.cache_creation_input_tokens",
            ]}
            colors={["cyan", "purple"]}
            valueFormatter={valueFormatter}
            customTooltip={CustomTooltip}
            showLegend={false}
          />
        </Card>
      </Grid>
    </div>
  );
};

export const ActivityMetrics: React.FC<ActivityMetricsProps> = ({
  modelMetrics,
}) => {
  const modelNames = Object.keys(modelMetrics).sort((a, b) => {
    if (a === "") return 1;
    if (b === "") return -1;
    return modelMetrics[b].total_spend - modelMetrics[a].total_spend;
  });

  // Calculate total metrics across all models
  const totalMetrics = {
    total_requests: 0,
    total_successful_requests: 0,
    total_tokens: 0,
    total_spend: 0,
    total_cache_read_input_tokens: 0,
    total_cache_creation_input_tokens: 0,
    daily_data: {} as Record<
      string,
      {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
        api_requests: number;
        spend: number;
        successful_requests: number;
        failed_requests: number;
        cache_read_input_tokens: number;
        cache_creation_input_tokens: number;
      }
    >,
  };

  // Aggregate data
  Object.values(modelMetrics).forEach((model) => {
    totalMetrics.total_requests += model.total_requests;
    totalMetrics.total_successful_requests += model.total_successful_requests;
    totalMetrics.total_tokens += model.total_tokens;
    totalMetrics.total_spend += model.total_spend;
    totalMetrics.total_cache_read_input_tokens +=
      model.total_cache_read_input_tokens || 0;
    totalMetrics.total_cache_creation_input_tokens +=
      model.total_cache_creation_input_tokens || 0;

    // Aggregate daily data
    model.daily_data.forEach((day) => {
      if (!totalMetrics.daily_data[day.date]) {
        totalMetrics.daily_data[day.date] = {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
          api_requests: 0,
          spend: 0,
          successful_requests: 0,
          failed_requests: 0,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        };
      }
      totalMetrics.daily_data[day.date].prompt_tokens +=
        day.metrics.prompt_tokens;
      totalMetrics.daily_data[day.date].completion_tokens +=
        day.metrics.completion_tokens;
      totalMetrics.daily_data[day.date].total_tokens +=
        day.metrics.total_tokens;
      totalMetrics.daily_data[day.date].api_requests +=
        day.metrics.api_requests;
      totalMetrics.daily_data[day.date].spend += day.metrics.spend;
      totalMetrics.daily_data[day.date].successful_requests +=
        day.metrics.successful_requests;
      totalMetrics.daily_data[day.date].failed_requests +=
        day.metrics.failed_requests;
      totalMetrics.daily_data[day.date].cache_read_input_tokens +=
        day.metrics.cache_read_input_tokens || 0;
      totalMetrics.daily_data[day.date].cache_creation_input_tokens +=
        day.metrics.cache_creation_input_tokens || 0;
    });
  });

  // Convert daily_data object to array and sort by date
  const sortedDailyData = Object.entries(totalMetrics.daily_data)
    .map(([date, metrics]) => ({ date, metrics }))
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  return (
    <div className="space-y-8">
      {/* Global Summary */}
      <div className="border rounded-lg p-4">
        <Title>Overall Usage</Title>
        <Grid numItems={4} className="gap-4 mb-4">
          <Card>
            <Text>Total Requests</Text>
            <Title>{totalMetrics.total_requests.toLocaleString()}</Title>
          </Card>
          <Card>
            <Text>Total Successful Requests</Text>
            <Title>
              {totalMetrics.total_successful_requests.toLocaleString()}
            </Title>
          </Card>
          <Card>
            <Text>Total Tokens</Text>
            <Title>{totalMetrics.total_tokens.toLocaleString()}</Title>
          </Card>
          <Card>
            <Text>Total Spend</Text>
            <Title>
              ${formatNumberWithCommas(totalMetrics.total_spend, 2)}
            </Title>
          </Card>
        </Grid>

        <Grid numItems={2} className="gap-4">
          <Card>
            <div className="flex justify-between items-center">
              <Title>Total Tokens Over Time</Title>
              <CustomLegend
                categories={[
                  "metrics.prompt_tokens",
                  "metrics.completion_tokens",
                  "metrics.total_tokens",
                ]}
                colors={["blue", "cyan", "indigo"]}
              />
            </div>
            <AreaChart
              className="mt-4"
              data={sortedDailyData}
              index="date"
              categories={[
                "metrics.prompt_tokens",
                "metrics.completion_tokens",
                "metrics.total_tokens",
              ]}
              colors={["blue", "cyan", "indigo"]}
              valueFormatter={valueFormatter}
              customTooltip={CustomTooltip}
              showLegend={false}
            />
          </Card>
          <Card>
            <div className="flex justify-between items-center">
              <Title>Total Requests Over Time</Title>
              <CustomLegend
                categories={[
                  "metrics.successful_requests",
                  "metrics.failed_requests",
                ]}
                colors={["green", "red"]}
              />
            </div>
            <AreaChart
              className="mt-4"
              data={sortedDailyData}
              index="date"
              categories={[
                "metrics.successful_requests",
                "metrics.failed_requests",
              ]}
              colors={["green", "red"]}
              valueFormatter={valueFormatter}
              stack
              customTooltip={CustomTooltip}
              showLegend={false}
            />
          </Card>
        </Grid>
      </div>

      {/* Individual Model Sections */}
      <Collapse defaultActiveKey={modelNames[0]}>
        {modelNames.map((modelName) => (
          <Collapse.Panel
            key={modelName}
            header={
              <div className="flex justify-between items-center w-full">
                <Title>{modelMetrics[modelName].label || "Unknown Item"}</Title>
                <div className="flex space-x-4 text-sm text-gray-500">
                  <span>
                    $
                    {formatNumberWithCommas(
                      modelMetrics[modelName].total_spend,
                      2
                    )}
                  </span>
                  <span>
                    {modelMetrics[modelName].total_requests.toLocaleString()}{" "}
                    requests
                  </span>
                </div>
              </div>
            }
          >
            <ModelSection
              modelName={modelName || "Unknown Model"}
              metrics={modelMetrics[modelName]}
            />
          </Collapse.Panel>
        ))}
      </Collapse>
    </div>
  );
};

// Helper function to format key label
const formatKeyLabel = (
  modelData: KeyMetricWithMetadata,
  model: string
): string => {
  const keyAlias = modelData.metadata.key_alias || `key-hash-${model}`;
  const teamId = modelData.metadata.team_id;
  return teamId ? `${keyAlias} (team_id: ${teamId})` : keyAlias;
};

// Process data function
export const processActivityData = (dailyActivity: { results: DailyData[] }, key: "models" | "api_keys" | "mcp_servers"): Record<string, ModelActivityData> => {
  const modelMetrics: Record<string, ModelActivityData> = {};

  dailyActivity.results.forEach((day) => {
    Object.entries(day.breakdown[key] || {}).forEach(([model, modelData]) => {
      if (!modelMetrics[model]) {
        modelMetrics[model] = {
          label:
            key === "api_keys"
              ? formatKeyLabel(modelData as KeyMetricWithMetadata, model)
              : model,
          total_requests: 0,
          total_successful_requests: 0,
          total_failed_requests: 0,
          total_tokens: 0,
          prompt_tokens: 0,
          completion_tokens: 0,
          total_spend: 0,
          total_cache_read_input_tokens: 0,
          total_cache_creation_input_tokens: 0,
          top_api_keys: [],
          daily_data: []
        };
      }

      // Update totals
      modelMetrics[model].total_requests += modelData.metrics.api_requests;
      modelMetrics[model].prompt_tokens += modelData.metrics.prompt_tokens;
      modelMetrics[model].completion_tokens +=
        modelData.metrics.completion_tokens;
      modelMetrics[model].total_tokens += modelData.metrics.total_tokens;
      modelMetrics[model].total_spend += modelData.metrics.spend;
      modelMetrics[model].total_successful_requests +=
        modelData.metrics.successful_requests;
      modelMetrics[model].total_failed_requests +=
        modelData.metrics.failed_requests;
      modelMetrics[model].total_cache_read_input_tokens +=
        modelData.metrics.cache_read_input_tokens || 0;
      modelMetrics[model].total_cache_creation_input_tokens +=
        modelData.metrics.cache_creation_input_tokens || 0;

      // Add daily data
      modelMetrics[model].daily_data.push({
        date: day.date,
        metrics: {
          prompt_tokens: modelData.metrics.prompt_tokens,
          completion_tokens: modelData.metrics.completion_tokens,
          total_tokens: modelData.metrics.total_tokens,
          api_requests: modelData.metrics.api_requests,
          spend: modelData.metrics.spend,
          successful_requests: modelData.metrics.successful_requests,
          failed_requests: modelData.metrics.failed_requests,
          cache_read_input_tokens:
            modelData.metrics.cache_read_input_tokens || 0,
          cache_creation_input_tokens:
            modelData.metrics.cache_creation_input_tokens || 0,
        },
      });
    });
  });

  // Process API key breakdowns for each metric (skip if key is 'api_keys' to avoid duplication)
  if (key !== 'api_keys') {
    Object.entries(modelMetrics).forEach(([model, _]) => {
      const apiKeyBreakdown: Record<string, TopApiKeyData> = {};
      
      // Aggregate API key data across all days
      dailyActivity.results.forEach((day) => {
        const modelData = day.breakdown[key]?.[model];
        if (modelData && 'api_key_breakdown' in modelData) {
          Object.entries(modelData.api_key_breakdown || {}).forEach(([apiKey, keyData]) => {
            if (!apiKeyBreakdown[apiKey]) {
              apiKeyBreakdown[apiKey] = {
                api_key: apiKey,
                key_alias: keyData.metadata.key_alias,
                team_id: keyData.metadata.team_id,
                spend: 0,
                requests: 0,
                tokens: 0,
              };
            }
            
            apiKeyBreakdown[apiKey].spend += keyData.metrics.spend;
            apiKeyBreakdown[apiKey].requests += keyData.metrics.api_requests;
            apiKeyBreakdown[apiKey].tokens += keyData.metrics.total_tokens;
          });
        }
      });

      // Sort by spend and take top 5
      modelMetrics[model].top_api_keys = Object.values(apiKeyBreakdown)
        .sort((a, b) => b.spend - a.spend)
        .slice(0, 5);
    });
  }

  // Sort daily data
  Object.values(modelMetrics).forEach((metrics) => {
    metrics.daily_data.sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    );
  });

  return modelMetrics;
};
