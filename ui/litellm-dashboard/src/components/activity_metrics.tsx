import React from 'react';
import { Card, Grid, Text, Title, Accordion, AccordionHeader, AccordionBody } from '@tremor/react';
import { AreaChart, BarChart } from '@tremor/react';
import { SpendMetrics, DailyData, ModelActivityData, MetricWithMetadata, KeyMetricWithMetadata, TopApiKeyData } from './usage/types';
import { Collapse } from 'antd';
import { formatNumberWithCommas } from '@/utils/dataUtils';

// Configuration for component visibility
const COMPONENT_CONFIG = {
  models: {
    summary: ['total_requests', 'total_successful_requests', 'total_tokens', 'total_spend'],
    charts: ['tokens_chart', 'requests_chart', 'spend_chart', 'success_failed_chart', 'prompt_caching_chart'],
    sections: ['top_api_keys']
  },
  api_keys: {
    summary: ['total_requests', 'total_successful_requests', 'total_tokens', 'total_spend'],
    charts: ['tokens_chart', 'requests_chart', 'spend_chart', 'success_failed_chart', 'prompt_caching_chart'],
    sections: ['top_api_keys']
  },
  mcp_servers: {
    summary: ['total_requests', 'total_successful_requests', 'total_spend'],
    charts: ['requests_chart', 'spend_chart', 'success_failed_chart'],
    sections: ['top_api_keys']
  }
};

interface ActivityMetricsProps {
  modelMetrics: Record<string, ModelActivityData>;
  dataType?: "models" | "api_keys" | "mcp_servers";
}

const isComponentVisible = (dataType: string | undefined, category: keyof typeof COMPONENT_CONFIG.models, component: string): boolean => {
  if (!dataType) return true;
  const config = COMPONENT_CONFIG[dataType as keyof typeof COMPONENT_CONFIG];
  return config?.[category]?.includes(component) ?? true;
};

const getVisibleSummaryCards = (dataType: string | undefined) => {
  if (!dataType) return COMPONENT_CONFIG.models.summary;
  return COMPONENT_CONFIG[dataType as keyof typeof COMPONENT_CONFIG]?.summary ?? COMPONENT_CONFIG.models.summary;
};

const getVisibleCharts = (dataType: string | undefined) => {
  if (!dataType) return COMPONENT_CONFIG.models.charts;
  return COMPONENT_CONFIG[dataType as keyof typeof COMPONENT_CONFIG]?.charts ?? COMPONENT_CONFIG.models.charts;
};

const ModelSection = ({ modelName, metrics, dataType }: { 
  modelName: string; 
  metrics: ModelActivityData;
  dataType?: "models" | "api_keys" | "mcp_servers";
}) => {
  const visibleSummaryCards = getVisibleSummaryCards(dataType);
  const visibleCharts = getVisibleCharts(dataType);

  const summaryCards = [
    {
      key: 'total_requests',
      title: 'Total Requests',
      value: metrics.total_requests.toLocaleString()
    },
    {
      key: 'total_successful_requests', 
      title: 'Total Successful Requests',
      value: metrics.total_successful_requests.toLocaleString()
    },
    {
      key: 'total_tokens',
      title: 'Total Tokens',
      value: metrics.total_tokens.toLocaleString(),
      subtitle: `${Math.round(metrics.total_tokens / metrics.total_successful_requests)} avg per successful request`
    },
    {
      key: 'total_spend',
      title: 'Total Spend', 
      value: `$${formatNumberWithCommas(metrics.total_spend, 2)}`,
      subtitle: `$${formatNumberWithCommas((metrics.total_spend / metrics.total_successful_requests), 3)} per successful request`
    }
  ];

  const chartComponents = [
    {
      key: 'tokens_chart',
      title: 'Total Tokens',
      component: (
        <AreaChart    
          data={metrics.daily_data}
          index="date"
          categories={["metrics.prompt_tokens", "metrics.completion_tokens", "metrics.total_tokens"]}
          colors={["blue", "cyan", "indigo"]}
          valueFormatter={(number: number) => number.toLocaleString()}
        />
      )
    },
    {
      key: 'requests_chart',
      title: 'Requests per day',
      component: (
        <BarChart
          data={metrics.daily_data}
          index="date"
          categories={["metrics.api_requests"]}
          colors={["blue"]}
          valueFormatter={(number: number) => number.toLocaleString()}
        />
      )
    },
    {
      key: 'spend_chart',
      title: 'Spend per day',
      component: (
        <BarChart
          data={metrics.daily_data}
          index="date"
          categories={["metrics.spend"]}
          colors={["green"]}
          valueFormatter={(value: number) => `$${formatNumberWithCommas(value, 2)}`}
        />
      )
    },
    {
      key: 'success_failed_chart',
      title: 'Success vs Failed Requests',
      component: (
        <AreaChart
          data={metrics.daily_data}
          index="date"
          categories={["metrics.successful_requests", "metrics.failed_requests"]}
          colors={["emerald", "red"]}
          valueFormatter={(number: number) => number.toLocaleString()}
          stack
        />
      )
    },
    {
      key: 'prompt_caching_chart',
      title: 'Prompt Caching Metrics',
      component: (
        <div>
          <div className="mb-2">
            <Text>Cache Read: {metrics.total_cache_read_input_tokens?.toLocaleString() || 0} tokens</Text>
            <Text>Cache Creation: {metrics.total_cache_creation_input_tokens?.toLocaleString() || 0} tokens</Text>
          </div>
          <AreaChart
            data={metrics.daily_data}
            index="date"
            categories={["metrics.cache_read_input_tokens", "metrics.cache_creation_input_tokens"]}
            colors={["cyan", "purple"]}
            valueFormatter={(number: number) => number.toLocaleString()}
          />
        </div>
      )
    }
  ];

  const filteredSummaryCards = summaryCards.filter(card => visibleSummaryCards.includes(card.key));
  const filteredCharts = chartComponents.filter(chart => visibleCharts.includes(chart.key));

  return (
    <div className="space-y-2">
      {/* Summary Cards */}
      {filteredSummaryCards.length > 0 && (
        <Grid numItems={filteredSummaryCards.length} className="gap-4">
          {filteredSummaryCards.map(card => (
            <Card key={card.key}>
              <Text>{card.title}</Text>
              <Title>{card.value}</Title>
              {card.subtitle && <Text>{card.subtitle}</Text>}
            </Card>
          ))}
        </Grid>
      )}

      {/* Top API Keys Section */}
      {isComponentVisible(dataType, 'sections', 'top_api_keys') && 
       metrics.top_api_keys && metrics.top_api_keys.length > 0 && (
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
      {filteredCharts.length > 0 && (
        <Grid numItems={2} className="gap-4">
          {filteredCharts.map(chart => (
            <Card key={chart.key}>
              <Title>{chart.title}</Title>
              {chart.component}
            </Card>
          ))}
        </Grid>
      )}
    </div>
  );
};

export const ActivityMetrics: React.FC<ActivityMetricsProps> = ({ modelMetrics, dataType = 'models' }) => {
  const modelNames = Object.keys(modelMetrics).sort((a, b) => {
    if (a === '') return 1;
    if (b === '') return -1;
    return modelMetrics[b].total_spend - modelMetrics[a].total_spend;
  });

  const visibleSummaryCards = getVisibleSummaryCards(dataType);
  const visibleCharts = getVisibleCharts(dataType);

  // Calculate total metrics across all models
  const totalMetrics = {
    total_requests: 0,
    total_successful_requests: 0,
    total_tokens: 0,
    total_spend: 0,
    total_cache_read_input_tokens: 0,
    total_cache_creation_input_tokens: 0,
    daily_data: {} as Record<string, {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
      api_requests: number;
      spend: number;
      successful_requests: number;
      failed_requests: number;
      cache_read_input_tokens: number;
      cache_creation_input_tokens: number;
    }>
  };

  // Aggregate data
  Object.values(modelMetrics).forEach(model => {
    totalMetrics.total_requests += model.total_requests;
    totalMetrics.total_successful_requests += model.total_successful_requests;
    totalMetrics.total_tokens += model.total_tokens;
    totalMetrics.total_spend += model.total_spend;
    totalMetrics.total_cache_read_input_tokens += model.total_cache_read_input_tokens || 0;
    totalMetrics.total_cache_creation_input_tokens += model.total_cache_creation_input_tokens || 0;

    // Aggregate daily data
    model.daily_data.forEach(day => {
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
          cache_creation_input_tokens: 0
        };
      }
      totalMetrics.daily_data[day.date].prompt_tokens += day.metrics.prompt_tokens;
      totalMetrics.daily_data[day.date].completion_tokens += day.metrics.completion_tokens;
      totalMetrics.daily_data[day.date].total_tokens += day.metrics.total_tokens;
      totalMetrics.daily_data[day.date].api_requests += day.metrics.api_requests;
      totalMetrics.daily_data[day.date].spend += day.metrics.spend;
      totalMetrics.daily_data[day.date].successful_requests += day.metrics.successful_requests;
      totalMetrics.daily_data[day.date].failed_requests += day.metrics.failed_requests;
      totalMetrics.daily_data[day.date].cache_read_input_tokens += day.metrics.cache_read_input_tokens || 0;
      totalMetrics.daily_data[day.date].cache_creation_input_tokens += day.metrics.cache_creation_input_tokens || 0;
    });
  });

  // Convert daily_data object to array and sort by date
  const sortedDailyData = Object.entries(totalMetrics.daily_data)
    .map(([date, metrics]) => ({ date, metrics }))
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  // Define summary cards for overall section
  const overallSummaryCards = [
    {
      key: 'total_requests',
      title: 'Total Requests',
      value: totalMetrics.total_requests.toLocaleString()
    },
    {
      key: 'total_successful_requests',
      title: 'Total Successful Requests', 
      value: totalMetrics.total_successful_requests.toLocaleString()
    },
    {
      key: 'total_tokens',
      title: 'Total Tokens',
      value: totalMetrics.total_tokens.toLocaleString()
    },
    {
      key: 'total_spend',
      title: 'Total Spend',
      value: `$${formatNumberWithCommas(totalMetrics.total_spend, 2)}`
    }
  ];

  const overallChartComponents = [
    {
      key: 'tokens_chart',
      title: 'Total Tokens Over Time',
      component: (
        <AreaChart    
          data={sortedDailyData}
          index="date"
          categories={["metrics.prompt_tokens", "metrics.completion_tokens", "metrics.total_tokens"]}
          colors={["blue", "cyan", "indigo"]}
          valueFormatter={(number: number) => number.toLocaleString()}
        />
      )
    },
    {
      key: 'requests_chart',
      title: 'Total Requests Over Time',
      component: (
        <AreaChart
          data={sortedDailyData}
          index="date"
          categories={["metrics.successful_requests", "metrics.failed_requests"]}
          colors={["emerald", "red"]}
          valueFormatter={(number: number) => number.toLocaleString()}
          stack
        />
      )
    }
  ];

  const filteredOverallSummary = overallSummaryCards.filter(card => visibleSummaryCards.includes(card.key));
  const filteredOverallCharts = overallChartComponents.filter(chart => visibleCharts.includes(chart.key));

  return (
    <div className="space-y-8">
      {/* Global Summary */}
      <div className="border rounded-lg p-4">
        <Title>Overall Usage</Title>
        
        {filteredOverallSummary.length > 0 && (
          <Grid numItems={filteredOverallSummary.length} className="gap-4 mb-4">
            {filteredOverallSummary.map(card => (
              <Card key={card.key}>
                <Text>{card.title}</Text>
                <Title>{card.value}</Title>
              </Card>
            ))}
          </Grid>
        )}

        {filteredOverallCharts.length > 0 && (
          <Grid numItems={filteredOverallCharts.length} className="gap-4">
            {filteredOverallCharts.map(chart => (
              <Card key={chart.key}>
                <Title>{chart.title}</Title>
                {chart.component}
              </Card>
            ))}
          </Grid>
        )}
      </div>

      {/* Individual Model Sections */}
      <Collapse defaultActiveKey={modelNames[0]}>
        {modelNames.map((modelName) => (
          <Collapse.Panel 
            key={modelName} 
            header={
              <div className="flex justify-between items-center w-full">
                <Title>{modelMetrics[modelName].label || 'Unknown Item'}</Title>
                <div className="flex space-x-4 text-sm text-gray-500">
                  <span>${formatNumberWithCommas(modelMetrics[modelName].total_spend, 2)}</span>
                  <span>{modelMetrics[modelName].total_requests.toLocaleString()} requests</span>
                </div>
              </div>
            }
          >
            <ModelSection 
              modelName={modelName || 'Unknown Model'} 
              metrics={modelMetrics[modelName]}
              dataType={dataType}
            />
          </Collapse.Panel>
        ))}
      </Collapse>
    </div>
  );
};

// Helper function to format key label
const formatKeyLabel = (modelData: KeyMetricWithMetadata, model: string): string => {
  const keyAlias = modelData.metadata.key_alias || `key-hash-${model}`;
  const teamId = modelData.metadata.team_id;
  return teamId ? `${keyAlias} (team_id: ${teamId})` : keyAlias;
};

// Process data function - simplified since component visibility is now handled by configuration
export const processActivityData = (
  dailyActivity: { results: DailyData[] }, 
  key: "models" | "api_keys" | "mcp_servers",
  visibleComponents?: string[]
): Record<string, ModelActivityData> => {
  const modelMetrics: Record<string, ModelActivityData> = {};

  dailyActivity.results.forEach((day) => {
    Object.entries(day.breakdown[key] || {}).forEach(([model, modelData]) => {
      if (!modelMetrics[model]) {
        modelMetrics[model] = {
          label: key === 'api_keys' 
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
          visibleComponents,
          daily_data: []
        };
      }

      // Update totals
      modelMetrics[model].total_requests += modelData.metrics.api_requests;
      modelMetrics[model].prompt_tokens += modelData.metrics.prompt_tokens;
      modelMetrics[model].completion_tokens += modelData.metrics.completion_tokens;
      modelMetrics[model].total_tokens += modelData.metrics.total_tokens;
      modelMetrics[model].total_spend += modelData.metrics.spend;
      modelMetrics[model].total_successful_requests += modelData.metrics.successful_requests;
      modelMetrics[model].total_failed_requests += modelData.metrics.failed_requests;
      modelMetrics[model].total_cache_read_input_tokens += modelData.metrics.cache_read_input_tokens || 0;
      modelMetrics[model].total_cache_creation_input_tokens += modelData.metrics.cache_creation_input_tokens || 0;

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
          cache_read_input_tokens: modelData.metrics.cache_read_input_tokens || 0,
          cache_creation_input_tokens: modelData.metrics.cache_creation_input_tokens || 0
        }
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
  Object.values(modelMetrics).forEach(metrics => {
    metrics.daily_data.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  });

  return modelMetrics;
}; 