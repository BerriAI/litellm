import React, { useMemo } from "react";
import EndpointUsageBarChart from "./components/EndpointUsageBarChart";
import EndpointUsageLineChart from "./components/EndpointUsageLineChart";
import EndpointUsageTable from "./components/EndpointUsageTable";
import { DailyData, MetricWithMetadata } from "../../types";

interface EndpointUsageProps {
  userSpendData?: {
    results: DailyData[];
    metadata: any;
  };
}

const EndpointUsage: React.FC<EndpointUsageProps> = ({ userSpendData }) => {
  // Aggregate endpoints data from all days
  const endpointData = useMemo(() => {
    const aggregatedEndpoints: Record<string, MetricWithMetadata> = {};

    if (!userSpendData?.results) {
      return aggregatedEndpoints;
    }

    userSpendData.results.forEach((day) => {
      Object.entries(day.breakdown.endpoints || {}).forEach(([endpoint, metrics]) => {
        if (!aggregatedEndpoints[endpoint]) {
          aggregatedEndpoints[endpoint] = {
            metrics: {
              spend: 0,
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0,
              api_requests: 0,
              successful_requests: 0,
              failed_requests: 0,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
            },
            metadata: metrics.metadata || {},
            api_key_breakdown: {},
          };
        }
        aggregatedEndpoints[endpoint].metrics.spend += metrics.metrics.spend;
        aggregatedEndpoints[endpoint].metrics.prompt_tokens += metrics.metrics.prompt_tokens;
        aggregatedEndpoints[endpoint].metrics.completion_tokens += metrics.metrics.completion_tokens;
        aggregatedEndpoints[endpoint].metrics.total_tokens += metrics.metrics.total_tokens;
        aggregatedEndpoints[endpoint].metrics.api_requests += metrics.metrics.api_requests;
        aggregatedEndpoints[endpoint].metrics.successful_requests += metrics.metrics.successful_requests || 0;
        aggregatedEndpoints[endpoint].metrics.failed_requests += metrics.metrics.failed_requests || 0;
        aggregatedEndpoints[endpoint].metrics.cache_read_input_tokens += metrics.metrics.cache_read_input_tokens || 0;
        aggregatedEndpoints[endpoint].metrics.cache_creation_input_tokens +=
          metrics.metrics.cache_creation_input_tokens || 0;
      });
    });

    return aggregatedEndpoints;
  }, [userSpendData]);

  return (
    <div className="space-y-4">
      <EndpointUsageTable endpointData={endpointData} />
      <EndpointUsageBarChart endpointData={endpointData} />
      <EndpointUsageLineChart dailyData={userSpendData} endpointData={endpointData} />
    </div>
  );
};

export default EndpointUsage;
