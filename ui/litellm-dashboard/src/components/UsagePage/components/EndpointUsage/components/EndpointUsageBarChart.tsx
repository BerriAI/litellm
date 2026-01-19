import React from "react";
import { BarChart, Card, Title } from "@tremor/react";
import { CustomLegend, CustomTooltip } from "@/components/common_components/chartUtils";
import { MetricWithMetadata } from "../../../types";

interface EndpointUsageBarChartProps {
  endpointData?: Record<string, MetricWithMetadata>;
}

const EndpointUsageBarChart: React.FC<EndpointUsageBarChartProps> = ({ endpointData }) => {
  const dataToUse = endpointData || {};

  // Transform endpoint data into chart format
  const chartData = React.useMemo(() => {
    return Object.entries(dataToUse).map(([endpoint, data]) => ({
      endpoint,
      "metrics.successful_requests": data.metrics.successful_requests,
      "metrics.failed_requests": data.metrics.failed_requests,
      metrics: {
        successful_requests: data.metrics.successful_requests,
        failed_requests: data.metrics.failed_requests,
      },
    }));
  }, [dataToUse]);

  const valueFormatter = (value: number) => value.toLocaleString();

  return (
    <Card>
      <div className="flex justify-between items-center">
        <Title>Success vs Failed Requests by Endpoint</Title>
        <CustomLegend
          categories={["metrics.successful_requests", "metrics.failed_requests"]}
          colors={["green", "red"]}
        />
      </div>
      <BarChart
        className="mt-4"
        data={chartData}
        index="endpoint"
        categories={["metrics.successful_requests", "metrics.failed_requests"]}
        colors={["green", "red"]}
        valueFormatter={valueFormatter}
        customTooltip={CustomTooltip}
        showLegend={false}
        stack={true}
        yAxisWidth={60}
      />
    </Card>
  );
};

export default EndpointUsageBarChart;
