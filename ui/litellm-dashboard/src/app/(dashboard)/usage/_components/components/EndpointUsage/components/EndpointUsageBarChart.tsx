import React from "react";
import { ChartColumnStacked } from "lucide-react";
import { BarChart, CustomTooltip } from "@/components/shared/charts";
import { MetricWithMetadata } from "@/components/UsagePage/types";
import { Panel } from "../../overview/Primitives";

interface EndpointUsageBarChartProps {
  endpointData?: Record<string, MetricWithMetadata>;
}

const SUCCESS_COLOR = "#2b3fd6";
const FAILED_COLOR = "#ef4444";
const CATEGORIES = ["Successful", "Failed"] as const;
const COLORS = [SUCCESS_COLOR, FAILED_COLOR] as const;

const valueFormatter = (value: number) => value.toLocaleString();

const EndpointUsageBarChart: React.FC<EndpointUsageBarChartProps> = ({ endpointData }) => {
  // Transform endpoint data into chart format
  const chartData = React.useMemo(() => {
    return Object.entries(endpointData || {}).map(([endpoint, data]) => ({
      endpoint,
      Successful: data.metrics.successful_requests,
      Failed: data.metrics.failed_requests,
    }));
  }, [endpointData]);

  const totals = React.useMemo(
    () =>
      chartData.reduce(
        (acc, row) => ({ Successful: acc.Successful + row.Successful, Failed: acc.Failed + row.Failed }),
        { Successful: 0, Failed: 0 },
      ),
    [chartData],
  );

  return (
    <Panel
      icon={ChartColumnStacked}
      title="Success vs Failed Requests by Endpoint"
      action={
        <div className="flex shrink-0 items-center gap-3 text-xs text-muted-foreground">
          {CATEGORIES.map((category, i) => (
            <span key={category} className="flex items-center gap-1.5">
              <span aria-hidden="true" className="size-2 rounded-[2px]" style={{ backgroundColor: COLORS[i] }} />
              <span>{category}</span>
              <span className="tabular-nums text-foreground">{totals[category].toLocaleString()}</span>
            </span>
          ))}
        </div>
      }
    >
      <BarChart
        className="h-64"
        data={chartData}
        index="endpoint"
        categories={CATEGORIES}
        colors={COLORS}
        valueFormatter={valueFormatter}
        customTooltip={CustomTooltip}
        showLegend={false}
        stack={true}
        maxBarSize={48}
        yAxisWidth={56}
      />
    </Panel>
  );
};

export default EndpointUsageBarChart;
