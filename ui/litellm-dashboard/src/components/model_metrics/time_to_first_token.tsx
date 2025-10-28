import React from "react";
import { LineChart } from "@tremor/react";
interface TimeToFirstTokenProps {
  modelMetrics: any[];
  modelMetricsCategories: string[];
  customTooltip: any;
  premiumUser: boolean;
}

const TimeToFirstToken: React.FC<TimeToFirstTokenProps> = ({
  modelMetrics,
  modelMetricsCategories,
  customTooltip,
  premiumUser,
}) => {
  return (
    <LineChart
      title="Time to First token (s)"
      className="h-72"
      data={modelMetrics}
      index="date"
      showLegend={false}
      categories={modelMetricsCategories}
      colors={["indigo", "rose"]}
      connectNulls={true}
      customTooltip={customTooltip}
    />
  );
};

export default TimeToFirstToken;
