import React from "react";
import { LineChart, Callout, Button } from "@tremor/react";
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
  return premiumUser ? (
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
  ) : (
    <div>
      <Callout title="✨ Enterprise Feature" color="teal" className="mt-2 mb-4">
        Enterprise features are available for users with a specific license,
        please contact LiteLLM to unlock this limitation.
      </Callout>
      <Button variant="primary">
        <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
          Get in touch
        </a>
      </Button>
    </div>
  );
};

export default TimeToFirstToken;
