import React from "react";

import { filterModelActivity } from "../modelActivityFilter";
import type { ModelActivityData } from "../types";
import ActivitySearchPanel from "./ActivitySearchPanel";

interface ModelActivityPanelProps {
  modelMetrics: Record<string, ModelActivityData>;
  hidePromptCachingMetrics?: boolean;
}

const ModelActivityPanel: React.FC<ModelActivityPanelProps> = ({ modelMetrics, hidePromptCachingMetrics = false }) => {
  return (
    <ActivitySearchPanel
      metrics={modelMetrics}
      filter={filterModelActivity}
      noun="models"
      placeholder="Search by model name"
      searchLabel="Search models"
      clearLabel="Clear model search"
      hidePromptCachingMetrics={hidePromptCachingMetrics}
    />
  );
};

export default ModelActivityPanel;
