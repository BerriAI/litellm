import React from "react";

import { filterKeyActivity } from "../keyActivityFilter";
import type { ModelActivityData } from "../types";
import ActivitySearchPanel from "./ActivitySearchPanel";

interface KeyActivityPanelProps {
  keyMetrics: Record<string, ModelActivityData>;
  hidePromptCachingMetrics?: boolean;
}

const KeyActivityPanel: React.FC<KeyActivityPanelProps> = ({ keyMetrics, hidePromptCachingMetrics = false }) => {
  return (
    <ActivitySearchPanel
      metrics={keyMetrics}
      filter={filterKeyActivity}
      noun="keys"
      placeholder="Search by key alias, key hash, user ID, or email"
      searchLabel="Search keys"
      clearLabel="Clear key search"
      hidePromptCachingMetrics={hidePromptCachingMetrics}
    />
  );
};

export default KeyActivityPanel;
