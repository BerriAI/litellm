import { Search, X } from "lucide-react";
import React, { useMemo, useState } from "react";

import { ActivityMetrics } from "@/components/activity_metrics";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";

import type { ModelActivityData } from "../types";

interface ActivitySearchPanelProps {
  metrics: Record<string, ModelActivityData>;
  filter: (metrics: Record<string, ModelActivityData>, query: string) => Record<string, ModelActivityData>;
  noun: string;
  placeholder: string;
  searchLabel: string;
  clearLabel: string;
  hidePromptCachingMetrics?: boolean;
}

const ActivitySearchPanel: React.FC<ActivitySearchPanelProps> = ({
  metrics,
  filter,
  noun,
  placeholder,
  searchLabel,
  clearLabel,
  hidePromptCachingMetrics = false,
}) => {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => filter(metrics, query), [metrics, filter, query]);
  const totalKeys = Object.keys(metrics).length;
  const shownKeys = Object.keys(filtered).length;
  const isFiltering = query.trim() !== "";

  return (
    <div className="space-y-4">
      <div className="mt-2 flex items-center gap-3">
        <InputGroup className="max-w-md">
          <InputGroupAddon>
            <Search className="size-4 text-muted-foreground" />
          </InputGroupAddon>
          <InputGroupInput
            aria-label={searchLabel}
            placeholder={placeholder}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {isFiltering && (
            <InputGroupAddon align="inline-end">
              <InputGroupButton size="icon-xs" aria-label={clearLabel} onClick={() => setQuery("")}>
                <X />
              </InputGroupButton>
            </InputGroupAddon>
          )}
        </InputGroup>
        <span className="text-sm text-muted-foreground">
          Showing {shownKeys.toLocaleString()} of {totalKeys.toLocaleString()} {noun}
        </span>
      </div>
      {isFiltering && totalKeys > 0 && shownKeys === 0 ? (
        <p className="rounded-lg border p-6 text-center text-sm text-muted-foreground">
          No {noun} match &quot;{query.trim()}&quot; in this date range
        </p>
      ) : (
        <ActivityMetrics modelMetrics={filtered} hidePromptCachingMetrics={hidePromptCachingMetrics} />
      )}
    </div>
  );
};

export default ActivitySearchPanel;
