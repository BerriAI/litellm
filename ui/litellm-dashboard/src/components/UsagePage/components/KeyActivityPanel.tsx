import { Search, X } from "lucide-react";
import { parseAsString, throttle, useQueryState } from "nuqs";
import React, { useMemo } from "react";

import { ActivityMetrics } from "@/components/activity_metrics";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";

import { filterKeyActivity } from "../keyActivityFilter";
import type { ModelActivityData } from "../types";

const keySearchParser = parseAsString.withDefault("").withOptions({ limitUrlUpdates: throttle(300) });

interface KeyActivityPanelProps {
  keyMetrics: Record<string, ModelActivityData>;
  hidePromptCachingMetrics?: boolean;
}

const KeyActivityPanel: React.FC<KeyActivityPanelProps> = ({ keyMetrics, hidePromptCachingMetrics = false }) => {
  const [query, setQuery] = useQueryState("key_search", keySearchParser);
  const filtered = useMemo(() => filterKeyActivity(keyMetrics, query), [keyMetrics, query]);
  const totalKeys = Object.keys(keyMetrics).length;
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
            aria-label="Search keys"
            placeholder="Search by key alias, key hash, user ID, or email"
            value={query}
            onChange={(e) => void setQuery(e.target.value)}
          />
          {isFiltering && (
            <InputGroupAddon align="inline-end">
              <InputGroupButton size="icon-xs" aria-label="Clear key search" onClick={() => void setQuery(null)}>
                <X />
              </InputGroupButton>
            </InputGroupAddon>
          )}
        </InputGroup>
        <span className="text-sm text-muted-foreground">
          Showing {shownKeys.toLocaleString()} of {totalKeys.toLocaleString()} keys
        </span>
      </div>
      {isFiltering && totalKeys > 0 && shownKeys === 0 ? (
        <p className="rounded-lg border p-6 text-center text-sm text-muted-foreground">
          No keys match &quot;{query.trim()}&quot; in this date range
        </p>
      ) : (
        <ActivityMetrics modelMetrics={filtered} hidePromptCachingMetrics={hidePromptCachingMetrics} />
      )}
    </div>
  );
};

export default KeyActivityPanel;
