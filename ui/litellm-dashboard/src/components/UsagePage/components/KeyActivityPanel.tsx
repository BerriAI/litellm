import { Search, X } from "lucide-react";
import React, { useMemo, useState } from "react";

import { ActivityMetrics } from "@/components/activity_metrics";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";

import { filterKeyActivity, parseKeyQuery } from "../keyActivityFilter";
import type { ModelActivityData } from "../types";

interface KeyActivityPanelProps {
  keyMetrics: Record<string, ModelActivityData>;
  hidePromptCachingMetrics?: boolean;
}

const KeyActivityPanel: React.FC<KeyActivityPanelProps> = ({ keyMetrics, hidePromptCachingMetrics = false }) => {
  const [query, setQuery] = useState("");
  const parsed = useMemo(() => parseKeyQuery(query), [query]);
  const filtered = useMemo(() => filterKeyActivity(keyMetrics, query), [keyMetrics, query]);
  const totalKeys = Object.keys(keyMetrics).length;
  const shownKeys = Object.keys(filtered).length;
  const isFiltering = parsed.kind !== "all";
  const hasNoMatches = isFiltering && totalKeys > 0 && shownKeys === 0;

  return (
    <div className="space-y-4">
      <div className="mt-2 flex items-center gap-3">
        <InputGroup className="max-w-md">
          <InputGroupAddon>
            <Search className="size-4 text-muted-foreground" />
          </InputGroupAddon>
          <InputGroupInput
            aria-label="Search keys"
            placeholder="Search by key alias, hash, user ID, or email. Supports * and /regex/"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {isFiltering && (
            <InputGroupAddon align="inline-end">
              <InputGroupButton size="icon-xs" aria-label="Clear key search" onClick={() => setQuery("")}>
                <X />
              </InputGroupButton>
            </InputGroupAddon>
          )}
        </InputGroup>
        <span className="text-sm text-muted-foreground">
          Showing {shownKeys.toLocaleString()} of {totalKeys.toLocaleString()} keys
        </span>
      </div>
      {parsed.kind === "invalid" && (
        <p className="rounded-lg border p-6 text-center text-sm text-muted-foreground">
          Invalid regular expression: {parsed.source}
        </p>
      )}
      {parsed.kind !== "invalid" && hasNoMatches && (
        <p className="rounded-lg border p-6 text-center text-sm text-muted-foreground">
          No keys match &quot;{query.trim()}&quot; in this date range
        </p>
      )}
      {parsed.kind !== "invalid" && !hasNoMatches && (
        <ActivityMetrics modelMetrics={filtered} hidePromptCachingMetrics={hidePromptCachingMetrics} />
      )}
    </div>
  );
};

export default KeyActivityPanel;
