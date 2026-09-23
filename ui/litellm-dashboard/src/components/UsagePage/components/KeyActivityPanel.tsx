import { Search, X } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";

import { ActivityMetrics } from "@/components/activity_metrics";
import type { ApiKeyTruncation } from "@/components/EntityUsageExport/exportBlockedReason";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";

import { filterKeyActivity } from "../keyActivityFilter";
import type { ModelActivityData } from "../types";

interface KeyActivityPanelProps {
  keyMetrics: Record<string, ModelActivityData>;
  hidePromptCachingMetrics?: boolean;
  apiKeyTruncation?: ApiKeyTruncation;
  searchKeys?: (query: string) => Promise<Record<string, ModelActivityData>>;
}

type RemoteSearch =
  | { status: "idle" }
  | { status: "loading"; query: string }
  | { status: "done"; query: string; keys: Record<string, ModelActivityData> }
  | { status: "error"; query: string };

const REMOTE_SEARCH_DEBOUNCE_MS = 300;

const KeyActivityPanel: React.FC<KeyActivityPanelProps> = ({
  keyMetrics,
  hidePromptCachingMetrics = false,
  apiKeyTruncation,
  searchKeys,
}) => {
  const [query, setQuery] = useState("");
  const [remote, setRemote] = useState<RemoteSearch>({ status: "idle" });
  const filtered = useMemo(() => filterKeyActivity(keyMetrics, query), [keyMetrics, query]);
  const trimmedQuery = query.trim();
  const remoteEnabled = searchKeys !== undefined && apiKeyTruncation !== undefined && trimmedQuery !== "";

  useEffect(() => {
    if (!remoteEnabled) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      setRemote({ status: "loading", query: trimmedQuery });
      searchKeys(trimmedQuery)
        .then((keys) => {
          if (!cancelled) setRemote({ status: "done", query: trimmedQuery, keys });
        })
        .catch(() => {
          if (!cancelled) setRemote({ status: "error", query: trimmedQuery });
        });
    }, REMOTE_SEARCH_DEBOUNCE_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [remoteEnabled, trimmedQuery, searchKeys]);

  const remoteCurrent = remoteEnabled && "query" in remote && remote.query === trimmedQuery;
  const remoteLoading = remoteEnabled && !remoteCurrent;
  const remoteFailed = remoteCurrent && remote.status === "error";

  const extraRemoteKeys = useMemo(() => {
    const remoteKeys = remoteCurrent && remote.status === "done" ? remote.keys : {};
    return Object.fromEntries(Object.entries(remoteKeys).filter(([hash]) => !(hash in keyMetrics)));
  }, [remoteCurrent, remote, keyMetrics]);
  const displayed = useMemo(() => ({ ...extraRemoteKeys, ...filtered }), [extraRemoteKeys, filtered]);

  const totalKeys = Object.keys(keyMetrics).length;
  const shownKeys = Object.keys(displayed).length;
  const totalShown = totalKeys + Object.keys(extraRemoteKeys).length;
  const isFiltering = trimmedQuery !== "";
  const noMatches = isFiltering && !remoteLoading && totalKeys > 0 && shownKeys === 0;

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
          Showing {shownKeys.toLocaleString()} of {totalShown.toLocaleString()} keys
        </span>
        {remoteLoading && (
          <span role="status" className="text-sm text-muted-foreground">
            Searching all keys...
          </span>
        )}
        {remoteFailed && (
          <span role="alert" className="text-sm text-muted-foreground">
            Key search failed
          </span>
        )}
        {apiKeyTruncation !== undefined && (
          <span className="text-sm text-muted-foreground" role="note">
            Only the {apiKeyTruncation.limit.toLocaleString()} highest-spend keys of{" "}
            {apiKeyTruncation.total.toLocaleString()} are loaded
          </span>
        )}
      </div>
      {noMatches ? (
        <p className="rounded-lg border p-6 text-center text-sm text-muted-foreground">
          No keys match &quot;{trimmedQuery}&quot; in this date range
        </p>
      ) : (
        <ActivityMetrics modelMetrics={displayed} hidePromptCachingMetrics={hidePromptCachingMetrics} />
      )}
    </div>
  );
};

export default KeyActivityPanel;
