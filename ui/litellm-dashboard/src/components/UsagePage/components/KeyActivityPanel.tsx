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
  searchKeys?: SearchKeys;
  loadMoreKeys?: () => Promise<void>;
}

type SearchKeys = (query: string) => Promise<Record<string, ModelActivityData>>;

type RemoteSearch =
  | { status: "idle" }
  | { status: "loading"; query: string; searchKeys: SearchKeys }
  | { status: "done"; query: string; searchKeys: SearchKeys; keys: Record<string, ModelActivityData> }
  | { status: "error"; query: string; searchKeys: SearchKeys };

type LoadMoreState = { status: "idle" } | { status: "loading" } | { status: "error" };

const REMOTE_SEARCH_DEBOUNCE_MS = 300;

const KeyActivityPanel: React.FC<KeyActivityPanelProps> = ({
  keyMetrics,
  hidePromptCachingMetrics = false,
  apiKeyTruncation,
  searchKeys,
  loadMoreKeys,
}) => {
  const [query, setQuery] = useState("");
  const [remote, setRemote] = useState<RemoteSearch>({ status: "idle" });
  const [loadMore, setLoadMore] = useState<LoadMoreState>({ status: "idle" });
  const filtered = useMemo(() => filterKeyActivity(keyMetrics, query), [keyMetrics, query]);
  const trimmedQuery = query.trim();
  const remoteEnabled = searchKeys !== undefined && apiKeyTruncation !== undefined && trimmedQuery !== "";

  useEffect(() => {
    if (!remoteEnabled) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      setRemote({ status: "loading", query: trimmedQuery, searchKeys });
      searchKeys(trimmedQuery)
        .then((keys) => {
          if (!cancelled) setRemote({ status: "done", query: trimmedQuery, searchKeys, keys });
        })
        .catch(() => {
          if (!cancelled) setRemote({ status: "error", query: trimmedQuery, searchKeys });
        });
    }, REMOTE_SEARCH_DEBOUNCE_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [remoteEnabled, trimmedQuery, searchKeys]);

  const remoteMatchesSearch =
    "searchKeys" in remote && remote.searchKeys === searchKeys && remote.query === trimmedQuery;
  const remoteCurrent = remoteEnabled && remoteMatchesSearch;
  const remoteLoading = remoteEnabled && (remote.status === "loading" || !remoteCurrent);
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
        {apiKeyTruncation !== undefined && totalKeys < apiKeyTruncation.total && (
          <span className="text-sm text-muted-foreground" role="note">
            Only the {totalKeys.toLocaleString()} highest-spend keys of {apiKeyTruncation.total.toLocaleString()} are
            loaded
          </span>
        )}
        {loadMoreKeys !== undefined && (
          <button
            type="button"
            className="text-sm text-muted-foreground underline disabled:no-underline disabled:opacity-50"
            disabled={loadMore.status === "loading"}
            onClick={() => {
              setLoadMore({ status: "loading" });
              loadMoreKeys()
                .then(() => setLoadMore({ status: "idle" }))
                .catch(() => setLoadMore({ status: "error" }));
            }}
          >
            Load more keys
          </button>
        )}
        {loadMore.status === "loading" && (
          <span role="status" className="text-sm text-muted-foreground">
            Loading more keys...
          </span>
        )}
        {loadMore.status === "error" && (
          <span role="alert" className="text-sm text-muted-foreground">
            Loading more keys failed
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
