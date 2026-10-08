import { KeyRound, Search, X } from "lucide-react";
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import {
  formatCompact,
  formatUsd,
  type SeriesDay,
} from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { Panel, Sparkline, Stat } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";
import { ModelCollapsible, ModelSection } from "@/components/activity_metrics";
import type { Team } from "@/components/key_team_helpers/key_list";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";

import type {
  DailyActivityKeyPageResponse,
  DailyActivityKeySearchResponse,
  KeyActivityRow,
  KeySpendActivityRow,
} from "../dailyActivityApi";
import { filterKeyActivity } from "../keyActivityFilter";
import { keyActivityRowsToMetrics } from "./keySearch";
import { mergeKeyActivityPages } from "../keyActivityData";
import type { ModelActivityData } from "../types";

const PAGE_SIZE = 50;
const SEARCH_DEBOUNCE_MS = 300;
const MIN_SEARCH_LENGTH = 2;
const BRAND = "#2b3fd6";
const KEY_ROW_GRID = "grid w-full grid-cols-[minmax(0,1fr)_5.5rem_5.5rem] items-center gap-4";

type FetchKeyPage = (offset: number, limit: number) => Promise<DailyActivityKeyPageResponse>;
type FetchKeyDetail = (apiKey: string) => Promise<ModelActivityData | undefined>;
type SearchKeys = (search: string) => Promise<DailyActivityKeySearchResponse>;

interface KeyActivityPanelProps {
  summary: ModelActivityData;
  summaryLoading?: boolean;
  fetchKeyPage: FetchKeyPage;
  fetchKeyDetail: FetchKeyDetail;
  searchKeys: SearchKeys;
  teams: Team[];
  hidePromptCachingMetrics?: boolean;
}

interface KeyPageState {
  scope: FetchKeyPage;
  rows: KeySpendActivityRow[];
  total: number;
  nextOffset: number;
  hasMore: boolean;
  loading: boolean;
  loadingMore: boolean;
  failed: boolean;
}

type KeyDetailState =
  | { status: "loading" }
  | { status: "failed" }
  | { status: "loaded"; metrics: ModelActivityData | undefined };

const keyCountText = (searching: boolean, isFiltering: boolean, shownKeys: number, total: number): string => {
  if (searching) return "Searching...";
  if (isFiltering) return `${shownKeys.toLocaleString()} matching keys`;
  return `${total.toLocaleString()} keys`;
};

interface KeyDetailContentProps {
  apiKey: string;
  detail: KeyDetailState | undefined;
  hidePromptCachingMetrics: boolean;
  onRetry: () => void;
}

const KeyDetailContent: React.FC<KeyDetailContentProps> = ({ apiKey, detail, hidePromptCachingMetrics, onRetry }) => {
  if (detail?.status === "loading") {
    return <p className="py-4 text-sm text-muted-foreground">Loading key details...</p>;
  }
  if (detail?.status === "failed") {
    return (
      <p className="py-4 text-sm text-muted-foreground">
        Could not load key details.{" "}
        <button type="button" className="font-medium text-foreground underline" onClick={onRetry}>
          Retry
        </button>
      </p>
    );
  }
  if (detail?.status !== "loaded" || detail.metrics === undefined) {
    return <p className="py-4 text-sm text-muted-foreground">No daily details available</p>;
  }
  return (
    <ModelSection modelName={apiKey} metrics={detail.metrics} hidePromptCachingMetrics={hidePromptCachingMetrics} />
  );
};

const summarySeries = (summary: ModelActivityData): SeriesDay[] =>
  [...summary.daily_data]
    .sort((a, b) => a.date.localeCompare(b.date))
    .map((day) => ({
      date: day.date,
      label: day.date,
      spend: day.metrics.spend,
      requests: day.metrics.api_requests,
      tokens: day.metrics.total_tokens,
    }));

const StatCell: React.FC<{ children: React.ReactNode; trend?: React.ReactNode }> = ({ children, trend }) => (
  <div className="flex min-w-0 flex-col justify-between gap-3 border-b px-4 pt-3.5 pb-3 last:border-b-0 lg:border-b-0">
    <div className="min-w-0">{children}</div>
    {trend}
  </div>
);

interface KeySummaryStripProps {
  summary: ModelActivityData;
  loading: boolean;
  hidePromptCachingMetrics: boolean;
}

const KeySummaryStrip: React.FC<KeySummaryStripProps> = ({ summary, loading, hidePromptCachingMetrics }) => {
  const daily = useMemo(() => summarySeries(summary), [summary]);
  const trend = (dataKey: string) =>
    loading || daily.length < 2 ? undefined : (
      <Sparkline data={daily} dataKey={dataKey} color={BRAND} className="h-12" />
    );
  const avgCost = summary.total_requests > 0 ? summary.total_spend / summary.total_requests : 0;
  return (
    <section
      aria-busy={loading}
      className="grid grid-cols-1 overflow-hidden rounded-xl border bg-card sm:grid-cols-2 lg:grid-cols-4 lg:divide-x"
    >
      {loading ? (
        <span role="status" className="sr-only">
          Loading chart data...
        </span>
      ) : (
        <h3 className="sr-only">Overall Usage</h3>
      )}
      <StatCell trend={trend("spend")}>
        <Stat label="Spend" value={formatUsd(summary.total_spend)} pending={loading} />
      </StatCell>
      <StatCell trend={trend("requests")}>
        <Stat label="Requests" value={summary.total_requests.toLocaleString()} pending={loading} />
        {!loading && (
          <div className="mt-0.5 flex gap-3 text-xs text-muted-foreground">
            <span>
              <span className="text-success tabular-nums">{summary.total_successful_requests.toLocaleString()}</span> ok
            </span>
            <span>
              <span className={summary.total_failed_requests > 0 ? "text-destructive tabular-nums" : "tabular-nums"}>
                {summary.total_failed_requests.toLocaleString()}
              </span>{" "}
              failed
            </span>
          </div>
        )}
      </StatCell>
      <StatCell trend={trend("tokens")}>
        <Stat
          label="Tokens"
          value={summary.total_tokens.toLocaleString()}
          pending={loading}
          hint={
            loading
              ? undefined
              : `${formatCompact(summary.prompt_tokens)} in · ${formatCompact(summary.completion_tokens)} out`
          }
        />
      </StatCell>
      <StatCell>
        <Stat
          label="Avg cost / request"
          value={formatUsd(avgCost, 4)}
          pending={loading}
          hint={
            loading || hidePromptCachingMetrics
              ? undefined
              : `${formatCompact(summary.total_cache_read_input_tokens)} cache read tokens`
          }
        />
      </StatCell>
    </section>
  );
};

const KeyActivityPanel: React.FC<KeyActivityPanelProps> = ({
  summary,
  summaryLoading = false,
  fetchKeyPage,
  fetchKeyDetail,
  searchKeys,
  teams,
  hidePromptCachingMetrics = false,
}) => {
  const [queryState, setQueryState] = useState<{ scope: FetchKeyPage; value: string } | null>(null);
  const query = queryState?.scope === fetchKeyPage ? queryState.value : "";
  const updateQuery = useCallback((value: string) => setQueryState({ scope: fetchKeyPage, value }), [fetchKeyPage]);
  const [pageState, setPageState] = useState<KeyPageState | null>(null);
  const [detailState, setDetailState] = useState<{
    scope: FetchKeyPage;
    details: Record<string, KeyDetailState>;
  } | null>(null);
  const [searchResult, setSearchResult] = useState<{
    scope: FetchKeyPage;
    term: string;
    searchKeys: SearchKeys;
    rows: KeyActivityRow[];
    failed: boolean;
  } | null>(null);
  const [retryToken, setRetryToken] = useState(0);
  const [searchRetryToken, setSearchRetryToken] = useState(0);
  const searchIdRef = useRef(0);
  const pageRequestIdRef = useRef(0);
  const loadingMoreRef = useRef(false);
  const activeFetcherRef = useRef(fetchKeyPage);
  const sentinelRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    activeFetcherRef.current = fetchKeyPage;
  }, [fetchKeyPage]);

  useEffect(() => {
    const requestId = ++pageRequestIdRef.current;
    loadingMoreRef.current = false;
    void fetchKeyPage(0, PAGE_SIZE)
      .then((page) => {
        if (pageRequestIdRef.current !== requestId || activeFetcherRef.current !== fetchKeyPage) return;
        const loadedPageState = {
          scope: fetchKeyPage,
          rows: page.api_keys,
          total: page.total_api_keys,
          nextOffset: page.api_keys.length,
          hasMore: page.api_keys.length > 0 && page.api_keys.length < page.total_api_keys,
          loading: false,
          loadingMore: false,
          failed: false,
        };
        setPageState(loadedPageState);
      })
      .catch((error: unknown) => {
        if (pageRequestIdRef.current !== requestId || activeFetcherRef.current !== fetchKeyPage) return;
        console.error("Key activity page request failed:", error);
        const failedPageState = {
          scope: fetchKeyPage,
          rows: [],
          total: 0,
          nextOffset: 0,
          hasMore: false,
          loading: false,
          loadingMore: false,
          failed: true,
        };
        setPageState(failedPageState);
      });
    return () => {
      if (pageRequestIdRef.current === requestId) pageRequestIdRef.current += 1;
    };
  }, [fetchKeyPage, retryToken]);

  const currentPageState = pageState?.scope === fetchKeyPage ? pageState : null;
  const pageRows = useMemo(() => currentPageState?.rows ?? [], [currentPageState]);
  const loading = currentPageState?.loading ?? true;
  const loadingMore = currentPageState?.loadingMore ?? false;
  const hasMore = currentPageState?.hasMore ?? false;
  const total = currentPageState?.total ?? 0;
  const failed = currentPageState?.failed ?? false;
  const pageMetrics = useMemo(() => keyActivityRowsToMetrics(pageRows, teams), [pageRows, teams]);

  const loadMore = useCallback(() => {
    if (currentPageState === null) return;
    if (currentPageState.loading || currentPageState.loadingMore || !currentPageState.hasMore) return;
    if (query.trim() !== "" || loadingMoreRef.current) return;
    const requestId = pageRequestIdRef.current;
    const offset = currentPageState.nextOffset;
    loadingMoreRef.current = true;
    setPageState((current) => (current?.scope === fetchKeyPage ? { ...current, loadingMore: true } : current));
    void fetchKeyPage(offset, PAGE_SIZE)
      .then((page) => {
        if (pageRequestIdRef.current !== requestId || activeFetcherRef.current !== fetchKeyPage) return;
        setPageState((current) => {
          if (current?.scope !== fetchKeyPage) return current;
          const merged = mergeKeyActivityPages(current.rows, page.api_keys, page.total_api_keys, offset);
          return {
            ...current,
            rows: merged.rows,
            total: page.total_api_keys,
            nextOffset: merged.nextOffset,
            hasMore: merged.hasMore,
            loadingMore: false,
            failed: false,
          };
        });
      })
      .catch((error: unknown) => {
        if (pageRequestIdRef.current !== requestId || activeFetcherRef.current !== fetchKeyPage) return;
        console.error("Key activity page request failed:", error);
        setPageState((current) =>
          current?.scope === fetchKeyPage ? { ...current, loadingMore: false, failed: true } : current,
        );
      })
      .finally(() => {
        if (pageRequestIdRef.current === requestId) loadingMoreRef.current = false;
      });
  }, [currentPageState, fetchKeyPage, query]);

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (sentinel === null || currentPageState === null) return;
    if (currentPageState.failed) return;
    if (loading || loadingMore || !hasMore) return;
    if (query.trim() !== "") return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) loadMore();
    });
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [currentPageState, hasMore, loadMore, loading, loadingMore, query]);

  const trimmedQuery = query.trim();
  const searchTerm = trimmedQuery.length >= MIN_SEARCH_LENGTH ? trimmedQuery : null;
  useEffect(() => {
    if (searchTerm === null) return;
    const searchId = ++searchIdRef.current;
    const timer = setTimeout(() => {
      void searchKeys(searchTerm)
        .then((response) => {
          if (searchIdRef.current !== searchId || activeFetcherRef.current !== fetchKeyPage) return;
          const result = { scope: fetchKeyPage, term: searchTerm, searchKeys, rows: response.api_keys, failed: false };
          setSearchResult(result);
        })
        .catch((error: unknown) => {
          if (searchIdRef.current !== searchId || activeFetcherRef.current !== fetchKeyPage) return;
          console.error("Key activity search failed:", error);
          const failedResult = { scope: fetchKeyPage, term: searchTerm, searchKeys, rows: [], failed: true };
          setSearchResult(failedResult);
        });
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      if (searchIdRef.current === searchId) searchIdRef.current += 1;
    };
  }, [fetchKeyPage, searchKeys, searchRetryToken, searchTerm]);

  const currentSearch =
    searchResult?.scope === fetchKeyPage && searchResult.term === searchTerm && searchResult.searchKeys === searchKeys
      ? searchResult
      : null;
  const searching = searchTerm !== null && currentSearch === null;
  const searchFailed = currentSearch?.failed ?? false;
  const searchRows = currentSearch?.rows;
  const searchMetrics = useMemo(() => keyActivityRowsToMetrics(searchRows ?? [], teams), [searchRows, teams]);
  const localFiltered = useMemo(() => filterKeyActivity(pageMetrics, query), [pageMetrics, query]);
  const filteredMetrics = useMemo(() => ({ ...searchMetrics, ...localFiltered }), [searchMetrics, localFiltered]);
  const shownKeys = Object.keys(filteredMetrics).length;
  const isFiltering = trimmedQuery.length > 0;
  const searchSettled = isFiltering && !searching;
  const visibleDetails = detailState?.scope === fetchKeyPage ? detailState.details : {};
  const keyCountLabel =
    searchSettled && searchFailed && shownKeys === 0
      ? "Search failed"
      : keyCountText(searching, isFiltering, shownKeys, total);

  const retryFirstPage = () => {
    setPageState({
      scope: fetchKeyPage,
      rows: [],
      total: 0,
      nextOffset: 0,
      hasMore: false,
      loading: true,
      loadingMore: false,
      failed: false,
    });
    setRetryToken((token) => token + 1);
  };
  const retrySearch = () => {
    setSearchResult(null);
    setSearchRetryToken((token) => token + 1);
  };
  const retryNextPage = () => {
    setPageState((current) => (current?.scope === fetchKeyPage ? { ...current, failed: false } : current));
    loadMore();
  };
  const firstPageError = (
    <p className="px-4 py-8 text-center text-sm text-muted-foreground">
      Could not load keys for this range.{" "}
      <button type="button" className="font-medium text-foreground underline" onClick={retryFirstPage}>
        Retry
      </button>
    </p>
  );
  const emptyListBody = (() => {
    if (failed) return firstPageError;
    if (!loading) return null;
    return <p className="px-4 py-8 text-center text-sm text-muted-foreground">Loading keys...</p>;
  })();
  const maxSpend = Math.max(0, ...Object.values(filteredMetrics).map((metrics) => metrics.total_spend));
  const keyList = Object.entries(filteredMetrics).map(([apiKey, metrics]) => {
    const detail = visibleDetails[apiKey];
    const share = maxSpend > 0 ? (metrics.total_spend / maxSpend) * 100 : 0;
    return (
      <ModelCollapsible
        key={apiKey}
        defaultOpen={false}
        onFirstOpen={() => loadKeyDetail(apiKey)}
        header={
          <div className={KEY_ROW_GRID}>
            <div className="min-w-0">
              <div className="truncate text-sm text-foreground">{metrics.label}</div>
              <div aria-hidden="true" className="mt-1 h-1 w-full max-w-xs overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full opacity-80"
                  style={{ width: `${share}%`, backgroundColor: BRAND }}
                />
              </div>
            </div>
            <span className="text-right text-sm tabular-nums text-foreground">{formatUsd(metrics.total_spend)}</span>
            <span className="text-right text-sm tabular-nums text-muted-foreground">
              {metrics.total_requests.toLocaleString()} <span className="sr-only">requests</span>
            </span>
          </div>
        }
      >
        <KeyDetailContent
          apiKey={apiKey}
          detail={detail}
          hidePromptCachingMetrics={hidePromptCachingMetrics}
          onRetry={() => loadKeyDetail(apiKey)}
        />
      </ModelCollapsible>
    );
  });
  const firstPageFailed = failed && pageRows.length === 0;
  const showNextPageRetry = failed && !firstPageFailed && !isFiltering;
  const localOnly = searchTerm === null;
  const noLocalRows = localOnly && pageRows.length === 0;
  const pageUnavailable = noLocalRows && (loading || firstPageFailed);
  const listBody = pageUnavailable || (!isFiltering && pageRows.length === 0) ? emptyListBody : keyList;
  const noMatches = searchSettled && !pageUnavailable && shownKeys === 0;
  const showSearchErrorNote = !noMatches && searchFailed && !searching;
  const searchRetryButton = (
    <button type="button" className="font-medium text-foreground underline" onClick={retrySearch}>
      Retry search
    </button>
  );
  const emptyFilterBody = searchFailed ? (
    <p className="px-4 py-8 text-center text-sm text-muted-foreground">Could not search keys. {searchRetryButton}</p>
  ) : (
    <p className="px-4 py-8 text-center text-sm text-muted-foreground">
      No keys match &quot;{trimmedQuery}&quot; in this date range
    </p>
  );

  const loadKeyDetail = useCallback(
    (apiKey: string) => {
      const existingDetails = detailState?.scope === fetchKeyPage ? detailState.details : {};
      const existing = existingDetails[apiKey];
      if (existing !== undefined && existing.status !== "failed") return;
      setDetailState((current) => {
        const currentDetails = current?.scope === fetchKeyPage ? current.details : {};
        return { scope: fetchKeyPage, details: { ...currentDetails, [apiKey]: { status: "loading" } } };
      });
      void fetchKeyDetail(apiKey)
        .then((metrics) => {
          setDetailState((current) => {
            if (current?.scope !== fetchKeyPage) return current;
            return { scope: fetchKeyPage, details: { ...current.details, [apiKey]: { status: "loaded", metrics } } };
          });
        })
        .catch((error: unknown) => {
          console.error("Key activity detail request failed:", error);
          setDetailState((current) => {
            if (current?.scope !== fetchKeyPage) return current;
            return {
              scope: fetchKeyPage,
              details: { ...current.details, [apiKey]: { status: "failed" } },
            };
          });
        });
    },
    [detailState, fetchKeyDetail, fetchKeyPage],
  );

  const showColumnHeaders = !noMatches && listBody === keyList && keyList.length > 0;
  const showFooter = showSearchErrorNote || loadingMore || showNextPageRetry;
  const searchInput = (
    <InputGroup className="h-8 w-full sm:w-72">
      <InputGroupAddon>
        <Search className="size-3.5 text-muted-foreground" />
      </InputGroupAddon>
      <InputGroupInput
        aria-label="Search keys"
        placeholder="Search alias, hash, user ID, or email"
        value={query}
        onChange={(event) => updateQuery(event.target.value)}
      />
      {isFiltering && (
        <InputGroupAddon align="inline-end">
          <InputGroupButton size="icon-xs" aria-label="Clear key search" onClick={() => updateQuery("")}>
            <X />
          </InputGroupButton>
        </InputGroupAddon>
      )}
    </InputGroup>
  );

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-3">
        {searchInput}
        {!pageUnavailable && (
          <span className="text-xs text-muted-foreground tabular-nums" aria-live="polite">
            {keyCountLabel}
          </span>
        )}
      </div>
      <KeySummaryStrip summary={summary} loading={summaryLoading} hidePromptCachingMetrics={hidePromptCachingMetrics} />
      <Panel icon={KeyRound} title="Virtual keys" className="overflow-hidden" bodyClassName="px-0 pt-3 pb-0">
        {showColumnHeaders && (
          <div
            aria-hidden="true"
            className={`${KEY_ROW_GRID} border-y bg-muted/40 py-2 pr-4 pl-10 text-xs font-medium text-muted-foreground`}
          >
            <span>Key</span>
            <span className="text-right">Spend</span>
            <span className="text-right">Requests</span>
          </div>
        )}
        <div className={showColumnHeaders ? undefined : "border-t"}>{noMatches ? emptyFilterBody : listBody}</div>
        {showFooter && (
          <div className="border-t px-4 py-2.5 text-xs text-muted-foreground">
            {showSearchErrorNote && <p>Could not search all keys. {searchRetryButton}</p>}
            {loadingMore && <p>Loading more keys...</p>}
            {showNextPageRetry && (
              <p>
                Could not load more keys.{" "}
                <button type="button" className="font-medium text-foreground underline" onClick={retryNextPage}>
                  Retry
                </button>
              </p>
            )}
          </div>
        )}
      </Panel>
      {!isFiltering && <div ref={sentinelRef} aria-hidden="true" className="h-1" />}
    </div>
  );
};

export default KeyActivityPanel;
