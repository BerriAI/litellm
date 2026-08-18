"use client";

import { AlertTriangle, Pause, Play, RefreshCw } from "lucide-react";
import type { PaginationState } from "@tanstack/react-table";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DataTable } from "@/components/shared/DataTable";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  type ActiveRequest,
  type ActiveRequestFilters,
  activeRequestsCall,
  cancelActiveRequestCall,
} from "./activeRequestsApi";
import ActiveRequestCharts from "./ActiveRequestCharts";
import ActiveRequestDetail from "./ActiveRequestDetail";
import { activeRequestColumns, formatAge } from "./ActiveRequestColumns";

const POLL_INTERVAL_MS = 5000;
const PAGE_SIZE = 50;
const FILTER_KEYS = ["model", "end_user_id", "user_id", "organization_id", "project_id"] as const;
const FILTER_LABELS: Record<(typeof FILTER_KEYS)[number], string> = {
  model: "Model",
  end_user_id: "End User ID",
  user_id: "User ID",
  organization_id: "Organization ID",
  project_id: "Project ID",
};

const DEFAULT_SORTING = [{ id: "age", desc: false }];

const filtersEqual = (left: ActiveRequestFilters, right: ActiveRequestFilters) =>
  FILTER_KEYS.every((key) => left[key] === right[key]);

const Banner = ({ title, description }: { title: string; description: string }) => (
  <div
    role="alert"
    className="flex items-start gap-3 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3"
  >
    <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden />
    <div className="min-w-0">
      <p className="text-sm font-medium text-foreground">{title}</p>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  </div>
);

const StatTile = ({ title, value }: { title: string; value: string | number }) => (
  <Card>
    <CardHeader>
      <CardTitle className="text-sm font-normal text-muted-foreground">{title}</CardTitle>
    </CardHeader>
    <CardContent>
      <p className="text-2xl font-semibold tabular-nums">{value}</p>
    </CardContent>
  </Card>
);

interface ActiveRequestsProps {
  accessToken: string | null;
}

export default function ActiveRequests({ accessToken }: ActiveRequestsProps) {
  const [items, setItems] = useState<ActiveRequest[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [unavailableReason, setUnavailableReason] = useState<string | null>(null);
  const [truncated, setTruncated] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const router = useRouter();
  const searchParams = useSearchParams();
  const filtersFromUrl = useMemo<ActiveRequestFilters>(
    () =>
      FILTER_KEYS.reduce<ActiveRequestFilters>(
        (acc, key) => ({ ...acc, [key]: searchParams.get(key) || undefined }),
        {},
      ),
    [searchParams],
  );
  const [draftFilters, setDraftFilters] = useState<ActiveRequestFilters>(filtersFromUrl);
  const [filters, setFilters] = useState<ActiveRequestFilters>(filtersFromUrl);
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(paused);
  const [selected, setSelected] = useState<ActiveRequest | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const requestControllerRef = useRef<AbortController | null>(null);

  const loadRequests = useCallback(async () => {
    if (!accessToken || requestControllerRef.current) return;
    const controller = new AbortController();
    requestControllerRef.current = controller;
    setLoading(true);
    try {
      const response = await activeRequestsCall({ ...filters, page, page_size: PAGE_SIZE }, controller.signal);
      setItems(response.items);
      setTotal(response.total);
      setUnavailableReason(response.available ? null : response.reason || "Registry unavailable");
      setTruncated(Boolean(response.truncated));
      setLastUpdated(new Date(response.generated_at));
      setError(null);
      const lastPage = Math.max(1, Math.ceil(response.total / PAGE_SIZE));
      if (page > lastPage) setPage(lastPage);
    } catch (requestError) {
      if (requestError instanceof DOMException && requestError.name === "AbortError") return;
      setError(requestError instanceof Error ? requestError.message : "Failed to load active requests");
    } finally {
      if (requestControllerRef.current === controller) {
        requestControllerRef.current = null;
        setLoading(false);
      }
    }
  }, [accessToken, filters, page]);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  useEffect(() => {
    const timeout = window.setTimeout(() => void loadRequests(), 0);
    const interval = window.setInterval(() => {
      if (!pausedRef.current && document.visibilityState === "visible") void loadRequests();
    }, POLL_INTERVAL_MS);
    const handleVisibilityChange = () => {
      if (!pausedRef.current && document.visibilityState === "visible") void loadRequests();
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      window.clearTimeout(timeout);
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      requestControllerRef.current?.abort();
      requestControllerRef.current = null;
    };
  }, [loadRequests]);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      setPage(1);
      setFilters((current) => (filtersEqual(current, draftFilters) ? current : draftFilters));
      const params = new URLSearchParams();
      FILTER_KEYS.forEach((key) => {
        const value = draftFilters[key];
        if (value) params.set(key, value);
      });
      const query = params.toString();
      router.replace(query ? `?${query}` : "?", { scroll: false });
    }, 350);
    return () => window.clearTimeout(timeout);
  }, [draftFilters, router]);

  useEffect(() => {
    const interval = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, []);

  const longestSeconds = useMemo(() => Math.max(0, ...items.map((item) => now / 1000 - item.started_at)), [items, now]);
  const distinctEndUsers = useMemo(() => new Set(items.map((item) => item.end_user_id).filter(Boolean)).size, [items]);
  const distinctModels = useMemo(() => new Set(items.map((item) => item.model).filter(Boolean)).size, [items]);
  const pagination = useMemo<PaginationState>(() => ({ pageIndex: page - 1, pageSize: PAGE_SIZE }), [page]);
  const onPaginationChange = useCallback(
    (updater: PaginationState | ((old: PaginationState) => PaginationState)) => {
      const next = typeof updater === "function" ? updater({ pageIndex: page - 1, pageSize: PAGE_SIZE }) : updater;
      setPage(next.pageIndex + 1);
    },
    [page],
  );

  const updateFilter = (key: keyof ActiveRequestFilters, value: string) => {
    setDraftFilters((current) => ({ ...current, [key]: value || undefined }));
  };

  const cancelRequest = useCallback(
    async (request: ActiveRequest) => {
      setCancelling(true);
      try {
        await cancelActiveRequestCall(request.registry_id);
        setSelected(null);
        await loadRequests();
      } catch (cancelError) {
        setError(cancelError instanceof Error ? cancelError.message : "Failed to cancel the request");
      } finally {
        setCancelling(false);
      }
    },
    [loadRequests],
  );

  return (
    <div className="flex flex-col gap-5 p-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Active Requests</h1>
        <p className="text-sm text-muted-foreground">
          Authenticated requests currently running across all LiteLLM replicas. Refreshes every 5 seconds.
        </p>
      </div>

      {error && <Banner title="Could not refresh active requests" description={error} />}
      {unavailableReason && <Banner title="Live registry unavailable" description={unavailableReason} />}
      {truncated && (
        <Banner
          title="Filtered results are incomplete"
          description="The active request index is larger than the scan limit. Narrow the filters or reduce LITELLM_ACTIVE_REQUEST_TTL_SECONDS."
        />
      )}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatTile title="Active" value={total} />
        <StatTile title="Longest on page" value={formatAge(now / 1000 - longestSeconds, now)} />
        <StatTile title="End users on page" value={distinctEndUsers} />
        <StatTile title="Models on page" value={distinctModels} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-end gap-3">
            {FILTER_KEYS.map((key) => (
              <div key={key} className="flex min-w-40 flex-1 flex-col gap-1.5">
                <Label htmlFor={`active-requests-${key}`}>{FILTER_LABELS[key]}</Label>
                <Input
                  id={`active-requests-${key}`}
                  placeholder={FILTER_LABELS[key]}
                  value={draftFilters[key] ?? ""}
                  onChange={(event) => updateFilter(key, event.target.value)}
                />
              </div>
            ))}
            <Button variant="outline" onClick={() => void loadRequests()} disabled={loading}>
              <RefreshCw className={loading ? "animate-spin" : undefined} aria-hidden />
              Refresh
            </Button>
            <div className="flex items-center gap-2">
              <Switch id="active-requests-pause" checked={paused} onCheckedChange={setPaused} />
              <Label htmlFor="active-requests-pause" className="flex items-center gap-1.5">
                {paused ? <Play className="size-3.5" aria-hidden /> : <Pause className="size-3.5" aria-hidden />}
                {paused ? "Paused" : "Auto refresh"}
              </Label>
            </div>
          </div>
        </CardContent>
      </Card>

      <ActiveRequestCharts items={items} now={now} />

      <Card>
        <CardHeader>
          <CardTitle>Running requests</CardTitle>
          <CardAction>
            <span className="text-sm text-muted-foreground" aria-live="polite">
              {lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString()}` : "Waiting for data"}
            </span>
          </CardAction>
        </CardHeader>
        <CardContent>
          <DataTable
            data={items}
            columns={activeRequestColumns}
            getRowId={(item, index) => `${item.request_id}-${item.started_at}-${index}`}
            paginationMode="server"
            pagination={pagination}
            onPaginationChange={onPaginationChange}
            rowCount={total}
            pageSizeOptions={[PAGE_SIZE]}
            isLoading={loading && items.length === 0}
            loadingMessage="Loading active requests…"
            noDataMessage="No authenticated requests are running right now."
            size="compact"
            sortingMode="client"
            defaultSorting={DEFAULT_SORTING}
            onRowClick={setSelected}
          />
        </CardContent>
      </Card>

      <ActiveRequestDetail
        request={selected}
        now={now}
        onClose={() => setSelected(null)}
        onCancel={(request) => void cancelRequest(request)}
        cancelling={cancelling}
      />
    </div>
  );
}
