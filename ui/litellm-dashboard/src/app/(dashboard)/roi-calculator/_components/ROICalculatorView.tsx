"use client";

import { Page, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import React from "react";
import { Calculator, RefreshCw, Settings2 } from "lucide-react";

import { apiClient } from "@/components/networking";
import { DemoNotice } from "@/components/shared/DemoNotice";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { extractErrorMessage } from "@/utils/errorUtils";
import { isProxyAdminTierRole } from "@/utils/roles";
import ROISettingsPanel from "./ROISettingsPanel";
import { IdentityMatchDialog, type PersonMatchSelection, PullReasoningDialog } from "./ROICalculatorDialogs";
import { ROIBranches, ROIOverview, ROIPeopleView } from "./ROICalculatorViews";
import { filterPulls, formatSyncedAt } from "./roiCalculatorData";
import type {
  ROIIdentityMapResponse,
  ROIIdentityMapUpdate,
  ROIPull,
  ROIReportResponse,
  ROISettings,
  ROISummary,
  ROISyncStatus,
} from "./roiCalculatorData";

type View = "overview" | "people" | "branches";

const IDLE_STATUS: ROISyncStatus = {
  running: false,
  elapsed_seconds: 0,
  phase: "idle",
  stage: "Idle",
  done: 0,
  total: 0,
  estimated: 0,
  reused: 0,
  needs_attention: 0,
  error: null,
};

function updateDemoUrl(enabled: boolean) {
  const url = new URL(window.location.href);
  if (enabled) url.searchParams.set("demo", "1");
  else url.searchParams.delete("demo");
  window.history.replaceState(null, "", url);
}

export default function ROICalculatorView({
  accessToken,
  userRole = null,
  isViewOnly = false,
}: {
  accessToken: string | null;
  userRole?: string | null;
  isViewOnly?: boolean;
}) {
  const [sampleSummary, setSampleSummary] = React.useState<ROISummary | null>(null);
  const adminReadOnly = isViewOnly && isProxyAdminTierRole(userRole ?? "");
  const readOnly = adminReadOnly || sampleSummary !== null;
  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const [view, setView] = React.useState<View>("overview");
  const [settings, setSettings] = React.useState<ROISettings | null>(null);
  const [liveSummary, setSummary] = React.useState<ROISummary | null>(null);
  const summary = sampleSummary ?? liveSummary;
  const [status, setStatus] = React.useState<ROISyncStatus>(IDLE_STATUS);
  const [selectedPull, setSelectedPull] = React.useState<ROIPull | null>(null);
  const [matchingPerson, setMatchingPerson] = React.useState<PersonMatchSelection | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [demoError, setDemoError] = React.useState<string | null>(null);
  const [reportError, setReportError] = React.useState<string | null>(null);
  const [syncError, setSyncError] = React.useState<string | null>(null);
  const [loadingInitialData, setLoadingInitialData] = React.useState(true);
  const [loadingLiveData, setLoadingLiveData] = React.useState(true);
  const statusRef = React.useRef<ROISyncStatus>(IDLE_STATUS);
  const reportNeedsRefresh = React.useRef(false);
  const sourceRevision = React.useRef(0);
  const settingsLoaded = settings !== null && !loadingInitialData && !loadingLiveData;
  const requestError = [error, demoError, reportError, syncError].filter(Boolean).join(" ");
  const [query, setQuery] = React.useState("");

  const loadReport = React.useCallback(async () => {
    if (!accessToken) return null;
    const response: ROIReportResponse = await apiClient.get("/roi-calculator/report", { accessToken });
    return response.report;
  }, [accessToken]);

  React.useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    const demoRequested = new URLSearchParams(window.location.search).get("demo") === "1";
    const settingsRequest = apiClient.get<ROISettings>("/roi-calculator/settings", { accessToken });
    const reportRequest = apiClient
      .get<ROIReportResponse>("/roi-calculator/report", { accessToken })
      .then((response) => {
        if (cancelled) return;
        setSummary(response.report);
        setReportError(null);
        reportNeedsRefresh.current = false;
      })
      .catch((reason: unknown) => {
        if (cancelled) return;
        setReportError(extractErrorMessage(reason));
        reportNeedsRefresh.current = true;
      });
    const statusRequest = apiClient
      .get<ROISyncStatus>("/roi-calculator/sync", { accessToken })
      .then((syncStatus) => {
        if (cancelled) return;
        setStatus(syncStatus);
        statusRef.current = syncStatus;
        setSyncError(null);
      })
      .catch((reason: unknown) => {
        if (!cancelled) setSyncError(extractErrorMessage(reason));
      });
    const liveData = Promise.all([reportRequest, statusRequest])
      .then(() => null)
      .finally(() => {
        if (!cancelled) setLoadingLiveData(false);
      });
    Promise.all([
      settingsRequest,
      demoRequested
        ? apiClient
            .get<ROIReportResponse>("/roi-calculator/report", { accessToken, query: { mode: "demo" } })
            .catch((reason: unknown) => {
              if (!cancelled) {
                setDemoError(`Could not load demo data: ${extractErrorMessage(reason)}`);
                updateDemoUrl(false);
              }
              return liveData;
            })
        : liveData,
    ])
      .then(([nextSettings, sampleResponse]) => {
        if (cancelled) return;
        setSettings(nextSettings);
        setSampleSummary(sampleResponse?.report ?? null);
        setError(null);
        if (sampleResponse) setDemoError(null);
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(extractErrorMessage(reason));
      })
      .finally(() => {
        if (!cancelled) setLoadingInitialData(false);
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken]);

  React.useEffect(() => {
    if (!accessToken || !settingsLoaded) return;
    let cancelled = false;
    let requestInFlight = false;
    const interval = window.setInterval(() => {
      if (requestInFlight) return;
      requestInFlight = true;
      const revision = sourceRevision.current;
      const isCurrent = () => !cancelled && revision === sourceRevision.current;
      apiClient
        .get<ROISyncStatus>("/roi-calculator/sync", { accessToken })
        .then(async (nextStatus) => {
          if (!isCurrent()) return;
          const previousStatus = statusRef.current;
          statusRef.current = nextStatus;
          setStatus(nextStatus);
          setSyncError(null);
          const finished = !nextStatus.running && nextStatus.phase === "complete";
          const reportChanged = previousStatus.running || nextStatus.finished_at !== previousStatus.finished_at;
          if (reportNeedsRefresh.current || (finished && reportChanged)) {
            reportNeedsRefresh.current = true;
            try {
              const report = await loadReport();
              if (!isCurrent()) return;
              setSummary(report);
              setReportError(null);
              reportNeedsRefresh.current = false;
            } catch (reason: unknown) {
              if (isCurrent()) setReportError(extractErrorMessage(reason));
            }
          }
        })
        .catch((reason: unknown) => {
          if (isCurrent()) setSyncError(extractErrorMessage(reason));
        })
        .finally(() => {
          requestInFlight = false;
        });
    }, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [accessToken, loadReport, settingsLoaded]);

  const startSync = React.useCallback(async () => {
    if (!accessToken || readOnly) return;
    try {
      setError(null);
      const nextStatus = await apiClient.post<ROISyncStatus>("/roi-calculator/sync", { accessToken });
      statusRef.current = nextStatus;
      setStatus(nextStatus);
      setSyncError(null);
      setSettingsOpen(false);
    } catch (reason) {
      setError(extractErrorMessage(reason));
    }
  }, [accessToken, readOnly]);

  const cancelSync = React.useCallback(async () => {
    if (!accessToken || readOnly) return;
    try {
      const nextStatus = await apiClient.delete<ROISyncStatus>("/roi-calculator/sync", { accessToken });
      setStatus(nextStatus);
      statusRef.current = nextStatus;
      setSyncError(null);
      setError(null);
    } catch (reason) {
      setError(extractErrorMessage(reason));
    }
  }, [accessToken, readOnly]);

  const updateIdentity = React.useCallback(
    async (payload: ROIIdentityMapUpdate) => {
      if (!accessToken || readOnly) return;
      const response: ROIIdentityMapResponse = await apiClient.put("/roi-calculator/identity-map", {
        accessToken,
        body: payload,
      });
      setSummary(response.report);
      setReportError(null);
      reportNeedsRefresh.current = false;
      setSettings((current) => (current ? { ...current, identity_map: response.identity_map } : current));
    },
    [accessToken, readOnly],
  );

  const filteredPulls = React.useMemo(() => (summary ? filterPulls(summary.pulls, query) : []), [query, summary]);

  if (error && !settings && !loadingInitialData) {
    return (
      <div className="p-8">
        <Alert variant="destructive">
          <AlertTitle>Could not load ROI Calculator</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  const awaitingLiveData = !sampleSummary && loadingLiveData;
  if (!settings || loadingInitialData || awaitingLiveData) {
    return (
      <div className="space-y-6 p-8">
        <p role="status">Loading ROI Calculator…</p>
        <Skeleton className="h-16 w-96" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  const previewSample = async () => {
    try {
      const response = await apiClient.get<ROIReportResponse>("/roi-calculator/report", {
        accessToken,
        query: { mode: "demo" },
      });
      setSampleSummary(response.report);
      setDemoError(null);
      updateDemoUrl(true);
      setView("branches");
      setQuery("");
    } catch (reason) {
      setDemoError(`Could not load demo data: ${extractErrorMessage(reason)}`);
    }
  };
  const resetView = (updated: ROISettings, resetSyncStatus = true) => {
    sourceRevision.current += 1;
    setSettings(updated);
    setSummary(null);
    setReportError(null);
    reportNeedsRefresh.current = false;
    setSettingsOpen(false);
    setView("overview");
    if (resetSyncStatus) {
      setStatus(IDLE_STATUS);
      statusRef.current = IDLE_STATUS;
      setSyncError(null);
      setError(null);
    }
  };
  const showLiveStatus = !sampleSummary && !status.running;
  const showReportActions = summary !== null || reportError !== null;
  const progress = status.total > 0 ? Math.min(100, (status.done / status.total) * 100) : 0;
  const statusIsIdleOrComplete = status.phase === "idle" || status.phase === "complete";
  const syncIsUpToDate = !status.running && statusIsIdleOrComplete;
  const syncedAt = sampleSummary?.synced_at ?? (syncIsUpToDate ? summary?.synced_at : null);

  return (
    <Page className="gap-6">
      <PageHeader className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <PageHeaderTitle>
            <Calculator />
            ROI Calculator
          </PageHeaderTitle>
          {!sampleSummary && (
            <div className="flex items-center gap-2">
              {showLiveStatus && (
                <Button variant="ghost" onClick={() => void previewSample()}>
                  Preview sample report
                </Button>
              )}
              {showReportActions && (
                <Button variant="outline" onClick={() => setSettingsOpen(true)}>
                  <Settings2 />
                  Settings
                </Button>
              )}
              {showReportActions && !readOnly && (
                <Button onClick={() => void startSync()} disabled={status.running || !settings.ready}>
                  <RefreshCw className={status.running ? "animate-spin" : ""} />
                  {status.running ? "Syncing…" : "Run analysis"}
                </Button>
              )}
            </div>
          )}
        </div>
        <PageHeaderDescription className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <span>
            {summary
              ? `${summary.start} through ${summary.end} · UTC`
              : "Compare AI costs with estimated engineering effort"}
          </span>
          {syncedAt && (
            <span className="text-xs" role="status">
              Last synced {formatSyncedAt(syncedAt)}
            </span>
          )}
        </PageHeaderDescription>
      </PageHeader>
      {sampleSummary && (
        <DemoNotice
          onExit={() => {
            updateDemoUrl(false);
            setSampleSummary(null);
          }}
        />
      )}
      {adminReadOnly && (
        <p className="text-sm text-muted-foreground" role="note">
          Read-only access. Settings, analysis runs, and email matches are unavailable.
        </p>
      )}

      {summary && (
        <Tabs value={view} onValueChange={(value) => setView(value as View)}>
          <PageTabsList aria-label="ROI Calculator views">
            <PageTabsTrigger value="overview">Overview</PageTabsTrigger>
            <PageTabsTrigger value="people">People</PageTabsTrigger>
            <PageTabsTrigger value="branches">Branches</PageTabsTrigger>
          </PageTabsList>
        </Tabs>
      )}

      {!sampleSummary && requestError && (
        <Alert variant="destructive">
          <AlertTitle>ROI Calculator request failed</AlertTitle>
          <AlertDescription>{requestError}</AlertDescription>
        </Alert>
      )}
      {!sampleSummary && status.error && (
        <Alert variant="destructive">
          <AlertTitle>Sync failed</AlertTitle>
          <AlertDescription>{status.error}</AlertDescription>
        </Alert>
      )}
      {summary?.warnings.map((warning) => (
        <Alert key={warning}>
          <AlertTitle>Sync note</AlertTitle>
          <AlertDescription>{warning}</AlertDescription>
        </Alert>
      ))}
      {!sampleSummary && status.running && (
        <Card>
          <CardContent className="flex flex-wrap items-center justify-between gap-4">
            <div
              aria-label="Sync progress"
              aria-valuemax={100}
              aria-valuemin={0}
              aria-valuenow={progress}
              className="min-w-0 flex-1 space-y-2"
              role="progressbar"
            >
              <p className="font-medium">{status.stage}</p>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div className="h-full bg-primary transition-all" style={{ width: `${progress}%` }} />
              </div>
              <p className="text-sm text-muted-foreground">
                {status.done} of {status.total} changes processed · {status.reused} reused
                {` · ${status.elapsed_seconds ?? 0}s elapsed`}
                {status.remaining_seconds != null ? ` · about ${status.remaining_seconds}s remaining` : ""}
              </p>
            </div>
            {!readOnly && (
              <Button variant="outline" onClick={() => void cancelSync()}>
                Cancel sync
              </Button>
            )}
          </CardContent>
        </Card>
      )}

      {!summary && !status.running && !reportError ? (
        <ROISettingsPanel
          accessToken={accessToken}
          initialSettings={settings}
          onboarding={!summary}
          onSaved={setSettings}
          onReset={resetView}
          onStartSync={startSync}
          readOnly={readOnly}
          syncDisabled={status.running}
        />
      ) : null}
      {view === "overview" && summary && (
        <ROIOverview
          summary={summary}
          onSelectPull={setSelectedPull}
          onViewPeople={() => setView("people")}
          onViewBranches={() => setView("branches")}
        />
      )}
      {view === "branches" && summary && (
        <ROIBranches
          summary={summary}
          pulls={filteredPulls}
          query={query}
          onQueryChange={setQuery}
          onSelectPull={setSelectedPull}
        />
      )}
      {view === "people" && summary && (
        <ROIPeopleView
          summary={summary}
          identityMap={settings.identity_map}
          onMatch={(person, login) => setMatchingPerson({ person, login })}
          readOnly={readOnly}
        />
      )}
      <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
        <DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>Calculator settings</DialogTitle>
            <DialogDescription>Connect repositories and choose how to estimate effort.</DialogDescription>
          </DialogHeader>
          <ROISettingsPanel
            accessToken={accessToken}
            initialSettings={settings}
            onboarding={false}
            onSaved={(updated) => {
              if (
                updated.source_provider !== settings.source_provider ||
                updated.github_api_url !== settings.github_api_url ||
                updated.gitlab_api_url !== settings.gitlab_api_url
              ) {
                resetView(updated, false);
                return;
              }
              setSettings(updated);
            }}
            onReset={resetView}
            onStartSync={startSync}
            readOnly={readOnly}
            syncDisabled={status.running}
          />
        </DialogContent>
      </Dialog>
      <PullReasoningDialog pull={selectedPull} summary={summary} onClose={() => setSelectedPull(null)} />
      {!readOnly && (
        <IdentityMatchDialog
          key={matchingPerson?.login.toLowerCase() ?? "closed"}
          selection={matchingPerson}
          identityMap={settings.identity_map}
          gatewayEmails={liveSummary?.people.flatMap((person) => (person.email ? [person.email] : [])) ?? []}
          onClose={() => setMatchingPerson(null)}
          onSave={updateIdentity}
        />
      )}
    </Page>
  );
}
