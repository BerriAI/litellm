"use client";

import React from "react";
import { BarChart3, RefreshCw } from "lucide-react";

import { apiClient } from "@/components/networking";
import { PageHeader } from "@/components/shared/PageHeader";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { extractErrorMessage } from "@/utils/errorUtils";
import { isProxyAdminTierRole } from "@/utils/roles";
import ROISettingsPanel from "./ROISettingsPanel";
import { IdentityMatchDialog, type PersonMatchSelection, PullReasoningDialog } from "./ROICalculatorDialogs";
import { ROIOverview, ROIPeopleView } from "./ROICalculatorViews";
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

type View = "overview" | "people" | "settings";

const IDLE_STATUS: ROISyncStatus = {
  running: false,
  phase: "idle",
  stage: "Idle",
  done: 0,
  total: 0,
  estimated: 0,
  reused: 0,
  needs_attention: 0,
  error: null,
};

export default function ROICalculatorView({
  accessToken,
  userRole = null,
  isViewOnly = false,
}: {
  accessToken: string | null;
  userRole?: string | null;
  isViewOnly?: boolean;
}) {
  const readOnly = isViewOnly && isProxyAdminTierRole(userRole ?? "");
  const [view, setView] = React.useState<View>("overview");
  const [settings, setSettings] = React.useState<ROISettings | null>(null);
  const [summary, setSummary] = React.useState<ROISummary | null>(null);
  const [status, setStatus] = React.useState<ROISyncStatus>(IDLE_STATUS);
  const [selectedPull, setSelectedPull] = React.useState<ROIPull | null>(null);
  const [matchingPerson, setMatchingPerson] = React.useState<PersonMatchSelection | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [query, setQuery] = React.useState("");

  const loadReport = React.useCallback(async () => {
    if (!accessToken) return null;
    const response: ROIReportResponse = await apiClient.get("/roi-calculator/report", { accessToken });
    return response.report;
  }, [accessToken]);

  React.useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    Promise.all([
      apiClient.get<ROISettings>("/roi-calculator/settings", { accessToken }),
      apiClient.get<ROIReportResponse>("/roi-calculator/report", { accessToken }),
      apiClient.get<ROISyncStatus>("/roi-calculator/sync", { accessToken }),
    ])
      .then(([nextSettings, reportResponse, syncStatus]) => {
        if (cancelled) return;
        setSettings(nextSettings);
        setSummary(reportResponse.report);
        setStatus(syncStatus);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(extractErrorMessage(reason));
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken]);

  React.useEffect(() => {
    if (!accessToken || !status.running) return;
    let cancelled = false;
    let requestInFlight = false;
    const interval = window.setInterval(() => {
      if (requestInFlight) return;
      requestInFlight = true;
      apiClient
        .get<ROISyncStatus>("/roi-calculator/sync", { accessToken })
        .then(async (nextStatus) => {
          if (cancelled) return;
          setError(null);
          if (!nextStatus.running && nextStatus.phase === "complete") {
            try {
              const report = await loadReport();
              if (cancelled) return;
              setSummary(report);
              setError(null);
              if (view === "settings") setView("overview");
            } catch (reason) {
              if (!cancelled) setError(extractErrorMessage(reason));
            }
          }
          if (!cancelled) setStatus(nextStatus);
        })
        .catch((reason: unknown) => {
          if (!cancelled) setError(extractErrorMessage(reason));
        })
        .finally(() => {
          requestInFlight = false;
        });
    }, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [accessToken, loadReport, status.running, view]);

  const startSync = React.useCallback(async () => {
    if (!accessToken || readOnly) return;
    try {
      setError(null);
      setStatus(await apiClient.post<ROISyncStatus>("/roi-calculator/sync", { accessToken }));
    } catch (reason) {
      setError(extractErrorMessage(reason));
    }
  }, [accessToken, readOnly]);

  const cancelSync = React.useCallback(async () => {
    if (!accessToken || readOnly) return;
    try {
      setStatus(await apiClient.delete<ROISyncStatus>("/roi-calculator/sync", { accessToken }));
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
      setSettings((current) => (current ? { ...current, identity_map: response.identity_map } : current));
    },
    [accessToken, readOnly],
  );

  const filteredPulls = React.useMemo(() => (summary ? filterPulls(summary.pulls, query) : []), [query, summary]);

  if (error && !settings) {
    return (
      <div className="p-8">
        <Alert variant="destructive">
          <AlertTitle>Could not load ROI Calculator</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!settings) {
    return (
      <div className="space-y-6 p-8">
        <Skeleton className="h-16 w-96" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  const progress = status.total > 0 ? Math.min(100, (status.done / status.total) * 100) : 0;
  const statusIsIdleOrComplete = status.phase === "idle" || status.phase === "complete";
  const syncIsUpToDate = !status.running && statusIsIdleOrComplete;
  const syncedAt = syncIsUpToDate ? summary?.synced_at : null;

  return (
    <main className="w-full space-y-6 p-8">
      <PageHeader
        icon={<BarChart3 />}
        title="ROI Calculator"
        subtitle={
          <>
            {summary
              ? `${summary.start} through ${summary.end} · UTC`
              : "Compare gateway spend with estimated engineering effort for merged pull requests"}
            {syncedAt && (
              <span className="mt-1 block text-xs text-muted-foreground" role="status">
                Up to date · Last synced {formatSyncedAt(syncedAt)}
                {!status.running && status.phase === "complete" && status.reused > 0
                  ? ` · ${status.reused} of ${status.total} estimates reused`
                  : ""}
              </span>
            )}
          </>
        }
      />
      {readOnly && (
        <p className="text-sm text-muted-foreground" role="note">
          Read-only access. Settings, analysis runs, and email matches are unavailable.
        </p>
      )}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <Tabs value={view} onValueChange={(value) => setView(value as View)}>
          <TabsList aria-label="ROI Calculator views">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="people">People</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>
        </Tabs>
        {view !== "settings" && !readOnly && (
          <Button onClick={() => void startSync()} disabled={status.running || !settings.ready}>
            <RefreshCw className={status.running ? "animate-spin" : ""} />
            {status.running ? "Syncing…" : "Run analysis"}
          </Button>
        )}
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>ROI Calculator request failed</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {status.error && (
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
      {status.running && (
        <Card>
          <CardContent className="flex flex-wrap items-center justify-between gap-4 pt-6">
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
                {status.done} of {status.total} pull requests processed · {status.reused} reused
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

      {view === "settings" || (!summary && !status.running) ? (
        <ROISettingsPanel
          key={JSON.stringify(settings)}
          accessToken={accessToken}
          initialSettings={settings}
          onboarding={!summary}
          onSaved={setSettings}
          onStartSync={startSync}
          readOnly={readOnly}
          syncDisabled={status.running}
        />
      ) : null}
      {view === "overview" && summary && (
        <ROIOverview
          summary={summary}
          pulls={filteredPulls}
          query={query}
          onQueryChange={setQuery}
          onSelectPull={setSelectedPull}
          onViewPeople={() => setView("people")}
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
      <PullReasoningDialog pull={selectedPull} summary={summary} onClose={() => setSelectedPull(null)} />
      {!readOnly && (
        <IdentityMatchDialog
          key={matchingPerson?.login.toLowerCase() ?? "closed"}
          selection={matchingPerson}
          identityMap={settings.identity_map}
          onClose={() => setMatchingPerson(null)}
          onSave={updateIdentity}
        />
      )}
    </main>
  );
}
