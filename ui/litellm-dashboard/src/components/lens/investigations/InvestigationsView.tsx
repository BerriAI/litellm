"use client";
import { InvestigationMissing, InvestigationsLoading, InvestigationError } from "./InvestigationStatus";

import { InvestigationNavigation } from "./InvestigationNavigation";
import { useInvestigationResults } from "./useInvestigationResults";

import { useLensUpdate, useSaveLens, type LensWrite } from "../api/mutations";
import { lensKeys, lensQueries } from "../api/queries";
import { useLensApi } from "../services";

import { InvestigationDetail } from "./detail/InvestigationDetail";
import { ReadinessBanner } from "./ReadinessBanner";
import { RequestEvidenceSheet } from "./RequestEvidenceSheet";
import { useNow } from "@/hooks/useNow";

import { useLensDemo } from "../LensDemoContext";

import { useEffect, useState } from "react";
import { parseAsString, useQueryState } from "nuqs";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { FindingSheet } from "./FindingSheet";
import { TraceSheet } from "./TraceSheet";
import { InvestigationSetupDialog } from "../setup/InvestigationSetupDialog";
import { WorkerDialog } from "../setup/worker/WorkerDialog";
import { useAnalysisKeyInfo } from "../setup/worker/AnalysisKeyDetails";
import { InvestigationList } from "./InvestigationList";
import { HeaderActions } from "./HeaderActions";
import { RunNowDialog } from "./detail/RunNowDialog";
import { FindingsInbox } from "./FindingsInbox";
import { findingAgents, sampledExecutions, type InboxRow } from "../model/inbox";
import { WatchAllBanner } from "./WatchAllBanner";
import { MonitoringDialog } from "../setup/MonitoringDialog";
import { InvestigationsWelcome } from "./InvestigationsWelcome";
import { workerConnected, readiness } from "../model/status";
import { type Finding, type Settings } from "../model/types";

export function InvestigationsView({
  view = "findings",
  active = true,
  accessToken,
  readOnly = false,
  onDemo,
}: {
  view?: "findings" | "investigations";
  active?: boolean;
  accessToken: string;
  readOnly?: boolean;
  onDemo?: () => void;
}) {
  const demo = useLensDemo();
  const api = useLensApi(accessToken);
  const client = useQueryClient();
  const updateLens = useLensUpdate(accessToken);
  const saveLens = useSaveLens(accessToken);
  const [workerSetup, setWorkerSetup] = useState(false);
  const [monitoring, setMonitoring] = useState(false);
  const now = useNow(2000);
  const query = useQuery(lensQueries.list(api, !!demo, workerSetup));
  const models = useQuery(lensQueries.models(api));
  const modelDetails = useQuery(lensQueries.modelDetails(api));
  const [agentsAsOf] = useState(() => new Date().toISOString());
  const agents = useQuery(lensQueries.agents(api, agentsAsOf, "traces"));
  const [liveSelected, setLiveSelected] = useQueryState("lens", parseAsString.withOptions({ history: "push" }));
  const [demoSelected, setDemoSelected] = useState<string | null>(null);
  const selected = demo ? demoSelected : liveSelected;
  const setSelected = demo ? setDemoSelected : setLiveSelected;
  const [editing, setEditing] = useState<"new" | "edit" | "duplicate" | null>(null);
  const [peek, setPeek] = useState(false);
  const [peeked, setPeeked] = useState<InboxRow | null>(null);
  const [runNowId, setRunNowId] = useState<string | null>(null);
  const [skipped, setSkipped] = useState<readonly { id: string; name: string; reason: string }[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const lenses = [...(query.data?.lenses ?? [])].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
  const showEmpty = !query.isLoading && !query.error && lenses.length === 0;
  const loaded = !query.isLoading && !query.error;
  const showActions = !readOnly && !!query.data && !showEmpty;
  const showReadiness = loaded && !showEmpty && !readOnly;
  const missingSelection = !!selected && loaded;
  const lens = lenses.find((e) => e.id === selected);
  const results = useInvestigationResults(accessToken, lens);
  const {
    finding,
    sampledRuns,
    target,
    evidence,
    requestEvidenceData,
    requestEvidenceError,
    requestEvidenceLoading,
    requestOffset,
    setRequestOffset,
    setEvidence,
    setFindingId,
    reset: resetResults,
  } = results;
  const selectLens = (id: string | null) => {
    void setSelected(id);
    resetResults();
  };
  useEffect(() => {
    const restoreLocation = () => {
      setEditing(null);
      resetResults();
    };
    window.addEventListener("popstate", restoreLocation);
    return () => window.removeEventListener("popstate", restoreLocation);
  }, [resetResults]);
  const connected = query.data?.workers?.some((w) => workerConnected(w, now)) ?? false;
  const activeWorkers = query.data?.workers.filter((worker) => !worker.revoked) ?? [];
  const defaultKeyId = activeWorkers.length === 1 ? activeWorkers[0].analysis_key_id : undefined;
  const analysisAccess = useAnalysisKeyInfo(accessToken, defaultKeyId ?? undefined);
  const defaultModel = analysisAccess.data?.models.length === 1 ? analysisAccess.data.models[0] : undefined;
  const activity = useQuery(lensQueries.activity(api, loaded, !!demo));
  const { tracesReady, requestsReady, activityReady, ready } = readiness(
    activity.data,
    activity.error,
    connected,
    query.error,
  );
  const setupSettings = () => {
    if (editing === "new") return undefined;
    if (editing === "duplicate" && lens)
      return { ...lens.settings, name: `${lens.settings.name} copy`, enabled: false };
    return lens?.settings;
  };
  const refresh = () => {
    void client.invalidateQueries({ queryKey: lensKeys.list(api.scope) });
    void client.invalidateQueries({ queryKey: lensKeys.histories() });
  };
  const update = async (write: LensWrite): Promise<boolean> => {
    setBusy(true);
    setError("");
    try {
      await updateLens.mutateAsync(write);
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not update lens");
      return false;
    } finally {
      await client.invalidateQueries({ queryKey: lensKeys.list(api.scope) });
      setBusy(false);
    }
  };
  const save = async (settings: Settings) => {
    if (editing === "edit" && (!lens || lens.id !== selected))
      throw new Error("Reopen the investigation to edit its settings");
    if (editing !== "edit" && !ready)
      throw new Error("Wait for recorded activity and a connected worker before starting an investigation");
    const saved = await saveLens.mutateAsync({ id: editing === "edit" ? lens?.id : undefined, settings });
    selectLens(peek ? null : saved.id);
    setPeek(false);
    setEditing(null);
    refresh();
  };
  const editFromTable = (id: string) => {
    setPeek(true);
    selectLens(id);
    setEditing("edit");
  };
  const openFinding = (row: InboxRow) => {
    setPeek(true);
    setPeeked(row);
    selectLens(row.sources[0].lens.id);
  };
  const closeFinding = () => {
    setFindingId(null);
    setPeeked(null);
    if (!peek) return;
    setPeek(false);
    selectLens(null);
  };
  const changeFinding = async (status: Finding["status"], reason: string) => {
    if (peeked) {
      const saved = await update((current) =>
        Promise.all(peeked.sources.map((s) => current.reviewFinding(s.lens.id, s.finding.id, status, reason))),
      );
      if (saved) closeFinding();
      return;
    }
    if (!lens || !finding) return;
    await update((current) => current.reviewFinding(lens.id, finding.id, status, reason));
  };
  const sheetFinding = peeked ? peeked.sources[0].finding : finding;
  const sheetRuns = peeked ? peeked.sources.flatMap((s) => sampledExecutions(s.lens)) : sampledRuns;
  const detailAgents = lens && finding ? findingAgents(lens, finding) : [];
  const sheetAgents = peeked ? peeked.agents : detailAgents;

  const onDetail = !!selected && !peek;
  const showDetailNav = onDetail && !showEmpty;
  const showTables = !onDetail && lenses.length > 0;
  const showMissing = missingSelection && !lens && !peek;
  return (
    <section aria-label="Investigations" className="flex w-full min-w-0 flex-1 flex-col gap-3">
      {showDetailNav && (
        <InvestigationNavigation
          lens={lens}
          showActions={showActions}
          activityReady={activityReady}
          connected={connected}
          ready={ready}
          selectLens={selectLens}
          setWorkerSetup={setWorkerSetup}
          setEditing={setEditing}
        />
      )}
      {(error || query.error) && <InvestigationError error={error} queryError={query.error} refresh={refresh} />}
      {query.isLoading && <InvestigationsLoading />}
      {loaded && showEmpty && (
        <InvestigationsWelcome
          tracesReady={tracesReady}
          requestsReady={requestsReady}
          checking={activity.isPending}
          traceError={activity.error?.message}
          connected={connected}
          readOnly={!!readOnly}
          onRetry={() => {
            void activity.refetch();
            refresh();
          }}
          onConnect={() => setWorkerSetup(true)}
          onCreate={() => setEditing("new")}
          onDemo={activity.isSuccess && !ready ? onDemo : undefined}
        />
      )}
      {showReadiness && !ready && <ReadinessBanner activityReady={activityReady} className="py-2 text-xs" />}
      {showTables && (
        <div className="flex min-h-0 flex-1 flex-col gap-2">
          {active && (
            <HeaderActions>
              {!readOnly && (
                <WatchAllBanner
                  lenses={lenses}
                  busy={busy}
                  skipped={skipped}
                  onWatchAll={() =>
                    update(async (api) => {
                      const result = await api.watchAll();
                      setSkipped(result.skipped);
                    })
                  }
                />
              )}
              <InvestigationNavigation
                lens={undefined}
                showActions={showActions}
                activityReady={activityReady}
                connected={connected}
                ready={ready}
                selectLens={selectLens}
                setWorkerSetup={setWorkerSetup}
                setEditing={setEditing}
              />
            </HeaderActions>
          )}
          {view === "findings" ? (
            <FindingsInbox lenses={lenses} onOpen={openFinding} />
          ) : (
            <InvestigationList
              lenses={lenses}
              connected={connected}
              readOnly={readOnly}
              onEdit={editFromTable}
              onRunNow={(id) => setRunNowId(id)}
            />
          )}
        </div>
      )}
      {showMissing && <InvestigationMissing selectLens={selectLens} />}
      {lens && !peek && (
        <InvestigationDetail
          lens={lens}
          readOnly={readOnly}
          ready={ready}
          busy={busy}
          setEditing={setEditing}
          setMonitoring={setMonitoring}
          update={update}
          connected={connected}
          results={results}
          agents={Array.isArray(agents.data) ? agents.data : []}
        />
      )}
      {editing && (
        <InvestigationSetupDialog
          ready={ready}
          mode={editing}
          initial={setupSettings()}
          defaultModel={defaultModel}
          defaultSource={!tracesReady && requestsReady ? "requests" : "traces"}
          models={models.data?.data.map((m) => m.id) ?? []}
          modelDetails={modelDetails.data?.data ?? []}
          modelsLoading={models.isLoading}
          modelsError={models.error?.message}
          accessToken={accessToken}
          onClose={() => {
            setEditing(null);
            if (peek) {
              setPeek(false);
              selectLens(null);
            }
          }}
          onSave={save}
        />
      )}
      {runNowId && (
        <RunNowDialog
          lens={lenses.find((l) => l.id === runNowId) ?? lenses[0]}
          agents={Array.isArray(agents.data) ? agents.data : []}
          busy={busy}
          onClose={() => setRunNowId(null)}
          onRun={async (request) => {
            await update((api) => api.startRun(runNowId, request));
            setRunNowId(null);
          }}
        />
      )}
      {workerSetup && (
        <WorkerDialog
          accessToken={accessToken}
          workers={query.data?.workers ?? []}
          onClose={() => setWorkerSetup(false)}
          onChanged={refresh}
          onReady={
            ready && showEmpty
              ? () => {
                  setWorkerSetup(false);
                  setEditing("new");
                }
              : undefined
          }
        />
      )}
      {monitoring && lens && (
        <MonitoringDialog
          settings={lens.settings}
          ready={ready}
          onClose={() => setMonitoring(false)}
          onSave={async (settings) => {
            await saveLens.mutateAsync({ id: lens.id, settings });
            refresh();
          }}
        />
      )}
      <FindingSheet
        finding={sheetFinding}
        agents={sheetAgents}
        sampledRuns={sheetRuns}
        readOnly={readOnly}
        busy={busy}
        onClose={closeFinding}
        changeFinding={changeFinding}
        onEvidence={(value) => {
          setRequestOffset(0);
          setEvidence(value);
        }}
      />
      {lens && target?.source === "traces" && (
        <TraceSheet
          open={!!evidence}
          traceId={target.id}
          traceRef={target.traceRef}
          initialSpanId={evidence?.span}
          accessToken={accessToken}
          onClose={() => setEvidence(null)}
        />
      )}
      <RequestEvidenceSheet
        target={target}
        setEvidence={setEvidence}
        requestEvidenceData={requestEvidenceData}
        requestEvidenceError={requestEvidenceError}
        requestEvidenceLoading={requestEvidenceLoading}
        requestOffset={requestOffset}
        setRequestOffset={setRequestOffset}
      />
    </section>
  );
}
