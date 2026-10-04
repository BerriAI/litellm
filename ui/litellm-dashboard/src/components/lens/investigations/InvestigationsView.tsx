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

import { useDialogRoute, useIssueRoute, useLensRoute } from "../route";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { FindingSheet } from "./FindingSheet";
import { TraceSheet } from "./TraceSheet";
import { InvestigationSetupDialog } from "../setup/InvestigationSetupDialog";
import { WorkerDialog } from "../setup/worker/WorkerDialog";
import { useAnalysisKeyInfo } from "../setup/worker/AnalysisKeyDetails";
import { InvestigationList } from "./InvestigationList";
import { HeaderActions } from "../HeaderActions";
import { RunNowDialog } from "./detail/RunNowDialog";
import { findFinding, findingAgents, findingKey, sampledExecutions } from "../model/inbox";
import { WatchAllBanner } from "./WatchAllBanner";
import { MonitoringDialog } from "../setup/MonitoringDialog";
import { InvestigationsWelcome } from "./InvestigationsWelcome";
import { workerConnected, readiness } from "../model/status";
import { type Finding, type Lens, type Settings } from "../model/types";

export function InvestigationsView({
  active = true,
  accessToken,
  readOnly = false,
}: {
  active?: boolean;
  accessToken: string;
  readOnly?: boolean;
}) {
  const api = useLensApi(accessToken);
  const client = useQueryClient();
  const updateLens = useLensUpdate(accessToken);
  const saveLens = useSaveLens(accessToken);
  const { dialog, target: dialogTarget, openDialog, closeDialog } = useDialogRoute();
  const { issueKey, setIssueKey } = useIssueRoute();
  const workerSetup = dialog === "workers";
  const now = useNow(2000);
  const query = useQuery(lensQueries.list(api, workerSetup));
  const models = useQuery(lensQueries.models(api));
  const modelDetails = useQuery(lensQueries.modelDetails(api));
  const [agentsAsOf] = useState(() => new Date().toISOString());
  const agents = useQuery(lensQueries.agents(api, agentsAsOf, "traces"));
  const { lensId: selected, setLensId: setSelected } = useLensRoute();
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
  const targetLens = dialogTarget ? lenses.find((e) => e.id === dialogTarget) : lens;
  const editing = dialog === "new" || dialog === "edit" || dialog === "duplicate" ? dialog : null;
  const setupMode = editing === "new" || targetLens ? editing : null;
  const peeked = issueKey ? findFinding(lenses, issueKey) : undefined;
  const resultsLens = peeked?.lens ?? lens;
  const results = useInvestigationResults(accessToken, resultsLens);
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
    setSelected(id);
    resetResults();
  };
  const setEditing = (mode: "new" | "edit" | "duplicate") => openDialog(mode);
  const setWorkerSetup = (open: boolean) => (open ? openDialog("workers") : closeDialog());
  const setMonitoring = (open: boolean) => (open ? openDialog("monitoring") : closeDialog());
  const connected = query.data?.workers?.some((w) => workerConnected(w, now)) ?? false;
  const activeWorkers = query.data?.workers.filter((worker) => !worker.revoked) ?? [];
  const defaultKeyId = activeWorkers.length === 1 ? activeWorkers[0].analysis_key_id : undefined;
  const analysisAccess = useAnalysisKeyInfo(accessToken, defaultKeyId ?? undefined);
  const defaultModel = analysisAccess.data?.models.length === 1 ? analysisAccess.data.models[0] : undefined;
  const activity = useQuery(lensQueries.activity(api, loaded));
  const { tracesReady, requestsReady, activityReady, ready } = readiness(
    activity.data,
    activity.error,
    connected,
    query.error,
  );
  const setupSettings = () => {
    if (editing === "new") return undefined;
    if (editing === "duplicate" && targetLens)
      return { ...targetLens.settings, name: `${targetLens.settings.name} copy`, enabled: false };
    return targetLens?.settings;
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
    if (editing === "edit" && !targetLens) throw new Error("Reopen the investigation to edit its settings");
    if (editing !== "edit" && !ready)
      throw new Error("Wait for recorded activity and a connected worker before starting an investigation");
    const saved = await saveLens.mutateAsync({ id: editing === "edit" ? targetLens?.id : undefined, settings });
    closeDialog();
    if (!dialogTarget) selectLens(saved.id);
    refresh();
  };
  const openFinding = (owner: Lens, picked: Finding) => setIssueKey(findingKey(owner, picked));
  const closeFinding = () => {
    setFindingId(null);
    setIssueKey(null);
  };
  const changeFinding = async (status: Finding["status"], reason: string) => {
    if (peeked) {
      const saved = await update((current) => current.reviewFinding(peeked.lens.id, peeked.finding.id, status, reason));
      if (saved) closeFinding();
      return;
    }
    if (!lens || !finding) return;
    await update((current) => current.reviewFinding(lens.id, finding.id, status, reason));
  };
  const sheetFinding = peeked ? peeked.finding : finding;
  const sheetRuns = peeked ? sampledExecutions(peeked.lens) : sampledRuns;
  const detailAgents = lens && finding ? findingAgents(lens, finding) : [];
  const sheetAgents = peeked ? findingAgents(peeked.lens, peeked.finding) : detailAgents;

  const onDetail = !!selected;
  const showDetailNav = onDetail && !showEmpty;
  const showTables = !onDetail && lenses.length > 0;
  const showMissing = missingSelection && !lens;
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
          showPreview={activity.isSuccess && !ready}
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
          <InvestigationList
            lenses={lenses}
            connected={connected}
            readOnly={readOnly}
            onEdit={(id) => openDialog("edit", id)}
            onRunNow={(id) => openDialog("run_now", id)}
            onOpenFinding={openFinding}
          />
        </div>
      )}
      {showMissing && <InvestigationMissing selectLens={selectLens} />}
      {lens && (
        <InvestigationDetail
          lens={lens}
          readOnly={readOnly}
          ready={ready}
          busy={busy}
          setEditing={setEditing}
          setMonitoring={setMonitoring}
          onRunNow={() => openDialog("run_now")}
          update={update}
          connected={connected}
          results={results}
        />
      )}
      {setupMode && (
        <InvestigationSetupDialog
          ready={ready}
          mode={setupMode}
          initial={setupSettings()}
          defaultModel={defaultModel}
          defaultSource={!tracesReady && requestsReady ? "requests" : "traces"}
          models={models.data?.data.map((m) => m.id) ?? []}
          modelDetails={modelDetails.data?.data ?? []}
          modelsLoading={models.isLoading}
          modelsError={models.error?.message}
          accessToken={accessToken}
          onClose={closeDialog}
          onSave={save}
        />
      )}
      {dialog === "run_now" && targetLens && (
        <RunNowDialog
          lens={targetLens}
          agents={Array.isArray(agents.data) ? agents.data : []}
          busy={busy}
          onClose={closeDialog}
          onRun={async (request) => {
            await update((api) => api.startRun(targetLens.id, request));
            closeDialog();
          }}
        />
      )}
      {workerSetup && (
        <WorkerDialog
          accessToken={accessToken}
          workers={query.data?.workers ?? []}
          onClose={closeDialog}
          onChanged={refresh}
          onReady={ready && showEmpty ? () => openDialog("new") : undefined}
        />
      )}
      {dialog === "monitoring" && lens && (
        <MonitoringDialog
          settings={lens.settings}
          ready={ready}
          onClose={closeDialog}
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
      {resultsLens && target?.source === "traces" && (
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
