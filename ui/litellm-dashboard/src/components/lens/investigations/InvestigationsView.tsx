"use client";
import { InvestigationMissing, InvestigationsLoading, InvestigationError } from "./InvestigationStatus";

import { InvestigationNavigation } from "./InvestigationNavigation";
import { useInvestigationResults } from "./useInvestigationResults";

import { useLensUpdate, useSaveLens } from "../api/mutations";
import { lensKeys, lensQueries } from "../api/queries";
import { useLensApi } from "../api/useLensApi";

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
import { MonitoringDialog } from "../setup/MonitoringDialog";
import { InvestigationsWelcome } from "./InvestigationsWelcome";
import { workerConnected, readiness } from "../model/status";
import { type Finding, type Settings } from "../model/types";

export function InvestigationsView({
  accessToken,
  readOnly = false,
  onDemo,
}: {
  accessToken: string;
  readOnly?: boolean;
  onDemo?: () => void;
}) {
  const demo = useLensDemo();
  const apiClient = useLensApi();
  const client = useQueryClient();
  const updateLens = useLensUpdate(accessToken);
  const saveLens = useSaveLens(accessToken);
  const [workerSetup, setWorkerSetup] = useState(false);
  const [monitoring, setMonitoring] = useState(false);
  const now = useNow(2000);
  const query = useQuery(lensQueries.list(apiClient, accessToken, !!demo, workerSetup));
  const models = useQuery(lensQueries.models(apiClient, accessToken));
  const modelDetails = useQuery(lensQueries.modelDetails(apiClient, accessToken));
  const [liveSelected, setLiveSelected] = useQueryState("lens", parseAsString.withOptions({ history: "push" }));
  const [demoSelected, setDemoSelected] = useState<string | null>(null);
  const selected = demo ? demoSelected : liveSelected;
  const setSelected = demo ? setDemoSelected : setLiveSelected;
  const [editing, setEditing] = useState<"new" | "edit" | "duplicate" | null>(null);
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
  const activity = useQuery(lensQueries.activity(apiClient, accessToken, loaded, !!demo));
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
    void client.invalidateQueries({ queryKey: lensKeys.list(accessToken) });
    void client.invalidateQueries({ queryKey: lensKeys.histories() });
  };
  const update = async (path: string, body: unknown, method: "post" | "put" | "patch" = "post") => {
    setBusy(true);
    setError("");
    try {
      await updateLens.mutateAsync({ path, body, method });
      await client.invalidateQueries({ queryKey: lensKeys.list(accessToken) });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not update lens");
    } finally {
      setBusy(false);
    }
  };
  const save = async (settings: Settings) => {
    if (editing === "edit" && (!lens || lens.id !== selected))
      throw new Error("Reopen the investigation to edit its settings");
    if (editing !== "edit" && !ready)
      throw new Error("Wait for recorded activity and a connected worker before starting an investigation");
    const saved = await saveLens.mutateAsync({ id: editing === "edit" ? lens?.id : undefined, settings });
    selectLens(saved.id);
    setEditing(null);
    refresh();
  };
  const changeFinding = async (status: Finding["status"], reason: string) => {
    if (!lens || !finding) return;
    await update(`/lens/${lens.id}/findings/${finding.id}`, { status, reason }, "patch");
  };

  return (
    <section aria-label="Investigations" className="w-full min-w-0 space-y-6">
      {!showEmpty && (
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
      {showReadiness && !ready && <ReadinessBanner activityReady={activityReady} />}
      {!selected && lenses.length > 0 && (
        <InvestigationList lenses={lenses} connected={connected} onSelect={selectLens} />
      )}
      {missingSelection && !lens && <InvestigationMissing selectLens={selectLens} />}
      {lens && (
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
          onClose={() => setEditing(null)}
          onSave={save}
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
        finding={finding}
        sampledRuns={sampledRuns}
        readOnly={readOnly}
        busy={busy}
        onClose={() => setFindingId(null)}
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
