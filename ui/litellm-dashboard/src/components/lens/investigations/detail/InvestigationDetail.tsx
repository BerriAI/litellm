"use client";
import type { useInvestigationResults } from "../useInvestigationResults";

import { Button } from "@/components/ui/button";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { RunsTab } from "./RunsTab";
import { InvestigationProgress } from "../InvestigationProgress";
import { StepFeed } from "../StepFeed";
import { RunNowDialog } from "./RunNowDialog";
import { useState } from "react";
import { InvestigationSummary } from "./InvestigationSummary";
import { InvestigationFailure } from "./InvestigationFailure";
import { scopeLabel, sourceLabels } from "../../model/format";
import { type Lens } from "../../model/types";
import type { LensWrite } from "../../api/mutations";

import { InvestigationActions } from "./InvestigationActions";
import { RunPicker } from "./RunPicker";
import { FindingsTab } from "./FindingsTab";
import { CriteriaTab } from "./CriteriaTab";
import { HistoryTab } from "./HistoryTab";
export function InvestigationDetail({
  lens,
  readOnly,
  ready,
  busy,
  setEditing,
  setMonitoring,
  update,
  connected,
  results,
  agents = [],
}: {
  lens: Lens;
  readOnly: boolean;
  ready: boolean;
  busy: boolean;
  setEditing: (mode: "new" | "edit" | "duplicate") => void;
  setMonitoring: (open: boolean) => void;
  update: (write: LensWrite) => Promise<unknown>;
  connected: boolean;
  results: ReturnType<typeof useInvestigationResults>;
  agents?: readonly string[];
}) {
  const [runNow, setRunNow] = useState(false);
  const {
    active,
    job,
    tab,
    setTab,
    batchSettings,
    batchId,
    setBatchId,
    setFindingId,
    selectedOutsideHistory,
    history,
    historyError,
    refetchHistory,
    historicalError,
    refetchHistorical,
    missingSnapshot,
    kind,
    setKind,
    batchFindings,
    filter,
    setFilter,
    visibleFindings,
    setRequestOffset,
    setEvidence,
    openBatch,
    historyOffset,
    setHistoryOffset,
  } = results;
  return (
    <div>
      <section className="min-w-0 space-y-5">
        <div className="flex flex-wrap justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">{lens.settings.name}</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              {sourceLabels[lens.settings.source ?? "traces"]} · {scopeLabel(lens.settings)}
            </p>
          </div>
          {!readOnly && (
            <InvestigationActions
              lens={lens}
              ready={ready}
              busy={busy}
              active={active}
              setEditing={setEditing}
              setMonitoring={setMonitoring}
              update={update}
              onRunNow={() => setRunNow(true)}
            />
          )}
        </div>
        <InvestigationSummary lens={lens} connected={connected} />
        {active && (
          <InvestigationProgress
            key={active.id}
            job={active}
            onCancel={
              readOnly
                ? undefined
                : () => {
                    void update((api) => api.cancelRun(lens.id));
                  }
            }
          />
        )}
        {active && <StepFeed job={active} />}
        {runNow && (
          <RunNowDialog
            lens={lens}
            agents={agents}
            busy={busy}
            onClose={() => setRunNow(false)}
            onRun={async (request) => {
              await update((api) => api.startRun(lens.id, request));
              setRunNow(false);
            }}
          />
        )}
        {job?.error && <InvestigationFailure job={job} connected={connected} />}
        <Tabs value={tab} onValueChange={setTab} key={lens.id}>
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b">
            <TabsList variant="line">
              <TabsTrigger value="findings">Findings</TabsTrigger>
              <TabsTrigger value="checks">Criteria</TabsTrigger>
              <TabsTrigger value="runs">
                {
                  { requests: "Requests", both: "Traces & requests", traces: "Traces" }[
                    batchSettings?.source ?? "traces"
                  ]
                }
              </TabsTrigger>
              <TabsTrigger value="activity">History</TabsTrigger>
            </TabsList>
            {tab !== "activity" && (
              <RunPicker
                batchId={batchId}
                setBatchId={setBatchId}
                setFindingId={setFindingId}
                job={job}
                selectedOutsideHistory={selectedOutsideHistory}
                history={history}
                lens={lens}
              />
            )}
          </div>
          {historicalError && (
            <p role="alert" className="text-sm text-destructive">
              Could not load this run.{" "}
              <Button variant="link" onClick={() => void refetchHistorical()}>
                Retry
              </Button>
            </p>
          )}
          {missingSnapshot && tab !== "activity" && (
            <p className="text-sm text-muted-foreground">
              This older batch predates saved result snapshots. Its findings remain available under All accumulated
              findings.
            </p>
          )}
          <FindingsTab
            kind={kind}
            setKind={setKind}
            batchFindings={batchFindings}
            filter={filter}
            setFilter={setFilter}
            visibleFindings={visibleFindings}
            setFindingId={setFindingId}
            active={active}
            lens={lens}
            job={job}
          />
          <CriteriaTab batchSettings={batchSettings} readOnly={readOnly} setEditing={setEditing} />
          <TabsContent value="runs" className="pt-4 space-y-4">
            <RunsTab
              key={job?.id ?? batchId}
              job={job}
              onOpen={(id) => {
                setRequestOffset(0);
                setEvidence({ id, span: "" });
              }}
            />
          </TabsContent>
          <HistoryTab
            history={history}
            historyError={historyError}
            refetchHistory={refetchHistory}
            lens={lens}
            openBatch={openBatch}
            historyOffset={historyOffset}
            setHistoryOffset={setHistoryOffset}
          />
        </Tabs>
      </section>
    </div>
  );
}
