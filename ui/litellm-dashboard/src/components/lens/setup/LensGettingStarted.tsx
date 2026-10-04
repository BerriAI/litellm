"use client";

import { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { TraceSummary } from "@/components/view_logs/TraceView/traceTypes";
import { lensQueries } from "../api/queries";
import { useSaveLens } from "../api/mutations";
import { useLensApi } from "../services";
import { TraceSheet } from "../investigations/TraceSheet";
import { useAnalysisKeyInfo } from "./worker/AnalysisKeyDetails";
import { WorkerDialog } from "./worker/WorkerDialog";
import { InvestigationSetupDialog } from "./InvestigationSetupDialog";
import { LensIntroduction } from "./LensIntroduction";
import type { LensSetupState } from "./useLensSetup";
import type { Settings } from "../model/types";
import { initialSetupStep, LensSetupSteps } from "./LensSetupSteps";

export function LensGettingStarted({
  accessToken,
  state,
  readOnly,
  canInvestigate,
  canMintTracingKey,
  onStart,
  onExit,
  onCreated,
  onDemo,
}: {
  accessToken: string;
  state: LensSetupState;
  readOnly: boolean;
  canInvestigate: boolean;
  canMintTracingKey: boolean;
  onStart: () => void;
  onExit: () => void;
  onCreated: (id: string) => void;
  onDemo?: () => void;
}) {
  const setupRef = useRef<HTMLElement>(null);
  const [step, setStep] = useState(() => initialSetupStep(state));
  const [workerOpen, setWorkerOpen] = useState(false);
  const [investigationOpen, setInvestigationOpen] = useState(false);
  const [trace, setTrace] = useState<TraceSummary | null>(null);
  const selectStep = (index: number) => {
    onStart();
    setStep(index);
    setupRef.current?.querySelector<HTMLButtonElement>(`[aria-controls="lens-setup-step-${index}"]`)?.focus();
  };
  const canLeave = state.tracesReady || state.requestsReady || state.hasInvestigations;
  const start = () => {
    onStart();
    setupRef.current?.scrollIntoView({ block: "start" });
    setupRef.current?.focus({ preventScroll: true });
  };
  const connectWorker = () => {
    selectStep(2);
    setWorkerOpen(true);
  };
  const createInvestigation = () => {
    selectStep(3);
    setInvestigationOpen(true);
  };

  return (
    <div className="mx-auto w-full max-w-7xl space-y-8 py-3 sm:space-y-10 sm:py-4">
      <LensIntroduction onStart={start} onDemo={onDemo} />
      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_280px] xl:gap-8">
        <section
          ref={setupRef}
          tabIndex={-1}
          aria-labelledby="lens-setup-title"
          className="min-w-0 scroll-mt-4 outline-none"
          onFocusCapture={onStart}
        >
          <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 id="lens-setup-title" className="text-2xl font-semibold tracking-tight">
                Get Lens running
              </h2>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">
                Each step checks your connection, so you’ll see when it’s working.
              </p>
            </div>
            {canLeave && (
              <Button variant="outline" size="sm" onClick={onExit}>
                {state.tracesReady ? "View traces" : "View investigations"}
              </Button>
            )}
          </div>
          <LensSetupSteps
            step={step}
            state={state}
            accessToken={accessToken}
            readOnly={readOnly}
            canInvestigate={canInvestigate}
            canMintTracingKey={canMintTracingKey}
            onStep={selectStep}
            onConnect={connectWorker}
            onCreate={createInvestigation}
            onTrace={setTrace}
          />
          {state.error && (
            <div role="alert" className="mt-4 flex flex-wrap items-center gap-3 text-sm text-destructive">
              <p>Could not check setup. {state.error}</p>
              <Button variant="outline" size="sm" onClick={state.refresh} disabled={state.checking}>
                Retry
              </Button>
            </div>
          )}
          {(readOnly || !canInvestigate) && (
            <p className="mt-4 text-sm text-muted-foreground">
              A gateway administrator can connect a worker and run investigations.
            </p>
          )}
        </section>
        <aside className="rounded-2xl border bg-card p-6" aria-labelledby="lens-prerequisites">
          <h2 id="lens-prerequisites" className="text-base font-semibold">
            Before you start
          </h2>
          <ul className="mt-5 space-y-5 text-sm">
            {[
              { title: "LiteLLM gateway", detail: "Access to its configuration" },
              { title: "ClickHouse", detail: "Self-hosted or managed trace storage" },
              { title: "A server with Docker", detail: "To run the analysis worker" },
              { title: "An analysis model", detail: "Available through your gateway" },
            ].map((item) => (
              <li key={item.title} className="flex gap-3">
                <Check aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                <div>
                  <p className="font-medium">{item.title}</p>
                  <p className="mt-1 leading-5 text-muted-foreground">{item.detail}</p>
                </div>
              </li>
            ))}
          </ul>
          <div className="mt-5 flex gap-2 border-t pt-5 text-xs leading-5 text-muted-foreground">
            <ShieldCheck aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
            <p>
              Your infrastructure stores the traces. Investigation content is sent to your selected model provider
              through the gateway.
            </p>
          </div>
        </aside>
      </div>
      {workerOpen && (
        <WorkerDialog
          accessToken={accessToken}
          workers={state.workers}
          onClose={() => setWorkerOpen(false)}
          onChanged={state.refresh}
          onReady={
            state.ready
              ? () => {
                  setWorkerOpen(false);
                  createInvestigation();
                }
              : undefined
          }
        />
      )}
      {investigationOpen && (
        <FirstInvestigation
          accessToken={accessToken}
          state={state}
          onClose={() => setInvestigationOpen(false)}
          onCreated={onCreated}
        />
      )}
      {trace && (
        <TraceSheet
          open
          traceId={trace.trace_id}
          traceRef={trace.trace_ref}
          accessToken={accessToken}
          onClose={() => setTrace(null)}
        />
      )}
    </div>
  );
}

function FirstInvestigation({
  accessToken,
  state,
  onClose,
  onCreated,
}: {
  accessToken: string;
  state: LensSetupState;
  onClose: () => void;
  onCreated: (id: string) => void;
}) {
  const api = useLensApi(accessToken);
  const saveLens = useSaveLens(accessToken);
  const models = useQuery({ ...lensQueries.models(api) });
  const modelDetails = useQuery({ ...lensQueries.modelDetails(api) });
  const workers = state.workers.filter((worker) => !worker.revoked);
  const analysisAccess = useAnalysisKeyInfo(
    accessToken,
    workers.length === 1 ? workers[0].analysis_key_id ?? undefined : undefined,
  );
  const defaultModel = analysisAccess.data?.models.length === 1 ? analysisAccess.data.models[0] : undefined;
  const save = async (settings: Settings) => {
    if (!state.ready)
      throw new Error("Wait for recorded activity and a connected worker before starting an investigation");
    const saved = await saveLens.mutateAsync({ settings });
    state.refresh();
    onCreated(saved.id);
  };

  return (
    <InvestigationSetupDialog
      accessToken={accessToken}
      ready={state.ready}
      defaultSource={!state.tracesReady && state.requestsReady ? "requests" : "traces"}
      defaultModel={defaultModel}
      models={models.data?.data.map((model) => model.id) ?? []}
      modelDetails={modelDetails.data?.data ?? []}
      modelsLoading={models.isLoading}
      modelsError={models.error?.message}
      onClose={onClose}
      onSave={save}
    />
  );
}
