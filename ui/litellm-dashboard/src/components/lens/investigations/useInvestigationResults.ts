"use client";

import { useQueryStates } from "nuqs";
import { useCallback, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLensApi } from "../services";
import { lensQueries } from "../api/queries";
import { evidenceTarget, mergeFeedback, sortedFindings } from "../model/findings";
import type { Job, Lens } from "../model/types";
import { RESULT_PARSERS, RESULT_TABS, type ResultTab } from "../route";

const isResultTab = (tab: string): tab is ResultTab => (RESULT_TABS as readonly string[]).includes(tab);

export function useInvestigationResults(accessToken: string, lens: Lens | undefined) {
  const api = useLensApi(accessToken);
  const [params, setParams] = useQueryStates(RESULT_PARSERS);
  const { run: batchId, section: tab, finding: findingId, finding_status: filter, kind } = params;
  const [historyOffset, setHistoryOffset] = useState(0);
  const setBatchId = useCallback((run: string) => void setParams({ run }), [setParams]);
  const setTab = useCallback((section: string) => void (isResultTab(section) && setParams({ section })), [setParams]);
  const setFindingId = useCallback(
    (finding: string | null) => void setParams({ finding }, { history: "push" }),
    [setParams],
  );
  const setFilter = useCallback((finding_status: string) => void setParams({ finding_status }), [setParams]);
  const setKind = useCallback((next: typeof kind) => void setParams({ kind: next }), [setParams]);
  const evidence = params.evidence === null ? null : { id: params.evidence, span: params.evidence_span ?? "" };
  const setEvidence = useCallback(
    (next: { id: string; span: string } | null) =>
      void setParams({ evidence: next?.id ?? null, evidence_span: next?.span || null }, { history: "push" }),
    [setParams],
  );
  const history = useQuery(lensQueries.history(api, { lensId: lens?.id, historyOffset }));
  const historical = useQuery(lensQueries.run(api, lens?.id, batchId));
  const { job, missingSnapshot, selectedOutsideHistory, batchSettings, batchFindings } = runSnapshot(
    lens,
    batchId,
    historical.data,
    history.data,
  );
  const finding = batchFindings.find((f) => f.id === findingId);
  const openBatch = (id: string) => void setParams({ run: id, section: null, finding: null });
  const active = lens?.jobs?.find((j) => j.status === "queued" || j.status === "running");
  const visibleFindings = sortedFindings(
    batchFindings.filter((f) => (filter === "all" || f.status === filter) && f.kind === kind),
  );
  const sampledRuns = job?.sample?.executions ?? [];
  const target = evidence ? evidenceTarget(evidence.id) : null;
  const [requestOffset, setRequestOffset] = useState(0);
  const evidenceInput = { lensId: lens?.id, evidenceId: evidence?.id, requestOffset, source: target?.source };
  const requestEvidence = useQuery(lensQueries.evidence(api, evidenceInput));
  const reset = useCallback(() => {
    setHistoryOffset(0);
    void setParams({ run: null, finding: null, section: null });
  }, [setParams]);

  return {
    batchId,
    setBatchId,
    historyOffset,
    setHistoryOffset,
    tab,
    setTab,
    findingId,
    setFindingId,
    filter,
    setFilter,
    kind,
    setKind,
    evidence,
    setEvidence,
    history: history.data,
    historyError: history.error,
    refetchHistory: history.refetch,
    historicalError: historical.error,
    refetchHistorical: historical.refetch,
    job,
    missingSnapshot,
    selectedOutsideHistory,
    batchSettings,
    batchFindings,
    finding,
    openBatch,
    active,
    visibleFindings,
    sampledRuns,
    target,
    requestOffset,
    setRequestOffset,
    requestEvidenceData: requestEvidence.data,
    requestEvidenceError: requestEvidence.error,
    requestEvidenceLoading: requestEvidence.isLoading,
    reset,
  };
}

function runSnapshot(lens: Lens | undefined, batchId: string, historical: Job | undefined, history: Job[] | undefined) {
  const job = ["latest", "all"].includes(batchId) ? lens?.jobs?.[0] : historical;
  const missingSnapshot = job?.status === "completed" && job.findings == null && batchId !== "all";
  const selectedOutsideHistory = !["latest", "all"].includes(batchId) && !history?.some((j) => j.id === batchId);
  const batchSettings = job?.settings ?? lens?.settings;
  const batchFindings = mergeFeedback(
    batchId === "all" ? lens?.findings ?? [] : job?.findings ?? [],
    lens?.findings ?? [],
  );
  return { job, missingSnapshot, selectedOutsideHistory, batchSettings, batchFindings };
}
