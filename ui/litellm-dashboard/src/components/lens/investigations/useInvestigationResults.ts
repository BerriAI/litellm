"use client";

import { useCallback, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLensDemo } from "../LensDemoContext";
import { useLensApi } from "../api/useLensApi";
import { lensQueries } from "../api/queries";
import { evidenceTarget, mergeFeedback, sortedFindings } from "../model/findings";
import type { Job, Lens } from "../model/types";

export function useInvestigationResults(accessToken: string, lens: Lens | undefined) {
  const apiClient = useLensApi();
  const demo = useLensDemo();
  const [batchId, setBatchId] = useState("latest");
  const [historyOffset, setHistoryOffset] = useState(0);
  const [tab, setTab] = useState("findings");
  const [findingId, setFindingId] = useState<string | null>(null);
  const [filter, setFilter] = useState("open");
  const [kind, setKind] = useState<"issue" | "pattern">("issue");
  const [evidence, setEvidence] = useState<{ id: string; span: string } | null>(null);
  const history = useQuery(
    lensQueries.history(apiClient, accessToken, { lensId: lens?.id, historyOffset, demo: !!demo }),
  );
  const historical = useQuery(lensQueries.run(apiClient, accessToken, lens?.id, batchId));
  const { job, missingSnapshot, selectedOutsideHistory, batchSettings, batchFindings } = runSnapshot(
    lens,
    batchId,
    historical.data,
    history.data,
  );
  const finding = batchFindings.find((f) => f.id === findingId);
  const openBatch = (id: string) => {
    setBatchId(id);
    setTab("findings");
    setFindingId(null);
  };
  const active = lens?.jobs?.find((j) => j.status === "queued" || j.status === "running");
  const visibleFindings = sortedFindings(
    batchFindings.filter((f) => (filter === "all" || f.status === filter) && f.kind === kind),
  );
  const sampledRuns = job?.sample?.executions ?? [];
  const target = evidence ? evidenceTarget(evidence.id) : null;
  const [requestOffset, setRequestOffset] = useState(0);
  const evidenceInput = { lensId: lens?.id, evidenceId: evidence?.id, requestOffset, source: target?.source };
  const requestEvidence = useQuery(lensQueries.evidence(apiClient, accessToken, evidenceInput));
  const reset = useCallback(() => {
    setBatchId("latest");
    setHistoryOffset(0);
    setFindingId(null);
    setTab("findings");
  }, []);

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
