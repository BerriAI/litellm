"use client";

import { useQueryStates } from "nuqs";
import { useCallback } from "react";
import { FINDING_KINDS, RESULT_PARSERS, RESULT_TABS, type ResultTab } from "../route";

const isResultTab = (tab: string): tab is ResultTab => (RESULT_TABS as readonly string[]).includes(tab);

type FindingKind = (typeof FINDING_KINDS)[number];

export interface EvidenceRef {
  readonly id: string;
  readonly span: string;
}

export function useRunRoute() {
  const [{ run }, setParams] = useQueryStates(RESULT_PARSERS);
  const selectRun = useCallback((id: string) => void setParams({ run: id, finding: null }), [setParams]);
  const openRun = useCallback((id: string) => void setParams({ run: id, section: null, finding: null }), [setParams]);
  return { batchId: run, selectRun, openRun };
}

export function useSectionRoute() {
  const [{ section }, setParams] = useQueryStates(RESULT_PARSERS);
  const setSection = useCallback(
    (next: string) => void (isResultTab(next) && setParams({ section: next })),
    [setParams],
  );
  return { section, setSection };
}

export function useFindingRoute() {
  const [{ finding }, setParams] = useQueryStates(RESULT_PARSERS);
  const setFindingId = useCallback(
    (next: string | null) => void setParams({ finding: next }, { history: "push" }),
    [setParams],
  );
  return { findingId: finding, setFindingId };
}

export function useFindingFilters() {
  const [{ kind, finding_status }, setParams] = useQueryStates(RESULT_PARSERS);
  const setKind = useCallback((next: FindingKind) => void setParams({ kind: next }), [setParams]);
  const setStatus = useCallback((next: string) => void setParams({ finding_status: next }), [setParams]);
  return { kind, setKind, status: finding_status, setStatus };
}

export function useEvidenceRoute() {
  const [{ evidence, evidence_span }, setParams] = useQueryStates(RESULT_PARSERS);
  const setEvidence = useCallback(
    (next: EvidenceRef | null) =>
      void setParams({ evidence: next?.id ?? null, evidence_span: next?.span || null }, { history: "push" }),
    [setParams],
  );
  const current: EvidenceRef | null = evidence === null ? null : { id: evidence, span: evidence_span ?? "" };
  return { evidence: current, setEvidence };
}
