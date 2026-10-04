"use client";

import { useEffect, useId, useState, type ComponentProps } from "react";
import { useQuery } from "@tanstack/react-query";
import { useFormContext, useWatch } from "react-hook-form";
import { lensQueries } from "../api/queries";
import { useLensApi } from "../services";
import { durationLabel } from "../model/format";
import type { ScopeFields } from "./fields/ScopeFields";
import type { MatchingActivityPreview } from "./MatchingActivityPreview";
import type { InvestigationInput } from "./investigationSchema";

type ScopeProps = Omit<ComponentProps<typeof ScopeFields>, "nameField">;
type PreviewProps = Omit<ComponentProps<typeof MatchingActivityPreview>, "onOpen">;

export interface MatchingActivity {
  readonly scope: ScopeProps;
  readonly preview: PreviewProps;
  readonly canReview: boolean;
  readonly manualSelection: boolean;
  readonly selectedRunCount: number;
  clearSelection(): void;
}

type Selection = InvestigationInput["selection"];

function validWindow(selection: Selection): boolean {
  const hours = selection.lookback_hours ?? 24;
  return Number.isInteger(hours) && hours >= 1;
}

function validScope(scope: Selection): boolean {
  const percent = scope.sample_percent ?? 100;
  const cap = scope.sample_size;
  const validCap = cap == null || (Number.isInteger(cap) && cap > 0);
  const validSampling = percent > 0 && percent <= 100 && validCap;
  const validFilters = (scope.filters ?? []).every((f) => f.key.trim() && f.value.trim());
  return validWindow(scope) && validSampling && validFilters;
}

function previewTitle(
  state: { pending: boolean; validWindow: boolean; valid: boolean },
  source: Selection["source"],
  data: PreviewProps["data"],
): string {
  if (state.pending) return "Finding matching activity…";
  if (!state.validWindow) return "Choose a history window between 1 hour and 365 days";
  if (!state.valid) return "Complete your condition to preview matches";
  if (!data) return "Preview unavailable";
  const noun = source === "requests" ? "request" : "run";
  return `${data.eligible} matching ${noun}${data.eligible === 1 ? "" : "s"}`;
}

function manualSelectedCount(selection: Selection): number {
  const sampled = Math.ceil((selection.execution_ids.length * (selection.sample_percent ?? 100)) / 100);
  return Math.min(sampled, selection.sample_size ?? Infinity);
}

function useScopeFieldOptions(api: ReturnType<typeof useLensApi>, selection: Selection, asOf: string) {
  const windowValid = validWindow(selection);
  const discovery = useQuery(lensQueries.discovery(api, { value: selection, asOf, enabled: windowValid }));
  const agents = useQuery(lensQueries.agents(api, asOf, selection.source));
  const runs = discovery.data?.executions ?? [];
  const services = [...new Set(runs.map((r) => r.service).filter(Boolean))].sort();
  const attributes = runs.flatMap((r) => r.metadata ?? []);
  return {
    names: selection.source === "requests" ? services : agents.data ?? [],
    agentsLoading: agents.isFetching,
    agentsError: agents.isError,
    retryAgents: () => void agents.refetch(),
    attributes,
    keys: [...new Set(attributes.map((a) => a.key).filter((key) => !key.startsWith("litellm.")))].sort(),
    id: useId(),
  };
}

/** Live preview of the activity a draft selection matches, debounced so typing a filter does not spam the API. */
export function useMatchingActivity(accessToken: string): MatchingActivity {
  const { control, setValue } = useFormContext<InvestigationInput>();
  const [selection, manualSelection] = useWatch({ control, name: ["selection", "manualSelection"] });
  const api = useLensApi(accessToken);
  const [offset, setOffset] = useState(0);
  const [scope, setScope] = useState(selection);
  const [asOf, setAsOf] = useState(() => new Date().toISOString());
  const serialized = JSON.stringify({ ...selection, execution_ids: [] });
  useEffect(() => {
    const timer = setTimeout(() => {
      setScope(JSON.parse(serialized) as Selection);
      setOffset(0);
      setAsOf(new Date().toISOString());
    }, 350);
    return () => clearTimeout(timer);
  }, [serialized]);
  const windowValid = validWindow(selection);
  const valid = windowValid && validScope(scope);
  const scopeFields = useScopeFieldOptions(api, selection, asOf);
  const previewInput = { scope, offset, asOf, enabled: valid };
  const preview = useQuery(lensQueries.preview(api, previewInput));
  const empty = preview.data?.eligible === 0;
  useEffect(() => {
    if (!empty || !valid) return;
    const timer = window.setTimeout(() => setAsOf(new Date().toISOString()), 15000);
    return () => window.clearTimeout(timer);
  }, [empty, valid, asOf]);
  const refreshPreview = () => {
    setOffset(0);
    setAsOf(new Date().toISOString());
  };
  const pending = serialized !== JSON.stringify(scope) || preview.isFetching;
  const ready = !pending && valid;
  const hasSelection = !manualSelection || !!selection.execution_ids.length;
  const hasMatches = !preview.error && (preview.data?.selected ?? 0) > 0;
  const windowLabel = windowValid
    ? `Last ${durationLabel(selection.lookback_hours ?? 24, "hours")}`
    : "Choose a valid history window";
  const setExecutionIds = (ids: string[]) => setValue("selection.execution_ids", ids, { shouldValidate: true });
  return {
    scope: scopeFields,
    preview: {
      offset,
      onPage: setOffset,
      onSelect: (runId, checked) =>
        setExecutionIds(
          checked ? [...selection.execution_ids, runId] : selection.execution_ids.filter((id) => id !== runId),
        ),
      manualSelection,
      selectedIds: selection.execution_ids,
      selectedCount: manualSelection ? manualSelectedCount(selection) : preview.data?.selected ?? 0,
      title: previewTitle({ pending, validWindow: windowValid, valid }, selection.source, preview.data),
      windowLabel,
      ready,
      error: preview.error,
      data: preview.data,
      onRetry: refreshPreview,
    },
    canReview: ready && hasMatches && hasSelection,
    manualSelection,
    selectedRunCount: selection.execution_ids.length,
    clearSelection: () => setExecutionIds([]),
  };
}
