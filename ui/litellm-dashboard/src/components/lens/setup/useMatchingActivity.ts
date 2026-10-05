"use client";

import { useCallback, useEffect, useState } from "react";
import { useInfiniteQuery, useQuery, useQueryClient } from "@tanstack/react-query";
import { useFormContext, useWatch } from "react-hook-form";
import { lensKeys, lensQueries } from "../data/queries";
import { useLensApi } from "../data/LensServices";
import { durationLabel } from "../model/format";
import type { Sample } from "../model/types";
import { useDebouncedValue } from "./useDebouncedValue";
import type { InvestigationInput } from "./investigationSchema";

const PREVIEW_DEBOUNCE_MS = 350;
const EMPTY_PREVIEW_POLL_MS = 15000;

type Selection = InvestigationInput["selection"];
type PreviewPageData = Pick<Sample, "eligible" | "selected">;

export type Execution = Sample["executions"][number];
export type Attribute = NonNullable<Execution["metadata"]>[number];

export interface ScopeOptions {
  readonly names: readonly string[];
  readonly agentsLoading: boolean;
  readonly agentsError: boolean;
  readonly retryAgents: () => void;
  readonly attributes: readonly Attribute[];
  readonly keys: readonly string[];
}

export interface PreviewStatus {
  readonly title: string;
  readonly windowLabel: string;
  readonly ready: boolean;
  readonly error: Error | null;
  readonly refresh: () => void;
}

export interface PreviewPage {
  readonly eligible: number | undefined;
  readonly selected: number;
  readonly executions: readonly Execution[];
  readonly hasMore: boolean;
  readonly loadingMore: boolean;
  readonly loadMore: () => void;
}

/** Present only while the user picks individual runs; `count` applies the sampling settings to the picks. */
export interface PreviewSelection {
  readonly ids: readonly string[];
  readonly count: number;
  readonly toggle: (id: string, checked: boolean) => void;
  readonly clear: () => void;
}

export interface MatchingPreview {
  readonly status: PreviewStatus;
  readonly page: PreviewPage;
  readonly selection: PreviewSelection | null;
}

export interface MatchingActivity {
  readonly scope: ScopeOptions;
  readonly preview: MatchingPreview;
  readonly canReview: boolean;
}

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
  data: PreviewPageData | undefined,
): string {
  if (!state.validWindow) return "Choose a history window between 1 hour and 365 days";
  if (!state.valid) return "Complete your condition to preview matches";
  if (state.pending) return "Finding matching activity…";
  if (!data) return "Preview unavailable";
  const noun = source === "requests" ? "request" : "run";
  return `${data.eligible} matching ${noun}${data.eligible === 1 ? "" : "s"}`;
}

function manualSelectedCount(selection: Selection): number {
  const sampled = Math.ceil((selection.execution_ids.length * (selection.sample_percent ?? 100)) / 100);
  return Math.min(sampled, selection.sample_size ?? Infinity);
}

function manualPicks(selection: Selection, setExecutionIds: (ids: readonly string[]) => void): PreviewSelection {
  const ids = selection.execution_ids;
  return {
    ids,
    count: manualSelectedCount(selection),
    toggle: (id, checked) => setExecutionIds(checked ? [...ids, id] : ids.filter((other) => other !== id)),
    clear: () => setExecutionIds([]),
  };
}

function windowLabel(selection: Selection): string {
  if (!validWindow(selection)) return "Choose a valid history window";
  return `Last ${durationLabel(selection.lookback_hours ?? 24, "hours")}`;
}

function useScopeFieldOptions(api: ReturnType<typeof useLensApi>, selection: Selection): ScopeOptions {
  const discovery = useQuery(lensQueries.discovery(api, { value: selection, enabled: validWindow(selection) }));
  const agents = useQuery(lensQueries.agents(api, selection.source));
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
  };
}

/** Live preview of the activity a draft selection matches, debounced so typing a filter does not spam the API. */
export function useMatchingActivity(): MatchingActivity {
  const { control, setValue } = useFormContext<InvestigationInput>();
  const [selection, manualSelection] = useWatch({ control, name: ["selection", "manualSelection"] });
  const api = useLensApi();
  const client = useQueryClient();
  const {
    value: scope,
    pending: settling,
    settledAt,
  } = useDebouncedValue<Selection>({ ...selection, execution_ids: [] }, PREVIEW_DEBOUNCE_MS);
  const [refreshedAt, setRefreshedAt] = useState(settledAt);
  const asOf = settledAt > refreshedAt ? settledAt : refreshedAt;
  const windowValid = validWindow(selection);
  const valid = windowValid && validScope(scope);
  const scopeFields = useScopeFieldOptions(api, selection);
  const preview = useInfiniteQuery(lensQueries.preview(api, { scope, asOf, enabled: valid }));
  const firstPage = preview.data?.pages[0];
  const executions = preview.data?.pages.flatMap((page) => page.executions) ?? [];
  const empty = firstPage?.eligible === 0;
  const refresh = useCallback(() => {
    setRefreshedAt(new Date().toISOString());
    void client.invalidateQueries({ queryKey: lensKeys.discoveries() });
    void client.invalidateQueries({ queryKey: lensKeys.agents(api.scope) });
  }, [api.scope, client]);
  useEffect(() => {
    if (!empty || !valid) return;
    const timer = window.setTimeout(refresh, EMPTY_PREVIEW_POLL_MS);
    return () => window.clearTimeout(timer);
  }, [empty, valid, asOf, refresh]);
  const pending = settling || preview.isLoading || preview.isPlaceholderData;
  const ready = !pending && valid;
  const setExecutionIds = (next: readonly string[]) =>
    setValue("selection.execution_ids", [...next], { shouldValidate: true });
  const picked = manualSelection ? manualPicks(selection, setExecutionIds) : null;
  const hasMatches = !preview.error && (firstPage?.selected ?? 0) > 0;
  const hasSelection = !picked || picked.ids.length > 0;
  return {
    scope: scopeFields,
    preview: {
      status: {
        title: previewTitle({ pending, validWindow: windowValid, valid }, selection.source, firstPage),
        windowLabel: windowLabel(selection),
        ready,
        error: preview.error,
        refresh,
      },
      page: {
        eligible: firstPage?.eligible,
        selected: firstPage?.selected ?? 0,
        executions,
        hasMore: preview.hasNextPage,
        loadingMore: preview.isFetchingNextPage,
        loadMore: () => {
          if (preview.hasNextPage && !preview.isFetching) void preview.fetchNextPage({ cancelRefetch: false });
        },
      },
      selection: picked,
    },
    canReview: ready && hasMatches && hasSelection,
  };
}
