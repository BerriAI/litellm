"use client";

import { useEffect, useId, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Plus, X, ArrowUpRight } from "lucide-react";
import { apiClient } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { TracePanel } from "./TracePanel";
import { type Sample, type Settings, runTime, durationLabel } from "./engineData";

import { DurationInput } from "./DurationInput";

export type ActivitySelection = Pick<Settings, "source"> &
  Partial<
    Pick<
      Settings,
      "service" | "filters" | "lookback_hours" | "sample_percent" | "sample_size" | "team_id" | "execution_ids"
    >
  >;

const selectClass = "h-9 w-full rounded-md border border-input bg-background px-3 text-sm";

export function RunList({ executions }: { executions: Sample["executions"] }) {
  return (
    <div className="divide-y">
      {executions.map((run) => (
        <div key={run.id} className="py-3">
          <p className="text-sm font-medium">{run.name}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {runTime(run.start_time)} · {run.source === "traces" ? `${run.span_count} steps` : "LLM request"}
          </p>
          <p className="mt-1 truncate font-mono text-xs text-muted-foreground" title={run.trace_id}>
            {run.trace_id}
          </p>
        </div>
      ))}
    </div>
  );
}

export function ActivityScope({
  value,
  onChange,
  accessToken,
}: {
  value: ActivitySelection;
  onChange: (selection: ActivitySelection) => void;
  accessToken: string;
}) {
  const id = useId();
  const [offset, setOffset] = useState(0);
  const [scope, setScope] = useState(value);
  const [trace, setTrace] = useState<{ id: string; ref?: string } | null>(null);
  const [asOf, setAsOf] = useState(() => new Date().toISOString());
  const serialized = JSON.stringify({ ...value, execution_ids: [] });
  useEffect(() => {
    const timer = setTimeout(() => {
      setScope(JSON.parse(serialized) as ActivitySelection);
      setOffset(0);
      setAsOf(new Date().toISOString());
    }, 350);
    return () => clearTimeout(timer);
  }, [serialized]);
  const historyHours = value.lookback_hours ?? 24;
  const validWindow = Number.isInteger(historyHours) && historyHours >= 1 && historyHours <= 720;
  const percent = scope.sample_percent ?? 100;
  const cap = scope.sample_size;
  const validCap = cap == null || (Number.isInteger(cap) && cap > 0);
  const validSampling = percent > 0 && percent <= 100 && validCap;
  const validFilters = (scope.filters ?? []).every((f) => f.key.trim() && f.value.trim());
  const valid = validWindow && validSampling && validFilters;
  const load = (selection: ActivitySelection, pageOffset = 0) => {
    const { lookback_hours, ...selectionSettings } = selection;
    return apiClient.post<Sample>("/engine/preview/sample", {
      accessToken,
      body: {
        offset: pageOffset,
        as_of: asOf,
        settings: {
          ...selectionSettings,
          execution_ids: [],
          name: "Preview",
          model: "preview",

          checks: [{ id: "preview", instruction: "Preview recorded activity" }],
        },
        lookback_hours: lookback_hours ?? 24,
      },
    });
  };
  const discoveryScope: ActivitySelection = {
    source: value.source,
    service: "",
    filters: [],
    lookback_hours: value.lookback_hours,
  };
  const discoveryOptions = {
    queryKey: ["lens-activity-options", value.source, value.lookback_hours, accessToken],
    queryFn: () => load(discoveryScope),
    staleTime: 60000,
    enabled: validWindow,
  };
  const discovery = useQuery(discoveryOptions);
  const previewOptions = {
    queryKey: ["lens-activity-preview", scope, offset, asOf, accessToken],
    queryFn: () => load(scope, offset),
    enabled: valid,
    staleTime: 30000,
  };
  const preview = useQuery(previewOptions);
  const runs = discovery.data?.executions ?? [];
  const services = [...new Set(runs.map((r) => r.service).filter(Boolean))].sort();
  const attributes = runs.flatMap((r) => r.metadata ?? []);
  const keys = [...new Set(attributes.map((a) => a.key).filter((key) => !key.startsWith("litellm.")))].sort();
  const pending = serialized !== JSON.stringify(scope) || preview.isFetching;
  const ready = !pending && valid;
  const filters = value.filters ?? [];
  const edit = (index: number, field: "key" | "value", text: string) =>
    onChange({ ...value, filters: filters.map((f, i) => (i === index ? { ...f, [field]: text } : f)) });

  const changeSource = (source: Settings["source"]) => {
    const selection = { ...value, source, service: "", filters: [], execution_ids: [] };
    onChange(selection);
  };
  const windowLabel = validWindow
    ? `Last ${durationLabel(value.lookback_hours ?? 24, "hours")}`
    : "Choose a valid history window";
  const previewTitle = () => {
    if (pending) return "Finding matching activity…";
    if (!validWindow) return "Choose a history window between 1 and 720 hours";
    if (!valid) return "Complete your condition to preview matches";
    if (!preview.data) return "Preview unavailable";
    return `${preview.data.eligible} matching ${value.source === "requests" ? "requests" : "runs"}`;
  };
  return (
    <div className="grid gap-5 sm:grid-cols-2">
      <div className="space-y-4">
        <label className="grid gap-2 text-sm">
          Activity type
          <select
            className={selectClass}
            value={value.source}
            onChange={(e) => changeSource(e.target.value as Settings["source"])}
          >
            <option value="traces">Agent runs</option>
            <option value="requests">Individual LLM requests</option>
            <option value="both">Agent runs and LLM requests</option>
          </select>
        </label>
        <p className="text-xs text-muted-foreground">
          {value.source === "requests"
            ? "Each request is one model call, not an entire agent run."
            : "An agent run contains the steps recorded under one trace ID. Separate sessions are not joined automatically."}
        </p>
        <label className="grid gap-2 text-sm">
          {
            {
              requests: "Model group (optional)",
              traces: "Application (optional)",
              both: "Application or model group (optional)",
            }[value.source ?? "traces"]
          }
          <Input
            list={`${id}-services`}
            value={value.service}
            placeholder="All activity"
            onChange={(e) => onChange({ ...value, service: e.target.value })}
          />
          <datalist id={`${id}-services`}>
            {services.map((s) => (
              <option key={s} value={s} />
            ))}
          </datalist>
        </label>
        <p className="text-xs text-muted-foreground">
          {
            {
              requests: "The model alias configured on your LiteLLM gateway. Leave blank for all models.",
              both: "Matches the application name on agent runs or the model group on requests. Leave blank to include both without a name filter.",
              traces:
                "The service.name recorded by your agent’s OpenTelemetry instrumentation. Leave blank for all applications.",
            }[value.source ?? "traces"]
          }
        </p>
        <div className="space-y-2">
          <p className="text-sm font-medium">
            Narrow by metadata <span className="font-normal text-muted-foreground">(optional)</span>
          </p>
          <p className="text-xs text-muted-foreground">
            Match a recorded tag, swarm, or environment. Every condition must match exactly.
          </p>
          {filters.map((f, index) => (
            <div key={index} className="flex items-center gap-2">
              <Input
                aria-label={`Metadata key ${index + 1}`}
                list={`${id}-keys`}
                placeholder="Choose or enter a key"
                value={f.key}
                onChange={(e) => edit(index, "key", e.target.value)}
              />
              <span className="text-xs text-muted-foreground">is</span>
              <Input
                aria-label={`Metadata value ${index + 1}`}
                list={`${id}-values-${index}`}
                placeholder="Choose or enter a value"
                value={f.value}
                onChange={(e) => edit(index, "value", e.target.value)}
              />
              <datalist id={`${id}-values-${index}`}>
                {[...new Set(attributes.filter((a) => a.key === f.key).map((a) => a.value))].sort().map((v) => (
                  <option key={v} value={v} />
                ))}
              </datalist>
              <Button
                variant="ghost"
                size="icon"
                aria-label={`Remove condition ${index + 1}`}
                onClick={() => onChange({ ...value, filters: filters.filter((_, i) => i !== index) })}
              >
                <X className="size-4" />
              </Button>
            </div>
          ))}
          <datalist id={`${id}-keys`}>
            {keys.map((key) => (
              <option key={key} value={key} />
            ))}
          </datalist>
          <Button
            variant="outline"
            size="sm"
            disabled={filters.length >= 8}
            onClick={() => onChange({ ...value, filters: [...filters, { key: "", value: "" }] })}
          >
            <Plus className="size-3" />
            Add condition
          </Button>
          <p className="text-xs text-muted-foreground">
            Suggestions come from up to 100 recent runs. You can also type a recorded key or value.
          </p>
        </div>
        <label className="grid gap-2 text-sm">
          Team ID (optional)
          <Input
            value={value.team_id ?? ""}
            placeholder="All teams you can access"
            onChange={(e) => onChange({ ...value, team_id: e.target.value })}
          />
        </label>
        <DurationInput
          label="Review the last"
          value={value.lookback_hours ?? 24}
          base="hours"
          max={720}
          onChange={(lookback_hours) => onChange({ ...value, lookback_hours })}
        />
        <p className="text-xs text-muted-foreground">
          Time window used by each scan. Activity becomes eligible two minutes after it finishes.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <label className="grid gap-2 text-sm">
            Sample (%)
            <Input
              type="number"
              min="0.01"
              max="100"
              step="any"
              value={value.sample_percent ?? 100}
              onChange={(e) => onChange({ ...value, sample_percent: Number(e.target.value) })}
            />
          </label>
          <label className="grid gap-2 text-sm">
            Maximum runs (optional)
            <Input
              type="number"
              min="1"
              placeholder="No limit"
              value={value.sample_size ?? ""}
              onChange={(e) => onChange({ ...value, sample_size: e.target.value ? Number(e.target.value) : null })}
            />
          </label>
        </div>
        <p className="text-xs text-muted-foreground">100% with no limit selects all matching activity.</p>
        {!!value.execution_ids?.length && (
          <Button variant="outline" onClick={() => onChange({ ...value, execution_ids: [] })}>
            Clear {value.execution_ids.length} selected runs
          </Button>
        )}
      </div>
      <MatchingActivity
        offset={offset}
        onPage={setOffset}
        onSelect={(runId, checked) =>
          onChange({
            ...value,
            execution_ids: checked
              ? [...(value.execution_ids ?? []), runId]
              : (value.execution_ids ?? []).filter((id) => id !== runId),
          })
        }
        selectedIds={value.execution_ids ?? []}
        selectedCount={
          value.execution_ids?.length
            ? Math.min(
                Math.ceil((value.execution_ids.length * (value.sample_percent ?? 100)) / 100),
                value.sample_size ?? Infinity,
              )
            : preview.data?.selected ?? 0
        }
        title={previewTitle()}
        windowLabel={windowLabel}
        ready={ready}
        error={preview.error}
        data={preview.data}
        onOpen={(run) => setTrace({ id: run.trace_id, ref: run.trace_ref })}
      />
      {trace && (
        <TracePanel
          open
          traceId={trace.id}
          traceRef={trace.ref}
          accessToken={accessToken}
          onClose={() => setTrace(null)}
        />
      )}
    </div>
  );
}

function MatchingActivity({
  offset,
  onPage,
  onSelect,
  selectedIds,
  selectedCount,
  title,
  windowLabel,
  ready,
  error,
  data,
  onOpen,
}: {
  offset: number;
  onPage: (offset: number) => void;
  onSelect: (id: string, checked: boolean) => void;
  selectedIds: string[];
  selectedCount: number;
  title: string;
  windowLabel: string;
  ready: boolean;
  error: Error | null;
  data: Sample | undefined;
  onOpen: (run: Sample["executions"][number]) => void;
}) {
  return (
    <section aria-label="Matching activity" className="self-start rounded-lg border">
      <div className="border-b px-4 py-3">
        <p className="text-sm font-medium" role="status">
          {title}
        </p>
        <p className="mt-1 text-xs text-muted-foreground">{windowLabel} · Preview only, no analysis cost</p>
      </div>
      <div className="max-h-80 overflow-y-auto px-4">
        {ready && error && (
          <p role="alert" className="py-3 text-sm text-destructive">
            {error.message}
          </p>
        )}
        {ready && data?.eligible === 0 && (
          <p className="py-4 text-sm text-muted-foreground">
            No matches. Try removing a condition or check that your agent records this metadata. Very recent runs need
            two minutes to settle.
          </p>
        )}
        {ready &&
          data?.executions.map((run) => (
            <div key={run.id} className="flex items-center justify-between gap-3 border-b last:border-0">
              <input
                type="checkbox"
                aria-label={`Select ${run.name}`}
                checked={selectedIds.includes(run.id)}
                onChange={(e) => onSelect(run.id, e.target.checked)}
              />
              <div className="min-w-0">
                <RunList executions={[run]} />
              </div>
              {run.source === "traces" && (
                <Button variant="ghost" size="sm" aria-label={`Open ${run.name}`} onClick={() => onOpen(run)}>
                  Open run
                  <ArrowUpRight className="size-3" />
                </Button>
              )}
            </div>
          ))}
      </div>
      {ready && data && (
        <div className="border-t px-4 py-3 space-y-2">
          <p className="text-xs text-muted-foreground">
            {selectedCount} selected for analysis · Showing {offset + (data.executions.length ? 1 : 0)}–
            {offset + data.executions.length} of {data.eligible}
          </p>
          <div className="flex justify-between">
            <Button size="sm" variant="ghost" disabled={offset === 0} onClick={() => onPage(Math.max(0, offset - 100))}>
              Previous
            </Button>
            <Button
              size="sm"
              variant="ghost"
              disabled={data.next_offset == null}
              onClick={() => onPage(data.next_offset ?? offset)}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </section>
  );
}
