"use client";

import type { components } from "@/lib/http/schema";
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Aperture,
  ArrowUpRight,
  CheckCircle2,
  Circle,
  Info,
  Layers3,
  Pause,
  Play,
  Plus,
  Settings2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";
import { apiClient } from "@/components/networking";
import { TracePanel } from "./TracePanel";
import { EngineSetup } from "./EngineSetup";
import { LensRuns } from "./LensRuns";
import { EngineProgress, NextCheck, ScanDuration } from "./EngineProgress";
import { WorkerSetup } from "./WorkerSetup";
import { LensWelcome } from "./LensWelcome";
import {
  engineStatus,
  evidenceTarget,
  sortedFindings,
  runTime,
  type Engine,
  type EngineList,
  type Finding,
  type Settings,
  type Job,
} from "./engineData";

const money = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 3 }).format(n);
const when = (value?: string | null) => (value ? new Date(value).toLocaleString() : "Not yet");

const sourceLabels = { both: "Traces and requests", requests: "LLM requests", traces: "Agent traces" };
const priorityColors = { high: "bg-red-500", medium: "bg-amber-500", low: "bg-slate-400" };
function emptyFindingTitle(active: boolean, scanned: boolean) {
  if (active) return "Your findings will appear here";
  return scanned ? "No matching findings" : "Ready for the first analysis";
}

export function EngineView({ accessToken, readOnly = false }: { accessToken: string; readOnly?: boolean }) {
  const client = useQueryClient();
  const key = ["engines", accessToken];
  const query = useQuery({
    queryKey: key,
    queryFn: () => apiClient.get<EngineList>("/engine", { accessToken }),
    refetchInterval: 10000,
  });
  const models = useQuery({
    queryKey: ["engine-models", accessToken],
    queryFn: () => apiClient.get<{ data: { id: string }[] }>("/models", { accessToken }),
  });
  const modelDetails = useQuery({
    queryKey: ["lens-model-details", accessToken],
    queryFn: () =>
      apiClient.get<{ data: import("./engineData").AnalysisModelInfo[] }>("/model_group/info", { accessToken }),
  });
  const [selected, setSelected] = useState<string | null>(() =>
    typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("lens"),
  );
  const selectLens = (id: string) => {
    setSelected(id);
    setBatchId("latest");
    setHistoryOffset(0);
    setFindingId(null);
    const url = new URL(window.location.href);
    url.searchParams.set("lens", id);
    window.history.replaceState(window.history.state, "", url);
  };
  const [editing, setEditing] = useState<"new" | "edit" | "duplicate" | null>(null);
  const [workerSetup, setWorkerSetup] = useState(false);
  const [batchId, setBatchId] = useState("latest");
  const [historyOffset, setHistoryOffset] = useState(0);
  const [tab, setTab] = useState("findings");
  const [findingId, setFindingId] = useState<string | null>(null);
  const [filter, setFilter] = useState("open");
  const [kind, setKind] = useState<"issue" | "pattern">("issue");
  const [reason, setReason] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [evidence, setEvidence] = useState<{ id: string; span: string } | null>(null);
  const engines = [...(query.data?.engines ?? [])].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
  const showEmpty = !query.isLoading && !query.error && engines.length === 0;
  const engine = engines.find((e) => e.id === selected) ?? engines[0];
  const connected =
    query.data?.workers?.some((w) => !w.revoked && query.dataUpdatedAt - Date.parse(w.last_seen) < 120000) ?? false;
  const historyQuery = {
    queryKey: ["lens-history", engine?.id, historyOffset, accessToken],
    enabled: !!engine,
    queryFn: () =>
      apiClient.get<Job[]>(`/engine/${engine?.id}/runs`, { accessToken, query: { offset: historyOffset } }),
    refetchInterval: 10000,
  };
  const history = useQuery(historyQuery);
  const historical = useQuery({
    queryKey: ["lens-batch", engine?.id, batchId, accessToken],
    enabled: !!engine && !["latest", "all"].includes(batchId),
    queryFn: () => apiClient.get<Job>(`/engine/${engine?.id}/runs/${batchId}`, { accessToken }),
  });
  const job = ["latest", "all"].includes(batchId) ? engine?.jobs?.[0] : historical.data;
  const missingSnapshot = job?.status === "completed" && job.findings == null && batchId !== "all";
  const selectedOutsideHistory = !["latest", "all"].includes(batchId) && !history.data?.some((j) => j.id === batchId);
  const batchSettings = job?.settings ?? engine?.settings;
  const batchFindings = (batchId === "all" ? engine?.findings ?? [] : job?.findings ?? []).map((f) => {
    const feedback = engine?.findings?.find((current) => current.id === f.id);
    return feedback ? { ...f, status: feedback.status, reason: feedback.reason } : f;
  });
  const finding = batchFindings.find((f) => f.id === findingId);
  const openBatch = (id: string) => {
    setBatchId(id);
    setTab("findings");
    setFindingId(null);
  };
  const setupSettings = () => {
    if (editing === "new") return undefined;
    if (editing === "duplicate" && engine)
      return { ...engine.settings, name: `${engine.settings.name} copy`, enabled: false };
    return engine?.settings;
  };
  const lastCompleted = engine?.jobs?.find((j) => j.status === "completed");
  const active = engine?.jobs?.find((j) => j.status === "queued" || j.status === "running");
  const visibleFindings = sortedFindings(
    batchFindings.filter((f) => (filter === "all" || f.status === filter) && f.kind === kind),
  );
  const sampledRuns = job?.sample?.executions ?? [];
  const evidenceGroups = finding
    ? [...new Set(finding.evidence.map((e) => e.execution_id))].map((id) => ({
        id,
        run: sampledRuns.find((r) => r.id === id),
        quotes: finding.evidence.filter((e) => e.execution_id === id),
      }))
    : [];
  const target = evidence ? evidenceTarget(evidence.id) : null;
  const [requestOffset, setRequestOffset] = useState(0);
  const requestEvidence = useQuery({
    queryKey: ["engine-evidence", engine?.id, evidence?.id, requestOffset, accessToken],
    enabled: !!engine && target?.source === "requests",
    queryFn: () =>
      apiClient.get<components["schemas"]["ExecutionContent"]>(
        `/engine/${engine?.id}/executions/${encodeURIComponent(evidence?.id ?? "")}`,
        { accessToken, query: { offset: requestOffset } },
      ),
  });
  const refresh = () => {
    void client.invalidateQueries({ queryKey: key });
    void client.invalidateQueries({ queryKey: ["lens-history"] });
  };
  const update = async (path: string, body: unknown, method: "post" | "put" | "patch" = "post") => {
    setBusy(true);
    setError("");
    try {
      await apiClient[method](path, { accessToken, body });
      await client.invalidateQueries({ queryKey: key });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not update lens");
    } finally {
      setBusy(false);
    }
  };
  const save = async (settings: Settings) => {
    const saved = await apiClient.request<Engine>(
      editing === "edit" ? "PUT" : "POST",
      editing === "edit" ? `/engine/${engine.id}` : "/engine",
      { accessToken, body: settings },
    );
    selectLens(saved.id);
    setEditing(null);
    refresh();
  };
  const changeFinding = async (status: Finding["status"]) => {
    if (!engine || !finding) return;
    await update(`/engine/${engine.id}/findings/${finding.id}`, { status, reason }, "patch");
  };

  return (
    <main className="w-full min-w-0 p-6 md:p-8 space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Aperture aria-hidden="true" className="size-7" strokeWidth={1.75} />
            <h1 className="text-2xl font-semibold tracking-tight">Lens</h1>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Understand your agent activity. Find patterns worth acting on.
          </p>
        </div>
        {!readOnly && (
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => setWorkerSetup(true)}>
              <Circle
                className={`size-2 ${connected ? "fill-emerald-500 text-emerald-500" : "fill-amber-500 text-amber-500"}`}
              />
              {connected ? "Analyzer connected" : "Set up analysis"}
            </Button>
            {engines.length > 0 && (
              <Button onClick={() => setEditing("new")}>
                <Plus className="size-4" />
                New lens
              </Button>
            )}
          </div>
        )}
      </header>
      {(error || query.error) && (
        <div role="alert" className="rounded-lg border border-destructive/30 p-4 text-sm text-destructive">
          {error || query.error?.message}
          <Button variant="ghost" size="sm" onClick={refresh}>
            Retry
          </Button>
        </div>
      )}
      {query.isLoading && (
        <p role="status" className="py-20 text-center text-muted-foreground">
          Loading lenses…
        </p>
      )}
      {showEmpty && (
        <LensWelcome
          connected={connected}
          readOnly={!!readOnly}
          onConnect={() => setWorkerSetup(true)}
          onCreate={() => setEditing("new")}
        />
      )}
      {engine && (
        <div className="grid gap-6 lg:grid-cols-[220px_minmax(0,1fr)]">
          <nav aria-label="Lenses" className="flex gap-2 overflow-x-auto lg:flex-col lg:overflow-visible">
            {engines.map((e) => (
              <button
                key={e.id}
                onClick={() => selectLens(e.id)}
                aria-current={engine.id === e.id ? "page" : undefined}
                className={`min-w-44 rounded-lg px-3 py-3 text-left transition-colors ${engine.id === e.id ? "bg-muted" : "hover:bg-muted/50"}`}
              >
                <span className="block truncate text-sm font-medium">{e.settings.name}</span>
                <span className="mt-1 block text-xs text-muted-foreground">{engineStatus(e, connected)}</span>
              </button>
            ))}
          </nav>
          <section className="min-w-0 space-y-5">
            <div className="flex flex-wrap justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">{engine.settings.name}</h2>
                <p className="mt-1 text-xs text-muted-foreground">
                  {sourceLabels[engine.settings.source ?? "traces"]} ·{" "}
                  {engine.settings.service || "All accessible activity"}
                  {engine.settings.filters?.length ? ` · ${engine.settings.filters.length} filters` : ""}
                </p>
              </div>
              {!readOnly && (
                <div className="flex gap-2">
                  <Button variant="ghost" size="icon" aria-label="Lens settings" onClick={() => setEditing("edit")}>
                    <Settings2 className="size-4" />
                  </Button>
                  <Button variant="outline" onClick={() => setEditing("duplicate")}>
                    Duplicate
                  </Button>
                  <Button
                    variant="outline"
                    disabled={busy}
                    onClick={() =>
                      update(`/engine/${engine.id}`, { ...engine.settings, enabled: !engine.settings.enabled }, "put")
                    }
                  >
                    {engine.settings.enabled ? <Pause className="size-3" /> : <Play className="size-3" />}
                    {engine.settings.enabled ? "Pause" : "Resume"}
                  </Button>
                  <Button
                    disabled={busy || !!active || !connected}
                    onClick={() => update(`/engine/${engine.id}/runs`, {})}
                  >
                    <Play className="size-3" />
                    Run now
                  </Button>
                </div>
              )}
            </div>
            {!query.data?.tracing_enabled && (
              <div role="status" className="rounded-lg border border-amber-200 bg-amber-50/30 p-3 text-sm">
                Enable agent tracing and ClickHouse on this proxy before running an analysis.
              </div>
            )}
            <div className="grid grid-cols-1 gap-4 rounded-xl border p-4 sm:grid-cols-3">
              <div>
                <p className="text-xs text-muted-foreground">Status</p>
                <p className="mt-1 text-sm font-medium" role="status">
                  {engineStatus(engine, connected)}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {engine.settings.enabled
                    ? `Checks every ${engine.settings.interval_minutes} minutes`
                    : "Manual analysis available"}
                </p>
                <NextCheck engine={engine} />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Last successful scan</p>
                <p className="mt-1 text-sm">{when(lastCompleted?.finished_at ?? engine.last_scan_at)}</p>
                {lastCompleted && (
                  <p className="mt-1 text-xs text-muted-foreground">
                    {lastCompleted.coverage?.screened ?? 0} of {lastCompleted.coverage?.eligible ?? 0} eligible runs
                    reviewed <ScanDuration job={lastCompleted} />
                  </p>
                )}
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Analysis spend this month</p>
                <p className="mt-1 text-sm">
                  {money(engine.budget_month === new Date().toISOString().slice(0, 7) ? engine.spent ?? 0 : 0)}{" "}
                  <span className="text-muted-foreground">/ {money(engine.settings.monthly_budget ?? 20)}</span>
                </p>
                <p className="mt-1 text-xs text-muted-foreground">Includes reservations for pending calls</p>
              </div>
            </div>
            {active && (
              <EngineProgress
                key={active.id}
                job={active}
                onCancel={
                  readOnly
                    ? undefined
                    : () => {
                        void update(`/engine/${engine.id}/cancel`, {});
                      }
                }
              />
            )}
            {job?.error && (
              <p role="alert" className="text-sm text-destructive">
                {job.error}
              </p>
            )}
            <Tabs value={tab} onValueChange={setTab} key={engine.id}>
              <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b">
                <TabsList variant="line">
                  <TabsTrigger value="findings">Findings</TabsTrigger>
                  <TabsTrigger value="checks">Questions & checks</TabsTrigger>
                  <TabsTrigger value="runs">Runs</TabsTrigger>
                  <TabsTrigger value="activity">Scans</TabsTrigger>
                </TabsList>
                {tab !== "activity" && (
                  <div className="flex min-w-0 items-center gap-1 pb-1">
                    <select
                      aria-label="Investigation batch"
                      className="h-8 w-44 max-w-full truncate rounded-md border-0 bg-transparent px-2 text-xs text-muted-foreground hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
                      value={batchId}
                      onChange={(e) => {
                        setBatchId(e.target.value);
                        setFindingId(null);
                      }}
                    >
                      <option value="latest">Latest batch</option>
                      {job && selectedOutsideHistory && (
                        <option value={batchId}>
                          {when(job.created_at)} · {job.status}
                        </option>
                      )}
                      {(history.data ?? engine.jobs)?.map((j) => (
                        <option key={j.id} value={j.id}>
                          {when(j.created_at)} · {j.status}
                        </option>
                      ))}
                      <option value="all">All accumulated findings</option>
                    </select>
                    {job && batchId !== "all" && (
                      <Popover key={job.id}>
                        <PopoverTrigger
                          aria-label="Batch details"
                          render={<Button variant="ghost" size="icon" className="size-7 text-muted-foreground" />}
                        >
                          <Info className="size-3.5" />
                        </PopoverTrigger>
                        <PopoverContent align="end" className="gap-3">
                          <PopoverTitle>Batch details</PopoverTitle>
                          <p className="text-xs text-muted-foreground">
                            {job.coverage?.screened ?? 0} / {job.coverage?.selected ?? 0} selected runs reviewed
                            <ScanDuration job={job} />
                          </p>
                          <dl className="space-y-2 text-xs">
                            <div>
                              <dt className="text-muted-foreground">Activity window</dt>
                              <dd className="mt-1">
                                {when(job.start)} to {when(job.end)}
                              </dd>
                            </div>
                            <div className="flex justify-between gap-2">
                              <dt className="text-muted-foreground">Analysis cost</dt>
                              <dd>{money(job.cost ?? 0)}</dd>
                            </div>
                            <div className="flex justify-between gap-2">
                              <dt className="text-muted-foreground">Status</dt>
                              <dd className="capitalize">{job.status}</dd>
                            </div>
                          </dl>
                        </PopoverContent>
                      </Popover>
                    )}
                  </div>
                )}
              </div>
              {missingSnapshot && tab !== "activity" && (
                <p className="text-sm text-muted-foreground">
                  This older batch predates saved result snapshots. Its findings remain available under All accumulated
                  findings.
                </p>
              )}
              <TabsContent value="findings" className="pt-4 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex gap-1" aria-label="Finding category">
                    <Button
                      size="sm"
                      variant={kind === "issue" ? "secondary" : "ghost"}
                      onClick={() => setKind("issue")}
                    >
                      Needs attention (
                      {batchFindings.filter((f) => f.kind === "issue" && f.status === "open").length ?? 0})
                    </Button>
                    <Button
                      size="sm"
                      variant={kind === "pattern" ? "secondary" : "ghost"}
                      onClick={() => setKind("pattern")}
                    >
                      Patterns ({batchFindings.filter((f) => f.kind === "pattern" && f.status === "open").length ?? 0})
                    </Button>
                  </div>
                  <select
                    aria-label="Finding status"
                    className="rounded-md border bg-background px-2 py-1 text-xs"
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                  >
                    <option value="open">Open</option>
                    <option value="resolved">Resolved</option>
                    <option value="dismissed">Dismissed</option>
                    <option value="all">All statuses</option>
                  </select>
                </div>
                <p className="text-xs text-muted-foreground">
                  {kind === "issue"
                    ? "Problems worth investigating, highest priority first."
                    : "Useful behavior and trends. These do not necessarily need a fix."}
                </p>
                <div className="divide-y rounded-xl border">
                  {visibleFindings.map((f) => (
                    <button
                      key={f.id}
                      onClick={() => {
                        setFindingId(f.id);
                        setReason(f.reason ?? "");
                      }}
                      className="flex w-full gap-4 p-4 text-left hover:bg-muted/30"
                    >
                      <span
                        className={`mt-1 size-2 shrink-0 rounded-full ${priorityColors[f.priority ?? "medium"]}`}
                        aria-label={`${f.priority} priority`}
                      />
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium">{f.title}</p>
                        <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">{f.description}</p>
                        <p className="mt-2 text-xs text-muted-foreground">
                          {f.occurrences?.length ?? 0} linked runs ·{" "}
                          {f.kind === "issue" ? `${f.priority} priority` : "Pattern"}
                        </p>
                      </div>
                      <ArrowUpRight className="size-4 text-muted-foreground" />
                    </button>
                  ))}
                  {visibleFindings.length === 0 && (
                    <div className="px-6 py-14 text-center">
                      <CheckCircle2 className="mx-auto mb-3 size-5 text-muted-foreground" />
                      <p className="text-sm font-medium">{emptyFindingTitle(!!active, !!engine.last_scan_at)}</p>
                      <p className="mt-2 text-xs text-muted-foreground">
                        {active
                          ? "Lens is reviewing the selected activity."
                          : "Findings reflect the runs analyzed, not a guarantee about all activity."}
                      </p>
                    </div>
                  )}
                </div>
              </TabsContent>
              <TabsContent value="checks" className="pt-4 space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">Checks used for the selected batch</p>
                  {!readOnly && (
                    <Button variant="outline" size="sm" onClick={() => setEditing("edit")}>
                      Edit questions
                    </Button>
                  )}
                </div>
                {batchSettings?.context && (
                  <div className="rounded-lg bg-muted/40 p-4">
                    <p className="text-xs font-medium">Expected behavior</p>
                    <p className="mt-2 whitespace-pre-wrap text-sm">{batchSettings.context}</p>
                  </div>
                )}
                {batchSettings?.checks.map((c) => (
                  <div key={c.id} className="flex items-start gap-3 rounded-lg border p-4">
                    <Layers3 className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                    <p className="text-sm flex-1">{c.instruction}</p>
                    {!readOnly && batchId === "latest" && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() =>
                          update(
                            `/engine/${engine.id}`,
                            {
                              ...engine.settings,
                              checks: engine.settings.checks.map((q) =>
                                q.id === c.id ? { ...q, enabled: !q.enabled } : q,
                              ),
                            },
                            "put",
                          )
                        }
                      >
                        {c.enabled ? "Disable" : "Enable"}
                      </Button>
                    )}
                  </div>
                ))}
                {!readOnly && (
                  <Button
                    variant="outline"
                    disabled={!!active || !connected}
                    onClick={() => update(`/engine/${engine.id}/runs`, {})}
                  >
                    Run saved settings now
                  </Button>
                )}
                <p className="text-xs text-muted-foreground">
                  Changes apply to future scans. Rechecking history uses your analysis budget.
                </p>
              </TabsContent>
              <TabsContent value="runs" className="pt-4 space-y-4">
                <div className="rounded-lg border p-4 text-sm space-y-2">
                  <p className="font-medium">Activity this lens reviews</p>
                  <p>
                    {sourceLabels[batchSettings?.source ?? "traces"]} · {batchSettings?.service || "All services"}
                  </p>
                  {batchSettings?.filters?.map((f) => (
                    <p key={f.key} className="text-muted-foreground">
                      {f.key} is {f.value}
                    </p>
                  ))}
                  {!readOnly && (
                    <Button variant="outline" size="sm" onClick={() => setEditing("edit")}>
                      Change selection
                    </Button>
                  )}
                </div>
                <LensRuns
                  key={job?.id ?? batchId}
                  job={job}
                  onOpen={(id) => {
                    setRequestOffset(0);
                    setEvidence({ id, span: "" });
                  }}
                />
              </TabsContent>
              <TabsContent value="activity" className="pt-4 space-y-3">
                <div className="flex justify-between">
                  <Button
                    variant="outline"
                    disabled={!historyOffset}
                    onClick={() => setHistoryOffset(Math.max(0, historyOffset - 50))}
                  >
                    Newer batches
                  </Button>
                  <Button
                    variant="outline"
                    disabled={(history.data?.length ?? 0) < 50}
                    onClick={() => setHistoryOffset(historyOffset + 50)}
                  >
                    Older batches
                  </Button>
                </div>
                {history.error && <p role="alert">{history.error.message}</p>}
                {(history.data ?? engine.jobs)?.map((j) => (
                  <div key={j.id} className="rounded-lg border p-4">
                    <div className="flex justify-between gap-3 text-sm">
                      <span className="font-medium">{j.stage}</span>
                      <span>{money(j.cost ?? 0)}</span>
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {when(j.created_at)} · Settings version {j.revision}
                      <ScanDuration job={j} />
                    </p>
                    <p className="mt-3 text-sm">
                      {j.coverage?.screened ?? 0} reviewed / {j.coverage?.eligible ?? 0} eligible ·{" "}
                      {j.coverage?.investigated ?? 0} patterns investigated · {j.coverage?.inconclusive ?? 0}{" "}
                      inconclusive
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {j.coverage?.partial ?? 0} partial executions · {j.coverage?.unassessable ?? 0} could not be
                      assessed
                    </p>
                    <Button size="sm" variant="outline" className="mt-3" onClick={() => openBatch(j.id)}>
                      View results
                    </Button>
                    {j.error && <p className="mt-2 text-sm text-destructive">{j.error}</p>}
                  </div>
                ))}
              </TabsContent>
            </Tabs>
          </section>
        </div>
      )}
      {editing && (
        <EngineSetup
          mode={editing}
          initial={setupSettings()}
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
        <WorkerSetup
          accessToken={accessToken}
          workers={query.data?.workers ?? []}
          onClose={() => setWorkerSetup(false)}
          onChanged={refresh}
        />
      )}
      <Sheet
        open={!!finding}
        onOpenChange={(open) => {
          if (!open) setFindingId(null);
        }}
      >
        <SheetContent className="overflow-y-auto data-[side=right]:sm:max-w-2xl">
          {finding && (
            <>
              <SheetHeader>
                <SheetTitle className="pr-8 text-xl leading-snug">{finding.title}</SheetTitle>
                <SheetDescription>
                  {finding.kind === "issue" ? `${finding.priority} priority` : "Pattern"} ·{" "}
                  {finding.occurrences?.length ?? 0} linked runs
                </SheetDescription>
              </SheetHeader>
              <div className="space-y-6 p-4">
                <div>
                  <p className="mb-2 text-xs font-medium text-muted-foreground">What happened</p>
                  <p className="text-sm leading-6 whitespace-pre-wrap">{finding.description}</p>
                </div>
                {finding.suggestion && (
                  <div className="rounded-lg bg-muted/40 p-4">
                    <p className="text-sm font-medium">What to do next</p>
                    <p className="mt-2 text-sm leading-6">{finding.suggestion}</p>
                  </div>
                )}
                {finding.limitation && (
                  <details className="rounded-lg border p-3 text-sm">
                    <summary className="cursor-pointer font-medium">What this does and doesn’t tell us</summary>
                    <p className="mt-3 leading-6 text-muted-foreground">{finding.limitation}</p>
                  </details>
                )}
                <div>
                  <p className="text-sm font-medium">Evidence by run</p>
                  <p className="mt-1 mb-3 text-xs text-muted-foreground">
                    Exact quotes from the recorded activity. Counterexamples are labeled separately from supporting
                    evidence.
                  </p>
                  <div className="space-y-2">
                    {evidenceGroups.map((group) => (
                      <details key={group.id} className="rounded-lg border p-3">
                        <summary className="cursor-pointer text-sm font-medium">
                          {group.run?.name ?? evidenceTarget(group.id)?.id.slice(0, 12) ?? "Recorded run"}
                          <span className="ml-2 text-xs font-normal text-muted-foreground">
                            {group.quotes.length} quotes{group.run ? ` · ${runTime(group.run.start_time)}` : ""}
                          </span>
                        </summary>
                        <div className="mt-3 space-y-3">
                          {group.quotes.map((e, i) => (
                            <div key={`${e.span_id}-${i}`} className="rounded-md bg-muted/40 p-3">
                              {e.role === "counterexample" && (
                                <p className="mb-1 text-xs font-medium text-muted-foreground">Counterexample</p>
                              )}
                              <blockquote className="text-xs leading-5 whitespace-pre-wrap break-words">
                                {e.quote}
                              </blockquote>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="mt-2"
                                onClick={() => {
                                  setRequestOffset(0);
                                  setEvidence({ id: e.execution_id, span: e.span_id });
                                }}
                              >
                                {evidenceTarget(e.execution_id)?.source === "traces"
                                  ? "Open original step"
                                  : "Open request"}
                                <ArrowUpRight className="size-3" />
                              </Button>
                            </div>
                          ))}
                        </div>
                      </details>
                    ))}
                  </div>
                </div>
                {!readOnly && (
                  <div className="space-y-3 border-t pt-4">
                    <label className="grid gap-2 text-sm">
                      What should Lens remember?
                      <Textarea
                        value={reason}
                        onChange={(e) => setReason(e.target.value)}
                        placeholder="What should Lens know about this finding?"
                      />
                    </label>
                    <p className="text-xs text-muted-foreground">Your explanation informs future scans of this Lens.</p>
                    <div className="flex flex-wrap gap-2">
                      {finding.kind === "issue" && (
                        <Button
                          disabled={busy}
                          onClick={() => changeFinding(finding.status === "resolved" ? "open" : "resolved")}
                        >
                          {finding.status === "resolved" ? "Reopen" : "Mark resolved"}
                        </Button>
                      )}
                      <Button disabled={busy} variant="outline" onClick={() => changeFinding("dismissed")}>
                        This is expected
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </SheetContent>
      </Sheet>
      {engine && target?.source === "traces" && (
        <TracePanel
          open={!!evidence}
          traceId={target.id}
          traceRef={target.traceRef}
          initialSpanId={evidence?.span}
          accessToken={accessToken}
          onClose={() => setEvidence(null)}
        />
      )}
      <Sheet
        open={target?.source === "requests"}
        onOpenChange={(open) => {
          if (!open) setEvidence(null);
        }}
      >
        <SheetContent className="overflow-y-auto data-[side=right]:sm:max-w-2xl">
          <SheetHeader>
            <SheetTitle>Request evidence</SheetTitle>
            <SheetDescription>Original logged input and output</SheetDescription>
          </SheetHeader>
          <div className="p-4 space-y-3">
            {requestEvidence.isLoading && <p role="status">Loading request…</p>}
            {requestEvidence.error && <p role="alert">{requestEvidence.error.message}</p>}
            {requestEvidence.data?.parts.map((p) => (
              <pre className="whitespace-pre-wrap break-words text-xs" key={p.span_id}>
                {p.content}
              </pre>
            ))}
            {requestEvidence.data?.parts.length === 0 && <p>Request was not found or is past retention</p>}
            <div className="flex gap-2">
              {requestOffset > 0 && (
                <Button variant="outline" onClick={() => setRequestOffset(Math.max(0, requestOffset - 8000))}>
                  Previous section
                </Button>
              )}
              {requestEvidence.data?.parts.some((p) => p.truncated) && (
                <Button
                  variant="outline"
                  onClick={() => setRequestOffset(requestOffset === 0 ? 1 : requestOffset + 8000)}
                >
                  Next section
                </Button>
              )}
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </main>
  );
}
