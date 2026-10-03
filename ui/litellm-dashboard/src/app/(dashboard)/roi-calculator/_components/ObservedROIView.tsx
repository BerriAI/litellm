"use client";

import { useEffect, useState } from "react";
import { ArrowDown, ArrowUp, CalendarDays, ChevronRight, Github, Info, Link2, Search } from "lucide-react";
import { apiClient } from "@/components/networking";
import { Page, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { PageHeader, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import ObservedConnections from "./ObservedConnections";
import { CoverageDetails, PersonDetails, PullList } from "./ObservedDetails";
import {
  change,
  dateRange,
  duration,
  money,
  number,
  observedSnapshotSchema,
  visiblePeople,
  weeklyMerges,
  type Comparison,
  type ObservedPerson,
  type ObservedSnapshot,
  type PeopleSort,
} from "./observedData";

function Delta({ current, baseline, neutral = false }: { current: number; baseline: number; neutral?: boolean }) {
  const delta = change(current, baseline);
  if (delta === null) return <span className="text-xs text-muted-foreground">No baseline</span>;
  const Icon = delta >= 0 ? ArrowUp : ArrowDown;
  return (
    <span
      aria-label={`${number(Math.abs(delta))}% ${delta >= 0 ? "increase" : "decrease"}`}
      className={`inline-flex items-center gap-1 text-xs tabular-nums ${neutral ? "text-muted-foreground" : "text-foreground"}`}
    >
      <Icon className="size-3" />
      {number(Math.abs(delta))}%
    </span>
  );
}

function Metric({
  label,
  value,
  detail,
  current,
  baseline,
  onInfo,
}: {
  label: string;
  value: string;
  detail: string;
  current?: number;
  baseline?: number;
  onInfo?: () => void;
}) {
  return (
    <div className="min-w-0 px-5 py-5">
      <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        {label}
        {onInfo && (
          <button
            type="button"
            aria-label={`About ${label}`}
            onClick={onInfo}
            className="rounded hover:text-foreground"
          >
            <Info className="size-3.5" />
          </button>
        )}
      </div>
      <div className="mt-3 flex flex-wrap items-baseline gap-3">
        <span className="text-3xl font-semibold tracking-tight tabular-nums">{value}</span>
        {current !== undefined && baseline !== undefined && <Delta current={current} baseline={baseline} />}
      </div>
      <p className="mt-2 text-xs text-muted-foreground">{detail}</p>
    </div>
  );
}

function ShippingTrend({ snapshot, comparison }: { snapshot: ObservedSnapshot; comparison: Comparison }) {
  const current = weeklyMerges(snapshot, "current");
  const baseline = weeklyMerges(snapshot, comparison);
  const max = Math.max(1, ...current, ...baseline);
  return (
    <div className="rounded-xl border p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-medium">Shipping activity</h2>
        <div className="flex gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-sm bg-blue-500" />
            Current period
          </span>
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-sm bg-slate-300 dark:bg-slate-600" />
            {comparison === "previous" ? "Previous period" : "Last year"}
          </span>
        </div>
      </div>
      <div
        className="mt-5 grid grid-cols-4 gap-6"
        role="img"
        aria-label={`Merged PRs by week. Current: ${current.join(", ")}. Comparison: ${baseline.join(", ")}`}
      >
        {current.map((value, week) => (
          <div key={week} className="min-w-0">
            <div className="flex h-28 items-end justify-center gap-2 border-b border-border/60">
              <div
                className="group relative w-10 rounded-t bg-slate-200 dark:bg-slate-700"
                style={{ height: `${Math.max(2, (baseline[week] / max) * 85)}%` }}
              >
                <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] text-muted-foreground">
                  {baseline[week]}
                </span>
              </div>
              <div
                className="relative w-10 rounded-t bg-blue-500/90"
                style={{ height: `${Math.max(2, (value / max) * 85)}%` }}
              >
                <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-medium">{value}</span>
              </div>
            </div>
            <p className="mt-2 text-center text-[11px] text-muted-foreground">Week {week + 1}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PeopleTable({
  snapshot,
  comparison,
  onSelect,
}: {
  snapshot: ObservedSnapshot;
  comparison: Comparison;
  onSelect: (person: ObservedPerson) => void;
}) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<PeopleSort>("merged");
  const people = visiblePeople(snapshot.people, query, sort);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="relative w-72">
          <Search className="absolute top-2.5 left-3 size-4 text-muted-foreground" />
          <Input
            aria-label="Search engineers"
            placeholder="Search engineers…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            className="pl-9"
          />
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground">{people.length} people</span>
          <Select
            value={sort}
            onValueChange={(value) => {
              if (value) setSort(value);
            }}
            items={[
              { value: "merged", label: "Most merged PRs" },
              { value: "spend", label: "Highest spend" },
              { value: "cost", label: "Highest cost / PR" },
              { value: "name", label: "Name" },
            ]}
          >
            <SelectTrigger aria-label="Sort engineers">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="merged">Most merged PRs</SelectItem>
              <SelectItem value="spend">Highest spend</SelectItem>
              <SelectItem value="cost">Highest cost / PR</SelectItem>
              <SelectItem value="name">Name</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="overflow-hidden rounded-xl border">
        <Table>
          <TableHeader>
            <TableRow className="bg-muted/30">
              <TableHead className="pl-5">Engineer</TableHead>
              <TableHead className="text-right">Merged PRs</TableHead>
              <TableHead>Authored / agent</TableHead>
              <TableHead className="text-right">
                {comparison === "previous" ? "vs. previous" : "vs. last year"}
              </TableHead>
              <TableHead className="text-right">Median merge</TableHead>
              <TableHead className="text-right">Recorded spend</TableHead>
              <TableHead className="text-right">Spend / PR</TableHead>
              <TableHead>
                <span className="sr-only">Details</span>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {people.map((person) => {
              const current = person.periods.current;
              const baseline = person.periods[comparison];
              return (
                <TableRow key={person.email}>
                  <TableCell className="py-3 pl-5">
                    <button
                      className="flex items-center gap-3 rounded text-left hover:underline"
                      onClick={() => onSelect(person)}
                    >
                      <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium text-muted-foreground">
                        {person.name.slice(0, 2).toUpperCase()}
                      </span>
                      <span>
                        <span className="block font-medium">{person.name}</span>
                        <span className="text-xs text-muted-foreground">{person.email}</span>
                      </span>
                    </button>
                  </TableCell>
                  <TableCell className="text-right font-medium tabular-nums">{number(current.merged_prs)}</TableCell>
                  <TableCell>
                    <div className="flex w-28 items-center gap-2">
                      <div className="flex h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                        <span
                          className="bg-blue-500"
                          style={{
                            width: `${current.merged_prs ? (current.direct_authored / current.merged_prs) * 100 : 0}%`,
                          }}
                        />
                        <span
                          className="bg-violet-400"
                          style={{
                            width: `${current.merged_prs ? (current.declared_agent_owned / current.merged_prs) * 100 : 0}%`,
                          }}
                        />
                      </div>
                      <span className="text-[11px] text-muted-foreground">
                        {current.direct_authored}/{current.declared_agent_owned}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="text-right">
                    <Delta current={current.merged_prs} baseline={baseline.merged_prs} neutral />
                    <span className="ml-2 text-xs text-muted-foreground">({baseline.merged_prs})</span>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{duration(current.median_merge_hours)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {money(current.spend_observation === "no_records" ? null : current.gateway_recorded_spend)}
                  </TableCell>
                  <TableCell className="text-right font-medium tabular-nums">
                    {money(current.recorded_spend_per_attributed_pr)}
                  </TableCell>
                  <TableCell>
                    <Button
                      size="icon-xs"
                      variant="ghost"
                      aria-label={`View ${person.name}'s pull requests`}
                      onClick={() => onSelect(person)}
                    >
                      <ChevronRight />
                    </Button>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
        {people.length === 0 && (
          <div className="p-10 text-center text-sm text-muted-foreground">No engineers match “{query}”</div>
        )}
      </div>
      <p className="text-xs text-muted-foreground">
        <span className="mr-3 inline-flex items-center gap-1.5">
          <span className="size-2 rounded-sm bg-blue-500" />
          Authored
        </span>
        <span className="mr-3 inline-flex items-center gap-1.5">
          <span className="size-2 rounded-sm bg-violet-400" />
          Agent, explicit requester
        </span>
        Spend / PR is recorded period spend divided by matched PRs
      </p>
    </div>
  );
}

function Quality({ snapshot, comparison }: { snapshot: ObservedSnapshot; comparison: Comparison }) {
  const current = snapshot.periods.current;
  const baseline = snapshot.periods[comparison];
  const rows = [
    {
      label: "New bug-labeled issues",
      current: current.new_bug_labeled_issues,
      baseline: baseline.new_bug_labeled_issues,
      detail: "Opened during the period, with bug or kind:bug labels at collection",
    },
    {
      label: "New regression-labeled issues",
      current: current.new_regression_labeled_issues,
      baseline: baseline.new_regression_labeled_issues,
      detail: "Opened during the period and labeled as regressions",
    },
    {
      label: "Revert-titled PRs",
      current: current.explicitly_titled_revert_prs,
      baseline: baseline.explicitly_titled_revert_prs,
      detail: "Merged PRs whose titles explicitly indicate a revert",
    },
  ];
  return (
    <div className="space-y-4">
      <div className="rounded-xl border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="pl-5">Repository signal</TableHead>
              <TableHead className="text-right">Current</TableHead>
              <TableHead className="text-right">Comparison</TableHead>
              <TableHead className="pr-5 text-right">Change</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.label}>
                <TableCell className="py-5 pl-5">
                  <p className="font-medium">{row.label}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{row.detail}</p>
                </TableCell>
                <TableCell className="text-right font-medium">{row.current}</TableCell>
                <TableCell className="text-right text-muted-foreground">{row.baseline}</TableCell>
                <TableCell className="pr-5 text-right">
                  <Delta current={row.current} baseline={row.baseline} neutral />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      <p className="max-w-3xl text-xs leading-relaxed text-muted-foreground">
        These signals help check whether more shipping comes with more bugs. Labels and revert titles are incomplete
        proxies; they do not establish a change-failure rate or attribute bugs to an engineer.
      </p>
    </div>
  );
}

function Report({ snapshot }: { snapshot: ObservedSnapshot }) {
  const [comparison, setComparison] = useState<Comparison>("previous");
  const [connections, setConnections] = useState(false);
  const [coverage, setCoverage] = useState(false);
  const [person, setPerson] = useState<ObservedPerson | null>(null);
  const current = snapshot.periods.current;
  const baseline = snapshot.periods[comparison];
  const cost =
    current.spend_observation === "records_present" && current.matched_internal_prs > 0
      ? current.matched_users_recorded_spend / current.matched_internal_prs
      : null;
  const baselineCost =
    baseline.spend_observation === "records_present" && baseline.matched_internal_prs > 0
      ? baseline.matched_users_recorded_spend / baseline.matched_internal_prs
      : null;
  return (
    <Page className="mx-auto max-w-[1500px] gap-5 pb-10">
      <PageHeader>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <PageHeaderTitle>ROI Calculator</PageHeaderTitle>
              <Badge variant="secondary">Prototype</Badge>
            </div>
            <PageHeaderDescription>Are we shipping more, with fewer bugs, at a better cost?</PageHeaderDescription>
          </div>
          <Button variant="outline" onClick={() => setConnections(true)}>
            <Link2 />
            Connections
          </Button>
        </div>
      </PageHeader>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm">
          <Github className="size-4" />
          <span className="font-medium">{snapshot.repo}</span>
          <Badge variant="secondary">Validated snapshot</Badge>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="flex items-center gap-2 rounded-md border px-3 py-2 text-xs">
            <CalendarDays className="size-3.5 text-muted-foreground" />
            {dateRange(current.window)}
          </span>
          <Select
            value={comparison}
            onValueChange={(value) => {
              if (value) setComparison(value);
            }}
            items={[
              { value: "previous", label: "vs. previous 28 days" },
              { value: "last_year", label: "vs. same period last year" },
            ]}
          >
            <SelectTrigger aria-label="Comparison period">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="previous">vs. previous 28 days</SelectItem>
              <SelectItem value="last_year">vs. same period last year</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="grid grid-cols-1 divide-y rounded-xl border sm:grid-cols-2 lg:grid-cols-4 lg:divide-x lg:divide-y-0">
        <Metric
          label="Merged PRs"
          value={number(current.merged_prs)}
          current={current.merged_prs}
          baseline={baseline.merged_prs}
          detail={`${number(baseline.merged_prs)} in comparison · whole repository`}
        />
        <Metric
          label="Median time to merge"
          value={duration(current.median_merge_hours)}
          current={current.median_merge_hours ?? undefined}
          baseline={baseline.median_merge_hours ?? undefined}
          detail={`${duration(baseline.median_merge_hours)} in comparison · opened to merged`}
        />
        <Metric
          label="New bugs"
          value={number(current.new_bug_labeled_issues)}
          current={current.new_bug_labeled_issues}
          baseline={baseline.new_bug_labeled_issues}
          detail={`${number(baseline.new_bug_labeled_issues)} in comparison · bug-labeled issues`}
          onInfo={() => setCoverage(true)}
        />
        <Metric
          label="Recorded spend / matched PR"
          value={money(cost)}
          detail={
            baselineCost === null
              ? "No gateway records for comparison"
              : `${money(baselineCost)} in comparison · gateway only`
          }
          onInfo={() => setCoverage(true)}
        />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>
          {number(current.matched_internal_prs)} PRs matched to {snapshot.people.length} engineers ·{" "}
          {number(current.devin_without_requester)} agent PRs without a requester
        </span>
        <Button size="sm" variant="ghost" onClick={() => setCoverage(true)}>
          Data coverage
          <ChevronRight />
        </Button>
      </div>
      <Tabs defaultValue="people" className="gap-5">
        <PageTabsList>
          <PageTabsTrigger value="people">
            Engineers <span className="ml-1.5 text-muted-foreground">{snapshot.people.length}</span>
          </PageTabsTrigger>
          <PageTabsTrigger value="pulls">Pull requests</PageTabsTrigger>
          <PageTabsTrigger value="quality">Quality</PageTabsTrigger>
        </PageTabsList>
        <TabsContent value="people">
          <PeopleTable snapshot={snapshot} comparison={comparison} onSelect={setPerson} />
        </TabsContent>
        <TabsContent value="pulls" className="space-y-5">
          <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
            <ShippingTrend snapshot={snapshot} comparison={comparison} />
            <div className="flex flex-col justify-between rounded-xl border p-5">
              <div>
                <h2 className="text-sm font-medium">Behind the numbers</h2>
                <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                  {number(current.devin_authored + current.other_bot_authored)} of {number(current.merged_prs)} PRs were
                  authored by agents or bots. The mix of work changed.
                </p>
                <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
                  Human-authored median merge time:{" "}
                  <span className="font-medium text-foreground">
                    {duration(current.human_summary.median_merge_hours)}
                  </span>
                  , compared with {duration(baseline.human_summary.median_merge_hours)}.
                </p>
              </div>
              <div className="mt-5 flex flex-wrap items-center justify-between gap-2 border-t pt-4">
                <span className="text-xs text-muted-foreground">
                  {number(current.devin_without_requester)} agent PRs have no requester
                </span>
                <Button size="sm" variant="ghost" onClick={() => setCoverage(true)}>
                  Data coverage
                  <ChevronRight />
                </Button>
              </div>
            </div>
          </div>

          <p className="mb-4 text-xs text-muted-foreground">
            All repository PRs, including agent work without a known requester
          </p>
          <PullList pulls={snapshot.pulls.current} />
        </TabsContent>
        <TabsContent value="quality">
          <Quality snapshot={snapshot} comparison={comparison} />
        </TabsContent>
      </Tabs>
      <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-4 text-xs text-muted-foreground">
        <span>Comparing with {dateRange(baseline.window)} · All dates UTC</span>
        <button
          type="button"
          className="flex items-center gap-1.5 hover:text-foreground"
          onClick={() => setCoverage(true)}
        >
          <Info className="size-3.5" />
          How these metrics are calculated
        </button>
      </div>
      {connections && <ObservedConnections repo={snapshot.repo} onClose={() => setConnections(false)} />}
      {coverage && <CoverageDetails snapshot={snapshot} onClose={() => setCoverage(false)} />}
      {person && (
        <PersonDetails
          key={person.email}
          person={person}
          snapshot={snapshot}
          comparison={comparison}
          onClose={() => setPerson(null)}
        />
      )}
    </Page>
  );
}

export default function ObservedROIView({ accessToken }: { accessToken: string | null }) {
  const [snapshot, setSnapshot] = useState<ObservedSnapshot | null>(null);
  const [error, setError] = useState(false);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!accessToken) return;
    const controller = new AbortController();
    apiClient
      .get<unknown>("/roi-calculator/observed-preview", { accessToken, signal: controller.signal })
      .then((data) => {
        if (!controller.signal.aborted) {
          setSnapshot(observedSnapshotSchema.parse(data));
          setError(false);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) setError(true);
      });
    return () => controller.abort();
  }, [accessToken, retry]);
  if (snapshot) return <Report snapshot={snapshot} />;
  if (error)
    return (
      <Page>
        <PageHeaderTitle>ROI prototype unavailable</PageHeaderTitle>
        <p className="text-sm text-muted-foreground">
          This preview needs its private snapshot service. Check the preview server, then retry.
        </p>
        <Button
          className="w-fit"
          variant="outline"
          onClick={() => {
            setError(false);
            setRetry(retry + 1);
          }}
        >
          Retry
        </Button>
      </Page>
    );
  return (
    <Page>
      <Skeleton className="h-8 w-52" />
      <Skeleton className="h-32 w-full" />
      <Skeleton className="h-96 w-full" />
      <p className="text-sm text-muted-foreground">Loading validated repository and gateway data…</p>
    </Page>
  );
}
