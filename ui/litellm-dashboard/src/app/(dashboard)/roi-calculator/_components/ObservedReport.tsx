"use client";

import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, CalendarDays, ChevronDown, ChevronRight, Link2, Search, Users } from "lucide-react";
import { Page, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { PageHeader, PageHeaderTitle } from "@/components/shared/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import ObservedAccounts from "./ObservedAccounts";
import { MatchedPeopleToggle } from "./MatchedPeopleToggle";
import { BranchSpend, PersonDetails, PullList } from "./ObservedDetails";
import {
  change,
  dateRange,
  changeTerms,
  duration,
  money,
  number,
  visiblePeople,
  reportPeople,
  filterObservedPulls,
  weeklyMerges,
  type Comparison,
  type ReportPerson,
  type ObservedSnapshot,
  type PeopleSort,
} from "./observedData";

function Delta({
  current,
  baseline,
  neutral = false,
}: {
  current: number | null;
  baseline: number | null;
  neutral?: boolean;
}) {
  const delta = current === null || baseline === null ? null : change(current, baseline);
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
}: {
  label: string;
  value: string;
  detail: string;
  current?: number | null;
  baseline?: number | null;
}) {
  return (
    <div className="min-w-0 bg-background px-4 py-3">
      <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">{label}</div>
      <div className="mt-1 flex flex-wrap items-baseline gap-2">
        <span className="text-2xl font-semibold tracking-tight tabular-nums">{value}</span>
        {current !== undefined && baseline !== undefined && <Delta current={current} baseline={baseline} />}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">{detail}</p>
    </div>
  );
}

function ShippingTrend({ snapshot, comparison }: { snapshot: ObservedSnapshot; comparison: Comparison }) {
  const terms = changeTerms(snapshot.source_provider);
  const current = weeklyMerges(snapshot, "current");
  const baseline = weeklyMerges(snapshot, comparison);
  const max = Math.max(1, ...current, ...baseline);
  return (
    <div className="rounded-xl border p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-medium">Repository shipping activity</h2>
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
        className="mt-5 grid gap-2"
        style={{ gridTemplateColumns: `repeat(${current.length}, minmax(0, 1fr))` }}
        role="img"
        aria-label={`Merged ${terms.plural} by week. Current: ${current.join(", ")}. Comparison: ${baseline.join(", ")}`}
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
            <p className="mt-2 text-center text-[11px] text-muted-foreground">W{week + 1}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PeopleTable({
  rows,
  provider,
  matchedOnly,
  comparison,
  onSelect,
}: {
  rows: ReportPerson[];
  provider: ObservedSnapshot["source_provider"];
  matchedOnly: boolean;
  comparison: Comparison;
  onSelect: (person: ReportPerson) => void;
}) {
  const terms = changeTerms(provider);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<PeopleSort>("merged");
  const people = visiblePeople(rows, query, sort);
  const emptyMessage = matchedOnly
    ? "No matched people. Link accounts or turn off the filter to see all contributors"
    : "No contributors in these periods";
  return (
    <div className="space-y-3">
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
          <span className="text-xs text-muted-foreground">
            {people.length} {people.length === 1 ? "engineer" : "engineers"}
          </span>
          <Select
            value={sort}
            onValueChange={(value) => {
              if (value) setSort(value);
            }}
            items={[
              { value: "merged", label: `Most merged ${terms.plural}` },
              { value: "spend", label: "Highest spend" },
              { value: "cost", label: `Highest cost / ${terms.singular}` },
              { value: "name", label: "Name" },
            ]}
          >
            <SelectTrigger aria-label="Sort engineers">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="merged">Most merged {terms.plural}</SelectItem>
              <SelectItem value="spend">Highest spend</SelectItem>
              <SelectItem value="cost">Highest cost / {terms.singular}</SelectItem>
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
              <TableHead className="text-right">Merged {terms.plural}</TableHead>
              <TableHead>Authored / agent</TableHead>
              <TableHead className="text-right">
                {comparison === "previous" ? "vs. previous" : "vs. last year"}
              </TableHead>
              <TableHead className="text-right">Median merge</TableHead>
              <TableHead className="text-right">Recorded spend</TableHead>
              <TableHead className="text-right">Spend / {terms.singular}</TableHead>
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
                <TableRow key={person.id}>
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
                        <span className="text-xs text-muted-foreground">
                          {person.email || `Not linked · ${person.host}`}
                        </span>
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
                      aria-label={`View ${person.name}'s ${terms.lower}`}
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
          <div className="p-10 text-center text-sm text-muted-foreground">
            {query ? `No engineers match “${query}”` : emptyMessage}
          </div>
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
        Spend / {terms.singular} is recorded period spend divided by matched {terms.plural}
      </p>
    </div>
  );
}

function Quality({ snapshot, comparison }: { snapshot: ObservedSnapshot; comparison: Comparison }) {
  const terms = changeTerms(snapshot.source_provider);
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
      label: `Revert-titled ${terms.plural}`,
      current: current.explicitly_titled_revert_prs,
      baseline: baseline.explicitly_titled_revert_prs,
      detail: `Merged ${terms.plural} whose titles explicitly indicate a revert`,
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
                <TableCell className="text-right font-medium">{number(row.current)}</TableCell>
                <TableCell className="text-right text-muted-foreground">{number(row.baseline)}</TableCell>
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

function costPerChange(period: ObservedSnapshot["periods"]["current"]) {
  if (period.spend_observation !== "records_present" || period.matched_internal_prs === 0) return null;
  return period.matched_users_recorded_spend / period.matched_internal_prs;
}

export default function ObservedReport({
  snapshot,
  accessToken,
  readOnly,
  onRefresh,
  onConnect,
  actions,
  notice,
  syncing,
  onPeriod,
}: {
  snapshot: ObservedSnapshot;
  accessToken: string;
  readOnly: boolean;
  onRefresh: () => void;
  onConnect: () => void;
  actions: React.ReactNode;
  notice?: React.ReactNode;
  syncing: boolean;
  onPeriod?: (days: number) => void;
}) {
  const [comparison, setComparison] = useState<Comparison>("previous");
  const [activeTab, setActiveTab] = useState(
    snapshot.people.length && snapshot.periods.current.merged_prs > 0 ? "people" : "pulls",
  );
  const [accountEmail, setAccountEmail] = useState<string | null>(null);
  const [matchedOnly, setMatchedOnly] = useState(true);
  const people = useMemo(() => reportPeople(snapshot, matchedOnly), [snapshot, matchedOnly]);
  const pulls = useMemo(() => filterObservedPulls(snapshot, "current", matchedOnly), [snapshot, matchedOnly]);
  const [personId, setPersonId] = useState<string | null>(null);
  const person = people.find((entry) => entry.id === personId) ?? null;
  const terms = changeTerms(snapshot.source_provider);
  const current = snapshot.periods.current;
  const baseline = snapshot.periods[comparison];
  const days = Math.round((Date.parse(current.window.end) - Date.parse(current.window.start)) / 86400000) + 1;
  const rangeOptions = [...new Set([7, 28, 90, days])].sort((a, b) => a - b);
  const cost = costPerChange(current);
  const baselineCost = costPerChange(baseline);
  return (
    <Page className="mx-auto max-w-[1500px] gap-3 pb-10 sm:pt-4">
      <PageHeader className="flex flex-wrap items-center justify-between gap-3">
        <PageHeaderTitle className="text-xl">ROI Calculator</PageHeaderTitle>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          {actions}
          {!readOnly && (
            <Button size="sm" variant="outline" onClick={onConnect}>
              <Link2 />
              Connections
            </Button>
          )}
        </div>
      </PageHeader>
      {notice}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Popover>
          <PopoverTrigger render={<Button size="sm" variant="outline" className="self-start" />}>
            {number(snapshot.repos.length)} {snapshot.repos.length === 1 ? "repository" : "repositories"}
            <ChevronDown className="size-3.5" />
          </PopoverTrigger>
          <PopoverContent align="start" className="w-80 max-w-[calc(100vw-2rem)] gap-2 p-3">
            <PopoverTitle>Repositories</PopoverTitle>
            <ul className="max-h-64 space-y-2 overflow-y-auto text-xs break-words">
              {snapshot.repos.map((repo) => (
                <li key={repo}>{repo}</li>
              ))}
            </ul>
          </PopoverContent>
        </Popover>
        <div className="flex flex-wrap items-center gap-2">
          <Select
            value={String(days)}
            disabled={!onPeriod || syncing}
            onValueChange={(value) => {
              if (value && Number(value) !== days) onPeriod?.(Number(value));
            }}
            items={rangeOptions.map((value) => ({ value: String(value), label: `Last ${value} days` }))}
          >
            <SelectTrigger size="sm" aria-label="Reporting period">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {rangeOptions.map((value) => (
                <SelectItem key={value} value={String(value)}>
                  Last {value} days
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <span className="flex h-8 items-center gap-2 text-xs text-muted-foreground">
            <CalendarDays className="size-3.5 text-muted-foreground" />
            {dateRange(current.window)}
          </span>
          <Select
            value={comparison}
            onValueChange={(value) => {
              if (value) setComparison(value);
            }}
            items={[
              { value: "previous", label: "vs. previous period" },
              { value: "last_year", label: "vs. same period last year" },
            ]}
          >
            <SelectTrigger size="sm" aria-label="Comparison period">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="previous">vs. previous period</SelectItem>
              <SelectItem value="last_year">vs. same period last year</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border bg-border lg:grid-cols-4">
        <Metric
          label={`Merged ${terms.plural}`}
          value={number(current.merged_prs)}
          current={current.merged_prs}
          baseline={baseline.merged_prs}
          detail={`${number(baseline.merged_prs)} in comparison`}
        />
        <Metric
          label="Median time to merge"
          value={current.merged_prs === 0 ? "No merges" : duration(current.median_merge_hours)}
          current={current.median_merge_hours ?? undefined}
          baseline={baseline.median_merge_hours ?? undefined}
          detail={`${duration(baseline.median_merge_hours)} in comparison`}
        />
        <Metric
          label="New bugs"
          value={number(current.new_bug_labeled_issues)}
          current={current.new_bug_labeled_issues}
          baseline={baseline.new_bug_labeled_issues}
          detail={`${number(baseline.new_bug_labeled_issues)} in comparison · bug-labeled issues`}
        />
        <Metric
          label={`Recorded spend / matched ${terms.singular}`}
          value={money(cost)}
          detail={
            baselineCost === null
              ? "No gateway records for comparison"
              : `${money(baselineCost)} in comparison · gateway only`
          }
        />
      </div>
      <Tabs value={activeTab} onValueChange={setActiveTab} className="gap-3">
        <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-2 border-b">
          <PageTabsList className="min-w-0 flex-1 basis-full gap-4 border-0 xl:basis-auto">
            <PageTabsTrigger value="people">
              Engineers <span className="ml-1.5 text-muted-foreground">{people.length}</span>
            </PageTabsTrigger>
            <PageTabsTrigger value="pulls">{terms.requests}</PageTabsTrigger>
            <PageTabsTrigger value="quality">Quality</PageTabsTrigger>
            <PageTabsTrigger value="branches">Branch spend</PageTabsTrigger>
          </PageTabsList>
          {activeTab !== "quality" && <MatchedPeopleToggle checked={matchedOnly} onChange={setMatchedOnly} />}
          {!readOnly && (
            <Button
              size="sm"
              variant="outline"
              aria-label="Link accounts"
              title="Link accounts"
              onClick={() => setAccountEmail("")}
            >
              <Users />
              <span className="hidden sm:inline">Link accounts</span>
            </Button>
          )}
        </div>
        <TabsContent value="people">
          <PeopleTable
            rows={people}
            provider={snapshot.source_provider}
            matchedOnly={matchedOnly}
            comparison={comparison}
            onSelect={(selected) => setPersonId(selected.id)}
          />
        </TabsContent>
        <TabsContent value="pulls" className="space-y-5">
          {current.merged_prs > 0 || baseline.merged_prs > 0 ? (
            <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
              <ShippingTrend snapshot={snapshot} comparison={comparison} />
              <div className="flex flex-col justify-between rounded-xl border p-5">
                <div>
                  <h2 className="text-sm font-medium">Behind the numbers</h2>
                  <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                    {number(current.agent_authored)} of {number(current.merged_prs)} {terms.plural} were authored by
                    agents or bots.
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
                    {number(current.agents_without_requester)} agent {terms.plural} have no requester
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border p-8 text-center">
              <h2 className="text-base font-medium">No merged changes yet</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Your repositories are connected. New activity will appear after the next sync
              </p>
            </div>
          )}

          <p className="mb-4 text-xs text-muted-foreground">
            {matchedOnly
              ? "Changes from people linked to internal accounts"
              : `All repository ${terms.plural}, including agent work without a known requester`}
          </p>
          <PullList pulls={pulls} provider={snapshot.source_provider} matchedOnly={matchedOnly} />
        </TabsContent>
        <TabsContent value="quality">
          <Quality snapshot={snapshot} comparison={comparison} />
        </TabsContent>
        <TabsContent value="branches">
          <BranchSpend snapshot={snapshot} matchedOnly={matchedOnly} />
        </TabsContent>
      </Tabs>
      <div className="flex flex-wrap items-center justify-between gap-2 border-t pt-3 text-xs text-muted-foreground">
        <span className="w-full">
          {number(current.matched_internal_prs)} {terms.plural} matched to {snapshot.people.length} engineers ·{" "}
          {number(current.agents_without_requester)} agent {terms.plural} without a requester
        </span>
        <span>Comparing with {dateRange(baseline.window)} · All dates UTC</span>
        <span>Spend recorded by this gateway · Merge time is elapsed time, not effort</span>
      </div>
      {accountEmail !== null && (
        <ObservedAccounts
          accessToken={accessToken}
          people={snapshot.people}
          initialEmail={accountEmail}
          onClose={() => setAccountEmail(null)}
          onSaved={onRefresh}
        />
      )}
      {person && (
        <PersonDetails
          key={person.id}
          person={person}
          snapshot={snapshot}
          comparison={comparison}
          onClose={() => setPersonId(null)}
          onEdit={
            readOnly || !person.matched
              ? undefined
              : () => {
                  setAccountEmail(person.email);
                  setPersonId(null);
                }
          }
        />
      )}
    </Page>
  );
}
