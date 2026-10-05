"use client";

import React from "react";
import { ChevronDown, Download, GitBranch, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  peopleCsv,
  isMatchedPerson,
  effortNote,
  estimateLabel,
  formatMoney,
  formatNumber,
  branchCostLabel,
  highestCostPulls,
} from "./roiCalculatorData";
import type { ROIPerson, ROIPull, ROISummary } from "./roiCalculatorData";

type PullSelection = { summary: ROISummary; onSelectPull: (pull: ROIPull) => void };

export function ROIOverview({
  summary,
  onSelectPull,
  onViewPeople,
  onViewBranches,
}: PullSelection & {
  onViewPeople: () => void;
  onViewBranches: () => void;
}) {
  const metrics = summary.metrics;
  const branches = summary.branch_metrics;
  const topPulls = highestCostPulls(summary.pulls);
  return (
    <div className="space-y-8">
      <div className="@container overflow-hidden rounded-xl border">
        <dl aria-label="Report overview" className="grid grid-cols-2 @min-[760px]:grid-cols-4">
          <MetricCard
            title="Gateway AI cost"
            value={formatMoney(metrics.total_spend)}
            description="All gateway usage in this period"
            primary
          />
          <MetricCard
            title="Estimated effort"
            value={`${formatNumber(metrics.total_output_hours)} hrs`}
            description={summary.effort_basis === "without_ai" ? "Estimated without AI" : "Check estimate assumptions"}
          />
          <MetricCard
            title="Merged changes"
            value={formatNumber(metrics.merged_prs)}
            description={`${metrics.estimated_prs} estimated`}
          />
          <MetricCard
            title="People"
            value={formatNumber(metrics.people_with_prs)}
            description="Contributors to merged work"
          />
        </dl>
      </div>
      <section aria-label="Cost coverage" className="space-y-4">
        <div className="space-y-1">
          <h2 className="text-base font-semibold">Where AI costs are matched</h2>
          <p className="text-sm text-muted-foreground">
            People use gateway account costs. Branches use tagged requests for these repositories.
          </p>
        </div>
        <div className="overflow-hidden rounded-xl border">
          <Table className="min-w-[600px]">
            <TableHeader className="bg-muted/40">
              <TableRow className="hover:bg-transparent">
                <TableHead className="px-4 text-xs">View</TableHead>
                <TableHead className="px-4 text-right text-xs">Matched cost</TableHead>
                <TableHead className="px-4 text-right text-xs">Unmatched cost</TableHead>
                <TableHead className="px-4 text-right text-xs">Changes matched</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell className="px-4 py-3">
                  <Button variant="link" className="h-auto p-0" onClick={onViewPeople}>
                    People
                  </Button>
                </TableCell>
                <TableCell className="px-4 py-3 text-right font-medium tabular-nums">
                  {formatMoney(metrics.matched_spend)}
                </TableCell>
                <TableCell className="px-4 py-3 text-right text-muted-foreground tabular-nums">
                  {formatMoney(metrics.excluded_spend)}
                </TableCell>
                <TableCell className="px-4 py-3 text-right tabular-nums">
                  {metrics.matched_prs} / {metrics.merged_prs}
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell className="px-4 py-3">
                  <Button variant="link" className="h-auto p-0" onClick={onViewBranches}>
                    Branches
                  </Button>
                </TableCell>
                <TableCell className="px-4 py-3 text-right font-medium tabular-nums">
                  {formatMoney(branches?.spend)}
                </TableCell>
                <TableCell className="px-4 py-3 text-right text-muted-foreground tabular-nums">
                  {formatMoney(branches?.unlinked_spend)}
                </TableCell>
                <TableCell className="px-4 py-3 text-right tabular-nums">
                  {branches?.matched_pulls ?? 0} / {metrics.merged_prs}
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </div>
      </section>
      <ROIPulls
        summary={summary}
        pulls={topPulls}
        onSelectPull={onSelectPull}
        compact
        onViewBranches={onViewBranches}
      />
    </div>
  );
}

export function ROIBranches({
  summary,
  onSelectPull,
  pulls,
  query,
  onQueryChange,
  matchedOnly,
}: PullSelection & {
  pulls: ROIPull[];
  query: string;
  matchedOnly: boolean;
  onQueryChange: (query: string) => void;
}) {
  return (
    <div className="space-y-8">
      <section aria-label="Branch cost analysis" className="space-y-4">
        <div className="@container overflow-hidden rounded-xl border">
          <ROIMetrics summary={summary} branchMode />
          <ROIComparison summary={summary} branchMode />
        </div>
      </section>
      <ROIPulls
        summary={summary}
        pulls={pulls}
        query={query}
        matchedOnly={matchedOnly}
        onQueryChange={onQueryChange}
        onSelectPull={onSelectPull}
      />
    </div>
  );
}

function ROIPulls({
  summary,
  onSelectPull,
  pulls,
  query = "",
  onQueryChange,
  compact = false,
  matchedOnly = false,
  onViewBranches,
}: PullSelection & {
  pulls: ROIPull[];
  query?: string;
  onQueryChange?: (query: string) => void;
  compact?: boolean;
  matchedOnly?: boolean;
  onViewBranches?: () => void;
}) {
  const changeName = summary.source_provider === "gitlab" ? "merge request" : "pull request";
  const [pagination, setPagination] = React.useState({ query, visibleCount: 10 });
  const visibleCount = pagination.query === query ? pagination.visibleCount : 10;
  const metrics = summary.metrics;
  const emptyMessage = compact
    ? "No merged changes with tagged costs yet. Open Branches to see how to add tags."
    : `No merged ${changeName}s ${matchedOnly ? "from matched people " : ""}in this period.`;
  return (
    <section aria-label={`Merged ${changeName}s`} className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-base font-semibold">{compact ? "Highest-cost changes" : "Costs by branch"}</h2>
          <p className="text-sm text-muted-foreground">
            {compact
              ? "Merged work ranked by tagged AI cost"
              : `${pulls.length} of ${metrics.merged_prs} ${changeName}s`}
          </p>
        </div>
        {compact ? (
          <Button variant="outline" onClick={onViewBranches}>
            View all branches
          </Button>
        ) : (
          <div className="relative w-full sm:w-64">
            <Search
              aria-hidden="true"
              className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              aria-label={`Search ${changeName}s`}
              className="pl-9"
              placeholder={`Search ${changeName}s`}
              type="search"
              value={query}
              onChange={(event) => onQueryChange?.(event.target.value)}
            />
          </div>
        )}
      </div>
      <div className="overflow-hidden rounded-xl border">
        <Table className="min-w-[600px] table-fixed">
          <TableHeader className="bg-muted/40">
            <TableRow className="hover:bg-transparent">
              <TableHead className="w-3/5 px-4 text-xs text-muted-foreground">
                {changeName === "merge request" ? "Merge request" : "Pull request"}
              </TableHead>
              <TableHead className="px-4 text-right text-xs text-muted-foreground">AI cost</TableHead>
              <TableHead className="px-4 text-right text-xs text-muted-foreground">Estimated effort</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {pulls.slice(0, visibleCount).map((pull) => (
              <TableRow key={`${pull.repo}#${pull.number}`}>
                <TableCell className="whitespace-normal px-4 py-3">
                  <Button
                    aria-label={`Open estimate for ${pull.repo} ${changeName} ${pull.number}`}
                    className="h-auto w-full justify-start whitespace-normal p-0 text-left"
                    variant="link"
                    onClick={() => onSelectPull(pull)}
                  >
                    <span className="min-w-0 space-y-1">
                      <span className="block break-words font-medium leading-5">{pull.title}</span>
                      <span className="block break-all text-xs font-normal text-muted-foreground">
                        {pull.repo} #{pull.number} · {pull.login}
                      </span>
                      {!compact && pull.source_branch && (
                        <span className="flex items-start gap-1.5 text-xs font-normal text-muted-foreground">
                          <GitBranch aria-hidden="true" className="mt-0.5 size-3 shrink-0" />
                          <span className="break-all">{pull.source_branch}</span>
                        </span>
                      )}
                    </span>
                  </Button>
                </TableCell>
                <TableCell
                  className={`px-4 py-3 text-right tabular-nums ${pull.branch_cost?.status === "matched" ? "font-medium" : "whitespace-normal text-xs text-muted-foreground"}`}
                >
                  {branchCostLabel(pull)}
                </TableCell>
                <TableCell className="px-4 py-3 text-right tabular-nums">{estimateLabel(pull.estimate)}</TableCell>
              </TableRow>
            ))}
            {pulls.length === 0 && (
              <TableRow>
                <TableCell className="h-32 text-center text-muted-foreground" colSpan={3}>
                  {query ? `No matching ${changeName}s. Try another search.` : emptyMessage}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        {pulls.length > visibleCount && (
          <div className="flex items-center justify-between gap-4 border-t px-4 py-3">
            <p className="text-xs text-muted-foreground">
              Showing {Math.min(visibleCount, pulls.length)} of {pulls.length}
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                setPagination((current) => ({
                  query,
                  visibleCount: (current.query === query ? current.visibleCount : 10) + 25,
                }))
              }
            >
              Load more {changeName}s
            </Button>
          </div>
        )}
      </div>
    </section>
  );
}

function ROIMetrics({ summary, branchMode }: { summary: ROISummary; branchMode: boolean }) {
  const branches = summary.branch_metrics;
  const metrics = summary.metrics;
  return (
    <dl aria-label="Spend and estimated engineering effort" className="grid grid-cols-2 @min-[760px]:grid-cols-4">
      <MetricCard
        title="Cost / estimated hour"
        value={formatMoney(branchMode ? branches?.cost_per_hour : metrics.cost_per_hour)}
        description="AI cost ÷ estimated effort"
        primary
      />
      <MetricCard
        title="Matched AI costs"
        value={formatMoney(branchMode ? branches?.spend : metrics.matched_spend)}
        description={branchMode ? "Recorded branch requests" : "Matched gateway accounts"}
      />
      <MetricCard
        title="Estimated effort"
        value={`${formatNumber(branchMode ? branches?.hours : metrics.output_hours)} hrs`}
        description={summary.effort_basis === "without_ai" ? "Estimated without AI" : "Check estimate assumptions"}
      />
      <MetricCard
        title={branchMode ? "Branch coverage" : "Email coverage"}
        value={`${branchMode ? branches?.matched_pulls ?? 0 : metrics.matched_prs} / ${metrics.merged_prs}`}
        description={branchMode ? "Changes with recorded costs" : "Changes with email matches"}
      />
    </dl>
  );
}

function MetricCard({
  title,
  value,
  description,
  primary = false,
}: {
  title: string;
  value: string;
  description: string;
  primary?: boolean;
}) {
  return (
    <div className={`min-w-0 space-y-2 p-4 sm:p-5 ${primary ? "bg-muted/30" : ""}`}>
      <dt className="min-h-8 text-xs font-medium text-muted-foreground @min-[360px]:min-h-0">{title}</dt>
      <dd className="whitespace-nowrap text-xl font-semibold tracking-tight tabular-nums @min-[600px]:text-2xl">
        {value}
      </dd>
      <dd className="text-xs leading-5 text-muted-foreground">{description}</dd>
    </div>
  );
}

function ROIComparison({
  summary,
  branchMode,
  onViewPeople,
}: {
  summary: ROISummary;
  branchMode: boolean;
  onViewPeople?: () => void;
}) {
  const branches = summary.branch_metrics;
  const metrics = summary.metrics;
  const unavailableRate =
    metrics.output_hours > 0
      ? "Spend per estimated hour is unavailable until all selected repositories can be read."
      : "Match gateway accounts to calculate costs per estimated hour.";
  return (
    <div className="border-t">
      {!branchMode && metrics.cohort_people === 0 && onViewPeople && (
        <div className="flex flex-wrap items-center justify-between gap-3 border-b px-5 py-3">
          <p className="text-sm text-muted-foreground">Match people to gateway accounts to see their AI costs.</p>
          <Button variant="outline" size="sm" onClick={onViewPeople}>
            Match people
          </Button>
        </div>
      )}
      <details className="group">
        <summary className="flex cursor-pointer list-none flex-wrap items-center gap-x-3 gap-y-1 px-5 py-3 text-xs [&::-webkit-details-marker]:hidden">
          <span className="flex items-center gap-2 font-medium">
            <ChevronDown aria-hidden="true" className="size-3.5 shrink-0 transition-transform group-open:rotate-180" />
            How this is calculated
          </span>
          <span className="text-muted-foreground sm:ml-auto">
            {formatMoney(branchMode ? branches?.unlinked_spend : metrics.excluded_spend)}{" "}
            {branchMode ? "in unmatched costs" : "excluded from calculation"}
          </span>
        </summary>
        <div className="space-y-3 border-t bg-muted/20 px-5 py-4 text-sm leading-6 text-muted-foreground">
          <p>{effortNote(summary.effort_basis)}</p>
          {branchMode ? (
            <>
              <p>
                Only branches with matched request costs and complete effort estimates enter the calculation. Costs
                cover retained requests in this report’s UTC dates, not the branch’s lifetime.
              </p>
              <p>
                Open a change below to find its repository and branch tags. Send both with each gateway request. Email
                matching is not required. Shared branches stay ambiguous so their costs are not counted twice.
              </p>
              <p>
                {formatMoney(branches?.total_tagged_spend)} in tagged costs was found for these repositories. Costs
                without a unique, fully estimated change stay unmatched.
              </p>
              {(summary.unlinked_branches?.length ?? 0) > 0 && (
                <div className="space-y-2 border-t pt-3">
                  <h3 className="font-medium text-foreground">Unmatched branches</h3>
                  <ul className="divide-y">
                    {summary.unlinked_branches?.map((row) => (
                      <li
                        key={`${row.repo}:${row.branch}`}
                        className="flex items-start justify-between gap-4 py-2 text-xs"
                      >
                        <span className="min-w-0 break-all">
                          {row.repo}
                          <br />
                          {row.branch}
                        </span>
                        <span className="shrink-0 tabular-nums">{formatMoney(row.spend)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          ) : (
            <>
              <p>
                {metrics.cost_per_hour != null
                  ? `${formatMoney(metrics.matched_spend)} AI costs ÷ ${formatNumber(metrics.output_hours)} estimated hours = ${formatMoney(metrics.cost_per_hour)} per estimated hour.`
                  : unavailableRate}
              </p>
              <p>
                Includes {metrics.cohort_people} matched {metrics.cohort_people === 1 ? "person" : "people"} with
                complete estimates. Costs include each person’s full gateway usage across repositories during this UTC
                period.
              </p>
              {onViewPeople && (
                <Button variant="link" className="h-auto p-0" onClick={onViewPeople}>
                  Review email matches
                </Button>
              )}
            </>
          )}
        </div>
      </details>
    </div>
  );
}

export function ROIPeopleView({
  summary,
  identityMap,
  onMatch,
  readOnly = false,
  matchedOnly = true,
}: {
  summary: ROISummary;
  identityMap: Record<string, string>;
  onMatch: (person: ROIPerson, login: string) => void;
  readOnly?: boolean;
  matchedOnly?: boolean;
}) {
  const people = matchedOnly ? summary.people.filter(isMatchedPerson) : summary.people;
  const exportCsv = () => {
    const url = URL.createObjectURL(new Blob([peopleCsv({ ...summary, people })], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "litellm-roi.csv";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <div className="space-y-6">
      <div className="@container overflow-hidden rounded-xl border">
        <ROIMetrics summary={summary} branchMode={false} />
        <ROIComparison summary={summary} branchMode={false} />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-base font-semibold">People and account matches</h2>
          <p className="text-sm text-muted-foreground">Select a person to match their gateway email.</p>
        </div>
        <Button variant="outline" onClick={exportCsv}>
          <Download />
          Export CSV
        </Button>
      </div>
      <div className="overflow-hidden rounded-xl border">
        <Table className="min-w-[640px]">
          <TableHeader className="bg-muted/40">
            <TableRow className="hover:bg-transparent">
              <TableHead className="px-4 text-xs text-muted-foreground">Person</TableHead>
              <TableHead className="px-4 text-right text-xs text-muted-foreground">AI cost</TableHead>
              <TableHead className="px-4 text-right text-xs text-muted-foreground">Estimated effort</TableHead>
              <TableHead className="px-4 text-right text-xs text-muted-foreground">Cost / est. hour</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {people.map((person) => (
              <TableRow key={person.id}>
                <TableCell className="px-4 py-3">
                  <div className="flex flex-wrap items-center gap-2">
                    {person.logins.length ? (
                      person.logins.map((login) =>
                        readOnly ? (
                          <span key={login}>{login}</span>
                        ) : (
                          <Button
                            key={login}
                            variant="link"
                            className="h-auto p-0"
                            onClick={() => onMatch(person, login)}
                          >
                            {login}
                          </Button>
                        ),
                      )
                    ) : (
                      <span>Unassigned gateway spend</span>
                    )}
                    <span className="text-xs text-muted-foreground">
                      {isMatchedPerson(person) ? "Matched" : "Unmatched"}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {person.email || "No public email"}
                    {person.logins.some((login) => identityMap[login.toLowerCase()]) ? " · Manual match" : ""}
                  </p>
                </TableCell>
                <TableCell className="px-4 py-3 text-right tabular-nums">{formatMoney(person.spend)}</TableCell>
                <TableCell className="px-4 py-3 text-right tabular-nums">
                  {person.estimated_prs > 0 ? `${formatNumber(person.hours)} hrs` : "—"}
                  <p className="mt-1 text-xs text-muted-foreground">
                    {person.prs} {person.prs === 1 ? "change" : "changes"}
                    {person.pending_prs > 0 ? ` · ${person.pending_prs} pending` : ""}
                  </p>
                </TableCell>
                <TableCell className="px-4 py-3 text-right tabular-nums">
                  {formatMoney(person.cost_per_hour)}
                  {!person.eligible && <p className="mt-1 text-xs text-muted-foreground">Not included</p>}
                </TableCell>
              </TableRow>
            ))}
            {people.length === 0 && (
              <TableRow>
                <TableCell className="h-32 text-center text-muted-foreground" colSpan={4}>
                  {matchedOnly
                    ? "No matched people in this period. Turn off the filter to see all contributors"
                    : "No people in this period"}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      <details className="group rounded-xl border">
        <summary className="flex cursor-pointer list-none items-center gap-2 px-5 py-3 text-xs font-medium [&::-webkit-details-marker]:hidden">
          <ChevronDown aria-hidden="true" className="size-3.5 transition-transform group-open:rotate-180" />
          How email matching works
        </summary>
        <div className="space-y-3 border-t px-5 py-4 text-sm leading-6 text-muted-foreground">
          <p>
            Matches use the author’s public profile email
            {summary.source_provider === "gitlab"
              ? "."
              : " or commit emails associated with their GitHub account."}{" "}
            Private, noreply, and ambiguous emails stay unmatched. Manual matches take priority.
          </p>
          <p>
            Costs include each person’s full gateway usage for this period. People without a spend record or with
            incomplete estimates are not included in the calculation.
          </p>
          <p>{effortNote(summary.effort_basis)}</p>
        </div>
      </details>
    </div>
  );
}
