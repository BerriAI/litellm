"use client";

import React from "react";
import { Bar, CartesianGrid, ComposedChart, Line, XAxis, YAxis } from "recharts";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { coverageLabel, peopleCsv, effortNote, estimateLabel, formatMoney, formatNumber } from "./roiCalculatorData";
import type { ROIPerson, ROIPull, ROISummary } from "./roiCalculatorData";

const CHART_CONFIG = {
  spend: { label: "Matched spend", color: "var(--chart-1)" },
  hours: { label: "Estimated hours", color: "var(--chart-2)" },
} satisfies ChartConfig;

export function ROIOverview({
  summary,
  pulls,
  query,
  onQueryChange,
  onSelectPull,
  onViewPeople,
}: {
  summary: ROISummary;
  pulls: ROIPull[];
  query: string;
  onQueryChange: (value: string) => void;
  onSelectPull: (pull: ROIPull) => void;
  onViewPeople: () => void;
}) {
  const [pagination, setPagination] = React.useState({ query, visibleCount: 10 });
  const visibleCount = pagination.query === query ? pagination.visibleCount : 10;
  const metrics = summary.metrics;
  return (
    <div className="space-y-6">
      <section aria-label="Spend and estimated engineering effort" className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard title="Spend per estimated engineering hour" value={formatMoney(metrics.cost_per_hour)} />
        <MetricCard title="Matched gateway spend" value={formatMoney(metrics.matched_spend)} />
        <MetricCard title="Estimated engineering hours" value={`${formatNumber(metrics.output_hours)} hrs`} />
        <MetricCard title="PR email coverage" value={coverageLabel(summary)} />
      </section>
      <p className="text-sm text-muted-foreground">
        {formatMoney(metrics.excluded_spend)} of {formatMoney(metrics.total_spend)} total gateway spend is excluded from
        the matched cohort.
      </p>
      <details className="rounded-lg border p-4 text-sm">
        <summary className="cursor-pointer font-medium">Calculation details</summary>
        <div className="space-y-3 pt-3 text-muted-foreground">
          <p>
            {metrics.cost_per_hour != null
              ? `${formatMoney(metrics.matched_spend)} gateway spend ÷ ${formatNumber(metrics.output_hours)} estimated engineering hours = ${formatMoney(metrics.cost_per_hour)} per estimated hour.`
              : "A rate is available when matched estimated hours are greater than zero."}
          </p>
          <p>
            The comparison includes {metrics.cohort_people} matched {metrics.cohort_people === 1 ? "person" : "people"}{" "}
            with complete PR estimates, for the same period in UTC. {metrics.matched_prs} of {metrics.merged_prs} PRs
            have email matches. {formatMoney(metrics.excluded_spend)} of {formatMoney(metrics.total_spend)} total
            gateway spend is excluded.
          </p>
          <p>
            Gateway spend includes all of each person’s usage, across repositories. This does not measure hours saved by
            AI or financial returns.
          </p>
          <Button variant="link" className="h-auto p-0" onClick={onViewPeople}>
            Review email matches
          </Button>
        </div>
      </details>

      <Card>
        <CardHeader>
          <CardTitle>Spend and estimated engineering effort</CardTitle>
          <CardDescription>
            Daily matched gateway spend and estimated engineering hours for the same UTC period
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={CHART_CONFIG} className="h-[320px] w-full">
            <ComposedChart data={summary.trend} margin={{ left: 8, right: 8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="date" tickLine={false} axisLine={false} minTickGap={36} />
              <YAxis yAxisId="spend" tickFormatter={(value) => formatMoney(Number(value))} />
              <YAxis yAxisId="hours" orientation="right" domain={[0, "auto"]} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <ChartLegend content={<ChartLegendContent />} />
              <Bar yAxisId="spend" dataKey="spend" fill="var(--color-spend)" isAnimationActive={false} />
              <Line
                yAxisId="hours"
                dataKey="hours"
                stroke="var(--color-hours)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ChartContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-4">
          <div>
            <CardTitle>Pull requests</CardTitle>
            <CardDescription>
              {metrics.merged_prs} merged · {metrics.estimated_prs} estimated · {metrics.pending_prs} need attention
            </CardDescription>
          </div>
          <Input
            aria-label="Search pull requests"
            className="w-full sm:max-w-xs"
            placeholder="Search pull requests"
            type="search"
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
          />
        </CardHeader>
        <CardContent className="space-y-4">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Pull request</TableHead>
                <TableHead className="text-right">Estimated hours</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {pulls.slice(0, visibleCount).map((pull) => (
                <TableRow key={`${pull.repo}#${pull.number}`}>
                  <TableCell>
                    <Button
                      aria-label={`Open estimate for ${pull.repo} pull request ${pull.number}`}
                      className="h-auto whitespace-normal p-0 text-left"
                      variant="link"
                      onClick={() => onSelectPull(pull)}
                    >
                      <span>
                        <span className="block font-medium">{pull.title}</span>
                        <span className="text-xs text-muted-foreground">
                          {pull.repo} #{pull.number} · {pull.login}
                        </span>
                      </span>
                    </Button>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{estimateLabel(pull.estimate)}</TableCell>
                </TableRow>
              ))}
              {pulls.length === 0 && (
                <TableRow>
                  <TableCell className="text-center text-muted-foreground" colSpan={2}>
                    {query ? "No matching pull requests." : "No merged pull requests in this period."}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
          {pulls.length > visibleCount && (
            <Button
              variant="link"
              className="px-0"
              onClick={() =>
                setPagination((current) => ({
                  query,
                  visibleCount: (current.query === query ? current.visibleCount : 10) + 25,
                }))
              }
            >
              Load more pull requests
            </Button>
          )}
          <Button variant="outline" onClick={onViewPeople}>
            Review email matches
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

function MetricCard({ title, value }: { title: string; value: string }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>{title}</CardDescription>
        <CardTitle className="break-words text-2xl tabular-nums">{value}</CardTitle>
      </CardHeader>
    </Card>
  );
}

export function ROIPeopleView({
  summary,
  identityMap,
  onMatch,
  readOnly = false,
}: {
  summary: ROISummary;
  identityMap: Record<string, string>;
  onMatch: (person: ROIPerson, login: string) => void;
  readOnly?: boolean;
}) {
  const exportCsv = () => {
    const url = URL.createObjectURL(new Blob([peopleCsv(summary)], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "litellm-roi.csv";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <div className="space-y-6">
      <div className="flex justify-end">
        <Button variant="outline" onClick={exportCsv}>
          Export CSV
        </Button>
      </div>
      <p className="text-sm leading-relaxed text-muted-foreground">
        {effortNote(summary.effort_basis)} Spend includes each person’s full gateway usage for this period. This does
        not measure hours saved by AI or financial returns.
      </p>
      <Card>
        <CardContent className="pt-6">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Person</TableHead>
                <TableHead className="text-right">Gateway spend</TableHead>
                <TableHead className="text-right">Estimated hours</TableHead>
                <TableHead className="text-right">Spend / estimated hour</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {summary.people.map((person) => (
                <TableRow key={person.id}>
                  <TableCell>
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
                      {person.match_methods.some(
                        (method) =>
                          ["manual", "commit email", "profile email"].includes(method) && person.spend != null,
                      ) ? (
                        <span className="text-xs text-emerald-700">Matched</span>
                      ) : (
                        <span className="text-xs text-muted-foreground">Unmatched</span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">{person.email || "Email unavailable"}</p>
                    {person.logins.some((login) => identityMap[login.toLowerCase()]) && (
                      <p className="text-xs text-muted-foreground">Manual email match</p>
                    )}
                    {!person.eligible && <p className="text-xs text-muted-foreground">Excluded from ratio</p>}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{formatMoney(person.spend)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {person.estimated_prs > 0 ? `${formatNumber(person.hours)} hrs` : "—"}
                    <p className="text-xs text-muted-foreground">
                      {person.prs} {person.prs === 1 ? "PR" : "PRs"}
                      {person.pending_prs > 0 ? ` · ${person.pending_prs} pending` : ""}
                    </p>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{formatMoney(person.cost_per_hour)}</TableCell>
                </TableRow>
              ))}
              {summary.people.length === 0 && (
                <TableRow>
                  <TableCell className="text-center text-muted-foreground" colSpan={4}>
                    No people in this period.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <details className="rounded-lg border p-4 text-sm">
        <summary className="cursor-pointer font-medium">How email matching works</summary>
        <p className="mt-3 text-muted-foreground">
          Matches use the author’s public GitHub email or commit emails associated with their GitHub account. Email
          matching ignores case. Private, noreply, and ambiguous emails stay unmatched. Manual matches take priority.
          People with no spend record or incomplete PR estimates are excluded from the ratio.
        </p>
      </details>
    </div>
  );
}
