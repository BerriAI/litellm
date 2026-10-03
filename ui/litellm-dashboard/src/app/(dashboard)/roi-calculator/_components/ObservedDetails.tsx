"use client";

import { useState } from "react";
import { ExternalLink, GitPullRequest } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  dateRange,
  duration,
  money,
  number,
  type Comparison,
  type ObservedPerson,
  type ObservedPull,
  type ObservedSnapshot,
  type Period,
} from "./observedData";

export function PullList({ pulls }: { pulls: ObservedPull[] }) {
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(20);
  const filtered = pulls.filter((pull) =>
    `${pull.number} ${pull.title} ${pull.author}`.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <div className="space-y-3">
      <Input
        aria-label="Search pull requests"
        placeholder="Search pull requests…"
        value={query}
        onChange={(event) => {
          setQuery(event.target.value);
          setLimit(20);
        }}
        className="max-w-sm"
      />
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Pull request</TableHead>
            <TableHead>Author</TableHead>
            <TableHead className="text-right">Time to merge</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {filtered.slice(0, limit).map((pull) => (
            <TableRow key={pull.number}>
              <TableCell className="max-w-lg whitespace-normal py-3">
                <a
                  className="group flex items-start gap-2 hover:underline"
                  href={pull.url}
                  target="_blank"
                  rel="noreferrer"
                >
                  <GitPullRequest className="mt-0.5 size-4 shrink-0 text-violet-500" />
                  <span>
                    <span className="mr-2 text-muted-foreground">#{pull.number}</span>
                    {pull.title}
                  </span>
                  <ExternalLink className="mt-0.5 size-3 shrink-0 text-muted-foreground" />
                </a>
              </TableCell>
              <TableCell>
                <span className="text-xs">{pull.author || "Deleted author"}</span>
                {pull.agent && (
                  <Badge variant="secondary" className="ml-2">
                    Agent
                  </Badge>
                )}
              </TableCell>
              <TableCell className="text-right tabular-nums">{duration(pull.merge_hours)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {filtered.length === 0 && (
        <p className="py-8 text-center text-sm text-muted-foreground">No pull requests match this search</p>
      )}
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>
          {number(Math.min(limit, filtered.length))} of {number(filtered.length)} pull requests
        </span>
        {limit < filtered.length && (
          <Button variant="outline" size="sm" onClick={() => setLimit(limit + 40)}>
            Show more
          </Button>
        )}
      </div>
    </div>
  );
}

export function PersonDetails({
  person,
  snapshot,
  comparison,
  onClose,
}: {
  person: ObservedPerson;
  snapshot: ObservedSnapshot;
  comparison: Comparison;
  onClose: () => void;
}) {
  const [period, setPeriod] = useState<Period>("current");
  const current = person.periods.current;
  const baseline = person.periods[comparison];
  const numbers = new Set(person.periods[period].pr_numbers);
  const pulls = snapshot.pulls[period].filter((pull) => numbers.has(pull.number));
  return (
    <Sheet
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <SheetContent className="w-full overflow-y-auto p-6 sm:max-w-3xl">
        <SheetHeader className="p-0 pr-8">
          <SheetTitle className="text-xl">{person.name}</SheetTitle>
          <SheetDescription>
            {person.email} · {person.github_logins.join(", ")}
          </SheetDescription>
        </SheetHeader>
        <div className="mt-2 grid grid-cols-3 divide-x rounded-lg border py-5">
          <div className="px-4">
            <p className="text-xs text-muted-foreground">Merged PRs</p>
            <p className="mt-2 text-2xl font-semibold">{number(current.merged_prs)}</p>
            <p className="mt-1 text-xs text-muted-foreground">{number(baseline.merged_prs)} in comparison</p>
          </div>
          <div className="px-4">
            <p className="text-xs text-muted-foreground">Recorded spend</p>
            <p className="mt-2 text-2xl font-semibold">
              {money(current.spend_observation === "no_records" ? null : current.gateway_recorded_spend)}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">Gateway only</p>
          </div>
          <div className="px-4">
            <p className="text-xs text-muted-foreground">Spend / matched PR</p>
            <p className="mt-2 text-2xl font-semibold">{money(current.recorded_spend_per_attributed_pr)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Period average</p>
          </div>
        </div>
        <p className="text-xs leading-relaxed text-muted-foreground">
          {current.direct_authored} authored directly · {current.declared_agent_owned} agent PRs with an explicit
          requester. Spend can include other repositories. Exact PR cost is unavailable because these requests have no
          repository or branch tags.
        </p>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <h3 className="font-medium">Merged pull requests</h3>
          <div className="flex gap-1 rounded-lg bg-muted p-1">
            <Button size="sm" variant={period === "current" ? "outline" : "ghost"} onClick={() => setPeriod("current")}>
              Current
            </Button>
            <Button
              size="sm"
              variant={period === comparison ? "outline" : "ghost"}
              onClick={() => setPeriod(comparison)}
            >
              {comparison === "previous" ? "Previous period" : "Last year"}
            </Button>
          </div>
        </div>
        <p className="text-xs text-muted-foreground">{dateRange(snapshot.periods[period].window)} · UTC</p>
        <PullList key={period} pulls={pulls} />
      </SheetContent>
    </Sheet>
  );
}

export function CoverageDetails({ snapshot, onClose }: { snapshot: ObservedSnapshot; onClose: () => void }) {
  const current = snapshot.periods.current;
  return (
    <Sheet
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <SheetContent className="overflow-y-auto p-6 sm:max-w-lg">
        <SheetHeader className="p-0 pr-6">
          <SheetTitle className="text-xl">About this data</SheetTitle>
          <SheetDescription>Observed activity, with the coverage made explicit</SheetDescription>
        </SheetHeader>
        <div className="space-y-6 py-3 text-sm leading-relaxed">
          <section>
            <h3 className="mb-2 font-medium">Repository activity</h3>
            <p className="text-muted-foreground">
              {number(current.merged_prs)} merged PRs in {snapshot.repo}. Includes {number(current.human_authored)}{" "}
              human-authored, {number(current.devin_authored)} Devin-authored, and {number(current.other_bot_authored)}{" "}
              other bot PRs. Each window is 28 complete UTC days.
            </p>
          </section>
          <section>
            <h3 className="mb-2 font-medium">People and agent ownership</h3>
            <p className="text-muted-foreground">
              {snapshot.people.length} gateway users matched to GitHub identities.{" "}
              {number(current.matched_internal_prs)} PRs match these people by author or explicit agent requester.{" "}
              {number(current.devin_without_requester)} Devin PRs have no explicit requester. They stay in repository
              totals, but are not assigned to a person.
            </p>
          </section>
          <section>
            <h3 className="mb-2 font-medium">Cost is a period average</h3>
            <p className="text-muted-foreground">
              Gateway-recorded spend divided by matched merged PRs. Some AI spend does not pass through the gateway, and
              recorded spend may include work outside this repository. Missing records are unavailable, never treated as
              free. Exact PR costs need repository, branch, or session links.
            </p>
          </section>
          <section>
            <h3 className="mb-2 font-medium">Quality signals</h3>
            <p className="text-muted-foreground">
              Bugs count newly opened issues carrying bug or kind:bug labels at collection time. Reverts count PR titles
              that explicitly say revert. These are repository signals, not a change-failure rate or a score for an
              engineer.
            </p>
          </section>
          <section>
            <h3 className="mb-2 font-medium">Comparisons need context</h3>
            <p className="text-muted-foreground">
              Changes show observed activity, not proof that AI caused them. Team composition, scope, and agent adoption
              also change. A missing historical baseline does not mean someone had no output.
            </p>
          </section>
          <section>
            <h3 className="mb-2 font-medium">Validated snapshot</h3>
            <p className="text-muted-foreground">
              Collected {new Date(snapshot.captured_at).toLocaleString("en-US", { timeZone: "UTC" })} UTC. Daily gateway
              user aggregates are the spend source. Sampled raw logs showed small differences from daily totals. This
              prototype does not sync new activity.
            </p>
          </section>
        </div>
      </SheetContent>
    </Sheet>
  );
}
