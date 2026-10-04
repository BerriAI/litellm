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
  changeTerms,
  duration,
  money,
  number,
  recordedBranches,
  type Comparison,
  type ObservedPerson,
  type ObservedPull,
  type ObservedSnapshot,
  type Period,
} from "./observedData";

export function PullList({
  pulls,
  provider,
  matchedOnly = false,
}: {
  matchedOnly?: boolean;
  pulls: ObservedPull[];
  provider: ObservedSnapshot["source_provider"];
}) {
  const terms = changeTerms(provider);
  const emptyMessage = matchedOnly
    ? "No merged changes from matched people in this period"
    : `No ${terms.lower} in this period`;
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(20);
  const filtered = pulls.filter((pull) =>
    `${pull.number} ${pull.title} ${pull.author} ${pull.repo} ${pull.source_repo} ${pull.source_branch}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  return (
    <div className="space-y-3">
      <Input
        aria-label={`Search ${terms.lower}`}
        placeholder={`Search ${terms.lower}…`}
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
            <TableHead>{terms.requests}</TableHead>
            <TableHead>Author</TableHead>
            <TableHead className="text-right">Opened to merged</TableHead>
            <TableHead className="text-right">Tagged spend</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {filtered.slice(0, limit).map((pull) => (
            <TableRow key={pull.url}>
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
                    <span className="mt-1 block text-xs text-muted-foreground">{pull.repo}</span>
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
              <TableCell
                className="text-right tabular-nums"
                title={
                  pull.branch_cost.status === "ambiguous"
                    ? "Branch is shared by multiple changes"
                    : "Spend tagged to this branch during the period"
                }
              >
                {money(pull.branch_cost.spend)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <p className="text-xs text-muted-foreground">
        Elapsed time from opening to merge, not engineering effort or time saved
      </p>
      {filtered.length === 0 && (
        <p className="py-8 text-center text-sm text-muted-foreground">
          {query ? `No ${terms.lower} match this search` : emptyMessage}
        </p>
      )}
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>
          {number(Math.min(limit, filtered.length))} of {number(filtered.length)} {terms.lower}
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
  onEdit,
}: {
  person: ObservedPerson;
  snapshot: ObservedSnapshot;
  comparison: Comparison;
  onClose: () => void;
  onEdit?: () => void;
}) {
  const terms = changeTerms(snapshot.source_provider);
  const [period, setPeriod] = useState<Period>("current");
  const current = person.periods.current;
  const baseline = person.periods[comparison];
  const urls = new Set(person.periods[period].pr_urls);
  const pulls = snapshot.pulls[period].filter((pull) => urls.has(pull.url));
  return (
    <Sheet
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <SheetContent className="overflow-y-auto p-6 data-[side=right]:w-full data-[side=right]:sm:max-w-3xl">
        <SheetHeader className="p-0 pr-8">
          <SheetTitle className="text-xl">{person.name}</SheetTitle>
          <SheetDescription>{[person.email, person.logins.join(", ")].filter(Boolean).join(" · ")}</SheetDescription>
        </SheetHeader>
        {onEdit && (
          <Button variant="outline" className="w-fit" onClick={onEdit}>
            Edit linked accounts
          </Button>
        )}
        <div className="mt-2 grid grid-cols-1 gap-y-4 divide-y rounded-lg border py-5 sm:grid-cols-3 sm:divide-x sm:divide-y-0">
          <div className="px-4">
            <p className="text-xs text-muted-foreground">Merged {terms.plural}</p>
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
            <p className="text-xs text-muted-foreground">Spend / matched {terms.singular}</p>
            <p className="mt-2 text-2xl font-semibold">{money(current.recorded_spend_per_attributed_pr)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Period average</p>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <h3 className="font-medium">Merged {terms.plural}</h3>
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
        <PullList key={period} pulls={pulls} provider={snapshot.source_provider} />
      </SheetContent>
    </Sheet>
  );
}

export function BranchSpend({ snapshot, matchedOnly = true }: { snapshot: ObservedSnapshot; matchedOnly?: boolean }) {
  const rows = recordedBranches(snapshot, matchedOnly);
  return (
    <div className="rounded-xl border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Repository</TableHead>
            <TableHead>Branch</TableHead>
            <TableHead className="text-right">Requests</TableHead>
            <TableHead className="text-right">Tagged spend</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={`${row.repo}/${row.branch}`}>
              <TableCell>{row.repo}</TableCell>
              <TableCell>{row.branch}</TableCell>
              <TableCell className="text-right">{number(row.requests)}</TableCell>
              <TableCell className="text-right">{money(row.spend)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {rows.length === 0 && (
        <p className="p-8 text-center text-sm text-muted-foreground">
          {matchedOnly
            ? "No tagged branch spend for matched people in this period"
            : "No tagged branch spend in this period"}
        </p>
      )}
    </div>
  );
}
