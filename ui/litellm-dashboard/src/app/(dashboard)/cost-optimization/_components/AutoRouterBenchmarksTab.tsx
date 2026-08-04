"use client";

import React, { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

import {
  ALL_ROUTERS,
  compactNumber,
  durationLabel,
  pct,
  toView,
  ttlLabel,
  usd,
  type AutoRouterCacheBenchmark,
  type AutoRouterGroupBenchmark,
  type BenchmarkView,
} from "./autoRouterBenchmarks";
import { benchmarksWindow, useAutoRouterBenchmarks } from "./useAutoRouterBenchmarks";

interface Bucket {
  key: "same_model" | "first_visit" | "return";
  label: string;
  turns: number;
  hitRate: number;
  fill: string;
}

const bucketsOf = (cache: AutoRouterCacheBenchmark): Bucket[] => [
  {
    key: "same_model",
    label: "Same model",
    turns: cache.same_model_turns,
    hitRate: cache.same_model_hit_rate_pct,
    fill: "bg-foreground",
  },
  {
    key: "first_visit",
    label: "First visit",
    turns: cache.first_visit_turns,
    hitRate: cache.first_visit_hit_rate_pct,
    fill: "bg-foreground/30",
  },
  {
    key: "return",
    label: "Return to tier",
    turns: cache.return_turns,
    hitRate: cache.return_hit_rate_pct,
    fill: "bg-foreground/60",
  },
];

const Metric: React.FC<{ label: string; value: string; muted?: boolean }> = ({ label, value, muted = false }) => (
  <Card size="sm">
    <CardHeader>
      <CardTitle className="text-sm font-normal text-muted-foreground">{label}</CardTitle>
    </CardHeader>
    <CardContent>
      <p className={`text-3xl font-semibold ${muted ? "text-muted-foreground" : "text-foreground"}`}>{value}</p>
    </CardContent>
  </Card>
);

/**
 * The one hero figure on the view. Savings leads because it is the number the tab
 * exists to prove; the session shape beside it is the denominator that number is
 * earned over, which is why they share a card rather than sitting in a tile row.
 */
const HeroCard: React.FC<{ view: BenchmarkView }> = ({ view }) => {
  const measured = view.baselineLabel !== null;
  const cheaper = view.savings >= 0;
  return (
    <Card className="overflow-hidden py-0">
      <div className="grid md:grid-cols-[1fr_20rem]">
        <div className="flex flex-col gap-3 p-6">
          <p className="text-sm text-muted-foreground">Total estimated savings</p>
          {measured ? (
            <>
              <div className="flex flex-wrap items-center gap-3">
                <p className="text-5xl font-semibold tracking-tight text-foreground">{usd(view.savings)}</p>
                <Badge variant="secondary" className={cheaper ? "text-muted-foreground" : "text-destructive"}>
                  {cheaper ? "-" : "+"}
                  {Math.abs(view.savings_pct).toFixed(0)}%
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                {usd(view.actual_spend)} routed
                <span className="mx-2 text-muted-foreground/50">/</span>
                {usd(view.baseline_spend)} all-{view.baselineLabel?.split("/").pop()} baseline
              </p>
            </>
          ) : (
            <>
              <p className="text-5xl font-semibold tracking-tight text-muted-foreground">Not measured</p>
              <p className="text-xs text-muted-foreground">
                These routers declare no counterfactual baseline, so there is nothing to compare the routed mix against
              </p>
            </>
          )}
        </div>

        <div className="flex flex-col gap-3 border-t bg-muted/40 p-6 md:border-t-0 md:border-l">
          <p className="text-sm text-muted-foreground">Sessions on auto-router</p>
          <div className="flex items-baseline gap-2">
            <p className="text-3xl font-semibold text-foreground">{view.sessions.toLocaleString()}</p>
            <p className="text-xs text-muted-foreground">{view.turns.toLocaleString()} turns</p>
          </div>
          <Separator />
          <dl className="flex flex-col gap-2 text-sm">
            <div className="flex items-baseline justify-between gap-2">
              <dt className="text-muted-foreground">Saved per session</dt>
              <dd className="font-medium tabular-nums text-foreground">
                {measured ? usd(view.saved_per_session) : "n/a"}
              </dd>
            </div>
            <div className="flex items-baseline justify-between gap-2">
              <dt className="text-muted-foreground">Auto-routers in scope</dt>
              <dd className="font-medium tabular-nums text-foreground">{view.routers}</dd>
            </div>
          </dl>
        </div>
      </div>
    </Card>
  );
};

/**
 * Segments are separated by a 2px gap in the surface colour rather than a stroke,
 * so neighbouring steps of the same ramp stay distinct without adding ink that
 * isn't data.
 */
const StackedTurnBar: React.FC<{ buckets: Bucket[]; total: number }> = ({ buckets, total }) => (
  <div
    className="flex h-2.5 w-full gap-0.5 overflow-hidden rounded-sm"
    role="img"
    aria-label="Share of turns by bucket"
  >
    {buckets
      .filter((b) => b.turns > 0)
      .map((b) => (
        <div
          key={b.key}
          className={`${b.fill} first:rounded-l-sm last:rounded-r-sm`}
          style={{ width: `${total > 0 ? (100 * b.turns) / total : 0}%` }}
          title={`${b.label}: ${b.turns.toLocaleString()} turns (${pct(total > 0 ? (100 * b.turns) / total : 0)})`}
        />
      ))}
  </div>
);

const BucketTable: React.FC<{ buckets: Bucket[] }> = ({ buckets }) => (
  <Table>
    <TableHeader>
      <TableRow className="hover:bg-transparent">
        <TableHead className="text-[11px] uppercase tracking-wide">Bucket</TableHead>
        <TableHead className="text-right text-[11px] uppercase tracking-wide">Turns</TableHead>
        <TableHead className="w-1/2" />
        <TableHead className="text-right text-[11px] uppercase tracking-wide">Hit</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {buckets.map((b) => (
        <TableRow key={b.key} className="hover:bg-transparent">
          <TableCell className="text-foreground">
            <span className="flex items-center gap-2">
              <span className={`inline-block size-2 shrink-0 rounded-sm ${b.fill}`} aria-hidden />
              {b.label}
            </span>
          </TableCell>
          <TableCell className="text-right tabular-nums text-muted-foreground">{b.turns.toLocaleString()}</TableCell>
          <TableCell>
            <div className="h-1.5 w-full rounded-full bg-muted">
              <div className="h-full rounded-full bg-foreground" style={{ width: `${b.hitRate}%` }} aria-hidden />
            </div>
          </TableCell>
          <TableCell className="text-right font-medium tabular-nums text-foreground">{pct(b.hitRate)}</TableCell>
        </TableRow>
      ))}
    </TableBody>
  </Table>
);

const HitRateCard: React.FC<{ cache: AutoRouterCacheBenchmark; mixedTtl: boolean }> = ({ cache, mixedTtl }) => {
  const buckets = bucketsOf(cache);
  return (
    <Card size="sm" className="gap-0">
      <CardHeader>
        <CardTitle className="text-sm font-normal text-muted-foreground">Cache hit rate</CardTitle>
        <CardAction>
          <Badge variant="secondary" className="text-muted-foreground">
            {pct(cache.usage_coverage_pct)} coverage
          </Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 pt-4">
        <div className="flex flex-wrap items-baseline gap-2">
          <p className="text-4xl font-semibold tracking-tight text-foreground">{pct(cache.hit_rate_pct)}</p>
          <p className="text-xs text-muted-foreground">
            across {cache.turns.toLocaleString()} turns
            {mixedTtl ? "; mixed TTLs" : ` · ${ttlLabel(cache.ttl_seconds)} TTL`}
          </p>
        </div>
        <div className="flex flex-col gap-1.5">
          <StackedTurnBar buckets={buckets} total={cache.turns} />
          <div className="flex justify-between text-[11px] text-muted-foreground">
            <span>share of turns</span>
            <span>{cache.turns.toLocaleString()} turns total</span>
          </div>
        </div>
      </CardContent>
      <Separator className="mt-4" />
      <CardContent className="pt-2">
        <BucketTable buckets={buckets} />
        <p className="mt-3 text-xs text-muted-foreground">
          Mutually exclusive buckets; every turn lands in exactly one, by what the router did. The headline is{" "}
          <span className="font-medium text-foreground">weighted by turn count, not averaged</span>, which is why{" "}
          {pct(cache.hit_rate_pct)} sits near same-model&apos;s {pct(cache.same_model_hit_rate_pct)} rather than in the
          middle. A first visit to a tier is cold by design
        </p>
      </CardContent>
    </Card>
  );
};

const WarmingCard: React.FC<{ cache: AutoRouterCacheBenchmark; routers: number }> = ({ cache, routers }) => {
  const paysOff = cache.warming_savable_miss_pct >= cache.warming_break_even_pct;
  return (
    <Card size="sm" className="gap-0">
      <CardHeader>
        <CardTitle className="text-sm font-normal text-muted-foreground">Savable by warming</CardTitle>
        <CardAction>
          <Badge variant="secondary" className="text-muted-foreground">
            estimate
          </Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 pt-4">
        <div className="flex flex-wrap items-baseline gap-2">
          <p className="text-4xl font-semibold tracking-tight text-foreground">{pct(cache.warming_savable_miss_pct)}</p>
          <p className="text-xs text-muted-foreground">of every cache miss</p>
        </div>
        <p className="text-xs text-muted-foreground">
          {pct(cache.stale_miss_share_pct)} of return-to-tier misses expired past the TTL; the rest missed because the
          prefix changed, which warming cannot fix
        </p>
        <div className="flex items-center gap-3">
          <div className="relative h-2.5 flex-1 rounded-sm bg-muted">
            <div
              className={`h-full rounded-sm ${paysOff ? "bg-foreground" : "bg-foreground/40"}`}
              style={{ width: `${Math.min(cache.warming_savable_miss_pct, 100)}%` }}
              title={`${pct(cache.warming_savable_miss_pct)} of misses are savable`}
            />
            <div
              className="absolute top-[-3px] h-[calc(100%+6px)] w-0.5 bg-muted-foreground"
              style={{ left: `${Math.min(cache.warming_break_even_pct, 100)}%` }}
              aria-hidden
            />
          </div>
          <span className="shrink-0 text-[11px] text-muted-foreground">
            break-even ≈ {pct(cache.warming_break_even_pct, 0)} at {ttlLabel(cache.ttl_seconds)}
          </span>
        </div>
      </CardContent>
      <Separator className="mt-4" />
      <CardContent className="pt-2">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">If warming were running</p>
        <dl className="mt-1 divide-y divide-border">
          <div className="flex items-baseline justify-between gap-2 py-3">
            <dt className="text-sm text-foreground">Cache writes rescued</dt>
            <dd className="text-sm tabular-nums text-foreground">{usd(cache.warming_rescued_spend)}</dd>
          </div>
          <div className="flex items-baseline justify-between gap-2 py-3">
            <div>
              <dt className="text-sm text-foreground">Cache warming costs</dt>
              <p className="text-xs text-muted-foreground">replays, one per idle window</p>
            </div>
            <dd className="text-sm tabular-nums text-destructive">{usd(-cache.warming_replay_spend)}</dd>
          </div>
          <div className="flex items-baseline justify-between gap-2 py-3">
            <div>
              <dt className="text-sm font-medium text-foreground">Warming estimate</dt>
              <p className="text-xs text-muted-foreground">
                net, across {routers} {routers === 1 ? "router" : "routers"}
              </p>
            </div>
            <dd
              className={`text-2xl font-semibold tabular-nums ${
                cache.warming_net_spend >= 0 ? "text-foreground" : "text-destructive"
              }`}
            >
              {usd(cache.warming_net_spend)}
            </dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  );
};

const Message: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <p className="py-8 text-center text-sm text-muted-foreground">{children}</p>
);

interface AutoRouterBenchmarksTabProps {
  accessToken: string | null;
}

const AutoRouterBenchmarksTab: React.FC<AutoRouterBenchmarksTabProps> = ({ accessToken }) => {
  const window = useMemo(() => benchmarksWindow(new Date()), []);
  const { data, isPending, isError } = useAutoRouterBenchmarks(accessToken, window);
  const [selected, setSelected] = useState<string>(ALL_ROUTERS);

  const groups: AutoRouterGroupBenchmark[] = useMemo(() => data?.groups ?? [], [data]);
  const shown = useMemo(
    () => (selected === ALL_ROUTERS ? groups : groups.filter((g) => g.model_group === selected)),
    [groups, selected],
  );
  const view = useMemo(() => toView(shown), [shown]);

  if (isPending) return <Message>Loading auto-router benchmarks...</Message>;
  if (isError) return <Message>Auto-router benchmarks are unavailable right now</Message>;
  if (groups.length === 0) return <Message>No auto-router sessions in the last 30 days yet</Message>;
  if (view === null) return <Message>No sessions for this auto-router in the last 30 days</Message>;

  return (
    <div className="w-full space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-foreground">Auto-router benchmarks</h2>
          <p className="mt-1 text-sm text-muted-foreground">Last 30 days</p>
        </div>
        <div className="w-full sm:w-64">
          <Select value={selected} onValueChange={(value) => setSelected(value ?? ALL_ROUTERS)}>
            <SelectTrigger className="w-full">
              <SelectValue>{selected === ALL_ROUTERS ? "All auto-routers" : selected}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL_ROUTERS}>All auto-routers</SelectItem>
              {groups.map((g) => (
                <SelectItem key={g.model_group} value={g.model_group}>
                  {g.model_group}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <HeroCard view={view} />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Metric label="Avg turns per session" value={view.avg_turns_per_session.toFixed(1)} />
        <Metric label="Avg session length" value={durationLabel(view.avg_session_length_seconds)} />
        <Metric label="Avg tokens per session" value={compactNumber(view.avg_tokens_per_session)} />
      </div>

      {view.baselineLabel !== null && (
        <p className="text-xs text-muted-foreground">
          Compares the routed model mix against sending every request to{" "}
          <code className="font-mono text-[11px]">{view.baselineLabel}</code> at list prices, pricing that baseline with
          a warm single-model cache. A router that thrashes the prompt cache can therefore show a loss, which is a real
          cost rather than a rounding artefact
        </p>
      )}

      {view.cache && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-baseline gap-2">
            <h3 className="text-lg font-semibold tracking-tight text-foreground">Auto-router prompt caching</h3>
            <p className="text-xs text-muted-foreground">
              per router; tier ladders differ, and a blended rate would hide who is paying for cold writes
            </p>
          </div>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <HitRateCard cache={view.cache} mixedTtl={view.mixedTtl} />
            <WarmingCard cache={view.cache} routers={view.routers} />
          </div>
        </div>
      )}
    </div>
  );
};

export default AutoRouterBenchmarksTab;
