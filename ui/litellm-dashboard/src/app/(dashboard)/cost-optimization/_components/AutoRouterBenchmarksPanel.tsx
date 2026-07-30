"use client";

import React from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AutoRouterGroupBenchmark } from "@/components/networking";
import { useAutoRouterBenchmarks } from "./useAutoRouterBenchmarks";

const compactNumber = (n: number): string =>
  new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n);

const currency = (n: number): string =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(n);

const durationLabel = (seconds: number): string => {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
};

const StatTile: React.FC<{ label: string; value: string; caption?: string }> = ({ label, value, caption }) => (
  <div className="rounded-lg border border-border bg-card p-4">
    <p className="text-sm text-muted-foreground">{label}</p>
    <p className="mt-1 text-2xl font-semibold text-foreground">{value}</p>
    {caption && <p className="mt-1 text-xs text-muted-foreground">{caption}</p>}
  </div>
);

const GroupBenchmark: React.FC<{ group: AutoRouterGroupBenchmark }> = ({ group }) => (
  <Card>
    <CardHeader>
      <div className="flex flex-col gap-1 md:flex-row md:items-center md:justify-between">
        <CardTitle>{group.model_group}</CardTitle>
        <span className="text-xs text-muted-foreground">{group.sessions} sessions over the last 30 days</span>
      </div>
    </CardHeader>
    <CardContent className="space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="Turns per session" value={group.avg_turns_per_session.toFixed(1)} />
        <StatTile label="Avg session length" value={durationLabel(group.avg_session_length_seconds)} />
        <StatTile label="Tokens per session" value={compactNumber(group.avg_tokens_per_session)} />
        <StatTile
          label="Estimated savings"
          value={currency(group.savings)}
          caption={`${group.savings_pct.toFixed(0)}% vs ${group.baseline_model}`}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        Savings estimate compares the routed model mix ({currency(group.actual_spend)}) against sending every request to{" "}
        {group.baseline_model} at list prices ({currency(group.baseline_spend)}); it does not model the caching a
        single-model baseline would have had
      </p>
    </CardContent>
  </Card>
);

interface AutoRouterBenchmarksPanelProps {
  accessToken: string | null;
}

const AutoRouterBenchmarksPanel: React.FC<AutoRouterBenchmarksPanelProps> = ({ accessToken }) => {
  const { data, loading, error } = useAutoRouterBenchmarks(accessToken);

  if (loading) {
    return <p className="py-8 text-center text-sm text-muted-foreground">Loading auto-router benchmarks...</p>;
  }
  if (error) {
    return null;
  }
  if (!data || data.groups.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-base font-semibold text-foreground">Benchmarks</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          What each auto-router is buying you, measured over routed sessions in the last 30 days
        </p>
      </div>
      {data.groups.map((group) => (
        <GroupBenchmark key={group.model_group} group={group} />
      ))}
    </div>
  );
};

export default AutoRouterBenchmarksPanel;
