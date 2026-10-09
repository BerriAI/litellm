"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ChartSkeleton, Stat } from "../overview/Primitives";
import { formatCompactUsd } from "../overview/overviewData";
import { BuilderDetail } from "./BuilderDetail";
import { BuilderList } from "./BuilderList";
import { sortBuilders, teamTotals, type BuilderSort } from "./builderInsightsData";
import { useBuilderInsights } from "./useBuilderInsights";

export default function BuilderInsights() {
  const { data, isPending, isError } = useBuilderInsights();
  const [sort, setSort] = useState<BuilderSort>("spend");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const sortedBuilders = useMemo(() => (data ? sortBuilders(data.builders, sort) : []), [data, sort]);
  const selectedBuilder = sortedBuilders.find((builder) => builder.id === selectedId) ?? sortedBuilders[0];
  const selectedIndex = selectedBuilder ? sortedBuilders.indexOf(selectedBuilder) : -1;
  const totals = useMemo(() => teamTotals(data?.builders ?? []), [data]);
  const navigate = useCallback(
    (direction: -1 | 1) => {
      const next = sortedBuilders[selectedIndex + direction];
      if (next) setSelectedId(next.id);
    },
    [selectedIndex, sortedBuilders],
  );

  useEffect(() => {
    if (!selectedId) return;
    document.getElementById("builder-insights-detail")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [selectedId]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      if (
        event.target instanceof HTMLElement &&
        event.target.closest("input, textarea, select, [contenteditable='true']")
      ) {
        return;
      }
      event.preventDefault();
      navigate(event.key === "ArrowLeft" ? -1 : 1);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [navigate]);

  if (isPending) {
    return (
      <div className="grid gap-3">
        <ChartSkeleton className="h-24" />
        <div className="grid gap-3 lg:grid-cols-5">
          <ChartSkeleton className="h-[42rem] lg:col-span-2" />
          <ChartSkeleton className="h-[42rem] lg:col-span-3" />
        </div>
      </div>
    );
  }

  if (isError || !data) return <p className="py-10 text-center text-sm text-muted-foreground">Builder Insights is unavailable</p>;

  const snapshotStart = new Date(`${data.window.start}T00:00:00`);
  const snapshotEnd = new Date(`${data.window.end}T00:00:00`);
  const snapshotStartLabel = snapshotStart.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const snapshotEndLabel = snapshotEnd.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  return (
    <div className="grid gap-3">
      <p className="text-right text-xs text-muted-foreground">Snapshot · {snapshotStartLabel} to {snapshotEndLabel}</p>
      <div className="grid grid-cols-1 overflow-hidden rounded-xl border bg-card sm:grid-cols-3 sm:divide-x">
        <div className="border-b px-5 py-4 sm:border-b-0">
          <Stat label="Team spend" value={formatCompactUsd(totals.spend)} hint={`${totals.builderCount} builders · 30 days`} />
        </div>
        <div className="border-b px-5 py-4 sm:border-b-0">
          <Stat label="Merged PRs" value={totals.prs.toLocaleString()} hint={`${totals.prsDevin} with Devin`} />
        </div>
        <div className="px-5 py-4">
          <Stat
            label="Median cost / PR"
            value={formatCompactUsd(data.teamMedianCostPerPr)}
          />
        </div>
      </div>
      <div className="grid gap-3 lg:grid-cols-5">
        <div className="lg:col-span-2">
          <BuilderList
            builders={data.builders}
            selectedId={selectedBuilder?.id ?? null}
            sort={sort}
            onSelect={setSelectedId}
            onSortChange={setSort}
          />
        </div>
        <div className="lg:sticky lg:top-4 lg:col-span-3 lg:self-start">
          {selectedBuilder && (
            <BuilderDetail
              builder={selectedBuilder}
              teamMedianCostPerPr={data.teamMedianCostPerPr}
              teamSpend={totals.spend}
              detailId="builder-insights-detail"
              canGoPrevious={selectedIndex > 0}
              canGoNext={selectedIndex < sortedBuilders.length - 1}
              onNavigate={navigate}
            />
          )}
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        Built from SpendLogs aggregates and merged PR counts. No prompts or code are read.{" "}
        {data.hidden.length} low-spend builders are hidden.
      </p>
    </div>
  );
}
