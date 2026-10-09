"use client";

import { useEffect, useMemo, useState } from "react";
import { ChartSkeleton, Stat } from "../overview/Primitives";
import { formatCompactUsd } from "../overview/overviewData";
import { BuilderDetail } from "./BuilderDetail";
import { BuilderInsightsDemoBanner } from "./BuilderInsightsDemoBanner";
import { BuilderList } from "./BuilderList";
import { sortBuilders, teamTotals, type BuilderSort } from "./builderInsightsData";
import { useBuilderInsightsRoute } from "./builderInsightsRoute";
import { useBuilderInsights } from "./useBuilderInsights";

export default function BuilderInsights() {
  const { data, isPending, isError } = useBuilderInsights();
  const [sort, setSort] = useState<BuilderSort>("spend");
  const { builderId, selectBuilder, closeBuilder } = useBuilderInsightsRoute();
  const sortedBuilders = useMemo(() => (data ? sortBuilders(data.builders, sort) : []), [data, sort]);
  const selectedBuilder = sortedBuilders.find((builder) => builder.id === builderId);
  const selectedIndex = selectedBuilder ? sortedBuilders.indexOf(selectedBuilder) : -1;
  const totals = useMemo(() => teamTotals(data?.builders ?? []), [data]);

  useEffect(() => {
    if (!selectedBuilder) return;
    document.getElementById("builder-insights-detail")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [selectedBuilder]);

  useEffect(() => {
    if (!selectedBuilder) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
      if (
        event.target instanceof HTMLElement &&
        event.target.closest("input, textarea, select, [contenteditable='true']")
      ) {
        return;
      }
      const next = sortedBuilders[selectedIndex + (event.key === "ArrowUp" ? -1 : 1)];
      if (!next) return;
      event.preventDefault();
      selectBuilder(next.id);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedBuilder, selectedIndex, selectBuilder, sortedBuilders]);

  if (isPending) {
    return (
      <div className="grid gap-3">
        <ChartSkeleton className="h-16" />
        <ChartSkeleton className="h-24" />
        <ChartSkeleton className="h-[42rem]" />
      </div>
    );
  }

  if (isError || !data)
    return <p className="py-10 text-center text-sm text-muted-foreground">Builder Insights is unavailable</p>;

  const snapshotStart = new Date(`${data.window.start}T00:00:00`);
  const snapshotEnd = new Date(`${data.window.end}T00:00:00`);
  const dateMeta = `${snapshotStart.toLocaleDateString("en-US", { month: "short", day: "numeric" })} to ${snapshotEnd.toLocaleDateString(
    "en-US",
    {
      month: "short",
      day: "numeric",
    },
  )}`;
  const snapshotLabel = `${dateMeta}, ${snapshotEnd.getFullYear()}`;

  return (
    <div className="grid gap-3">
      <BuilderInsightsDemoBanner />
      {!selectedBuilder && (
        <>
          <div className="grid grid-cols-1 overflow-hidden rounded-xl border bg-card sm:grid-cols-3 sm:divide-x">
            <div className="border-b px-5 py-4 sm:border-b-0">
              <Stat
                label="Team spend"
                value={formatCompactUsd(totals.spend)}
                hint={`${data.builders.length} builders · 30 days`}
              />
            </div>
            <div className="border-b px-5 py-4 sm:border-b-0">
              <Stat label="Merged PRs" value={totals.prs.toLocaleString()} hint={`${totals.prsDevin} with Devin`} />
            </div>
            <div className="px-5 py-4">
              <Stat label="Median cost / PR" value={formatCompactUsd(data.teamMedianCostPerPr)} />
            </div>
          </div>
          <BuilderList
            builders={data.builders}
            selectedId={null}
            sort={sort}
            onSelect={selectBuilder}
            onSortChange={setSort}
          />
        </>
      )}
      {selectedBuilder && (
        <BuilderDetail
          builder={selectedBuilder}
          builders={sortedBuilders}
          sort={sort}
          teamMedianCostPerPr={data.teamMedianCostPerPr}
          teamSpend={totals.spend}
          dateMeta={dateMeta}
          onSelectBuilder={selectBuilder}
          onClose={closeBuilder}
        />
      )}
      <p className="text-xs text-muted-foreground">
        Built from SpendLogs aggregates and merged PR counts. No prompts or code are read. {data.hidden.length}{" "}
        low-spend builders are hidden. Snapshot · {snapshotLabel}
      </p>
    </div>
  );
}
