"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatCompactUsd } from "../overview/overviewData";
import { Panel } from "../overview/Primitives";
import {
  aggregateModels,
  builderCostPerPr,
  builderInitials,
  type BuilderInsightBuilder,
} from "./builderInsightsData";
import { BuilderVerdictBadge } from "./BuilderVerdictBadge";
import { InlineCodeText } from "./InlineCodeText";
import { MarkdownFile } from "./MarkdownFile";

function DetailStat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="min-w-0">
      <div className="h-4 text-xs leading-4 text-muted-foreground">{label}</div>
      <div className="mt-1 truncate text-xl leading-7 font-semibold tracking-tight tabular-nums">{value}</div>
      <div className="mt-0.5 text-xs leading-4 text-muted-foreground">{hint}</div>
    </div>
  );
}

function ModelMix({ builder }: { builder: BuilderInsightBuilder }) {
  const models = aggregateModels(builder.models).slice(0, 4);
  return (
    <Panel title="Where the money went">
      {models.length === 0 ? (
        <p className="text-xs text-muted-foreground">No model spend in this window</p>
      ) : (
        <div className="grid gap-3">
          {models.map((model) => (
            <div key={model.model} className="grid gap-1">
              <div className="flex items-center justify-between gap-3 text-xs">
                <span className="min-w-0 truncate font-mono">{model.model}</span>
                <span className="shrink-0 tabular-nums text-muted-foreground">{formatCompactUsd(model.spend)}</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                <span
                  className="block h-full rounded-full bg-[var(--chart-1)]"
                  style={{ width: `${builder.spend > 0 ? (model.spend / builder.spend) * 100 : 0}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

export function BuilderDetail({
  builder,
  teamMedianCostPerPr,
  teamSpend,
  detailId,
  canGoPrevious,
  canGoNext,
  onNavigate,
}: {
  builder: BuilderInsightBuilder;
  teamMedianCostPerPr: number;
  teamSpend: number;
  detailId: string;
  canGoPrevious: boolean;
  canGoNext: boolean;
  onNavigate: (direction: -1 | 1) => void;
}) {
  const costPerPr = builderCostPerPr(builder);
  const mostlyDevin = builder.prs > 0 && builder.prsDevin / builder.prs > 0.5;
  const teamShare = teamSpend > 0 ? (builder.spend / teamSpend) * 100 : 0;

  return (
    <section id={detailId} className="scroll-mt-4 space-y-3">
      <div className="flex flex-wrap items-center gap-4 rounded-xl border bg-card p-5">
        <span className="flex size-12 shrink-0 items-center justify-center rounded-full bg-muted text-sm font-semibold text-muted-foreground">
          {builderInitials(builder.name)}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-semibold tracking-tight">{builder.name}</h2>
            <BuilderVerdictBadge verdict={builder.verdict} label={builder.verdictLabel} />
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            <InlineCodeText text={builder.verdictLine} />
          </p>
        </div>
        <div className="ml-auto flex items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Previous builder"
            disabled={!canGoPrevious}
            onClick={() => onNavigate(-1)}
          >
            <ChevronLeft aria-hidden="true" className="size-4" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Next builder"
            disabled={!canGoNext}
            onClick={() => onNavigate(1)}
          >
            <ChevronRight aria-hidden="true" className="size-4" />
          </Button>
        </div>
      </div>
      <div className="grid grid-cols-1 overflow-hidden rounded-xl border bg-card sm:grid-cols-3 sm:divide-x">
        <div className="border-b px-4 py-4 sm:border-b-0">
          <DetailStat
            label="Cost / PR"
            value={costPerPr === null ? "—" : formatCompactUsd(costPerPr)}
            hint={mostlyDevin ? "Devin billed elsewhere, real cost higher" : `vs ${formatCompactUsd(teamMedianCostPerPr)} team median`}
          />
        </div>
        <div className="border-b px-4 py-4 sm:border-b-0">
          <DetailStat label="Merged PRs" value={builder.prs.toLocaleString()} hint={`${builder.prsDevin} with Devin`} />
        </div>
        <div className="px-4 py-4">
          <DetailStat label="Spend" value={formatCompactUsd(builder.spend)} hint={`${teamShare.toFixed(1)}% of team`} />
        </div>
      </div>
      <MarkdownFile filename={`${builder.id}-verdict.md`} markdown={builder.markdown} />
      <ModelMix builder={builder} />
    </section>
  );
}
