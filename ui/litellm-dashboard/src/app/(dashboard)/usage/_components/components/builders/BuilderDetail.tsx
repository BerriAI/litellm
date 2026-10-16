"use client";

import { Check, Copy } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { copyToClipboard } from "@/utils/dataUtils";
import { formatCompactUsd } from "../overview/overviewData";
import { Panel } from "../overview/Primitives";
import {
  aggregateModels,
  builderCostPerPrLabel,
  builderVerdictDotClass,
  sortBuilders,
  type BuilderInsightBuilder,
} from "./builderInsightsData";
import { BuilderAgentSpendPanel } from "./BuilderAgentSpendPanel";
import { BuilderVerdictBadge } from "./BuilderVerdictBadge";
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

function BuilderSidebar({
  builders,
  sort,
  selectedId,
  onSelect,
}: {
  builders: readonly BuilderInsightBuilder[];
  sort: "spend" | "prs" | "efficiency";
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  return (
    <aside className="w-full shrink-0 border-b bg-background/50 p-3 md:w-72 md:border-r md:border-b-0">
      <h2 className="px-2 pb-2 text-sm font-semibold">Builders</h2>
      <div className="grid max-h-56 gap-1 overflow-y-auto md:max-h-[calc(100vh-14rem)]">
        {sortBuilders(builders, sort).map((builder) => {
          const costPerPrLabel = builderCostPerPrLabel(builder);
          const selected = builder.id === selectedId;
          return (
            <button
              key={builder.id}
              type="button"
              aria-current={selected ? "true" : undefined}
              onClick={() => onSelect(builder.id)}
              className={`rounded-lg px-2 py-2 text-left transition-colors ${
                selected ? "bg-muted" : "hover:bg-muted/60"
              }`}
            >
              <span className="block truncate text-sm font-medium">{builder.name}</span>
              <span className="mt-1 flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
                <span className={`size-1.5 shrink-0 rounded-full ${builderVerdictDotClass(builder.verdict)}`} />
                <span className="truncate">{builder.verdictLabel}</span>
                <span aria-hidden="true">·</span>
                <span className="shrink-0 tabular-nums">{costPerPrLabel}</span>
              </span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

function BuilderDocument({
  builder,
  median,
  teamSpend,
}: {
  builder: BuilderInsightBuilder;
  median: number;
  teamSpend: number;
}) {
  const mostlyDevin = builder.prs > 0 && builder.prsDevin / builder.prs > 0.5;
  const share = teamSpend > 0 ? (builder.spend / teamSpend) * 100 : 0;

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-7 px-6 py-8 md:px-10">
      <header>
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-2xl font-semibold tracking-tight">{builder.name}</h1>
          <BuilderVerdictBadge verdict={builder.verdict} label={builder.verdictLabel} />
        </div>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">{builder.verdictLine}</p>
      </header>
      <BuilderAgentSpendPanel agents={builder.agents} />
      <div className="grid grid-cols-1 gap-4 rounded-xl border bg-card px-4 py-4 sm:grid-cols-3 sm:divide-x sm:gap-0">
        <div className="sm:pr-4">
          <DetailStat
            label="Cost / PR"
            value={builderCostPerPrLabel(builder)}
            hint={
              mostlyDevin ? "Devin billed elsewhere, real cost higher" : `vs ${formatCompactUsd(median)} team median`
            }
          />
        </div>
        <div className="sm:px-4">
          <DetailStat label="Merged PRs" value={builder.prs.toLocaleString()} hint={`${builder.prsDevin} with Devin`} />
        </div>
        <div className="sm:pl-4">
          <DetailStat label="Spend" value={formatCompactUsd(builder.spend)} hint={`${share.toFixed(1)}% of team`} />
        </div>
      </div>
      <MarkdownFile
        filename={`${builder.id}-verdict.md`}
        markdown={builder.markdown}
        variant="document"
        showCopy={false}
      />
      <section className="grid gap-3">
        <h2 className="text-lg font-semibold">Evidence</h2>
        <ModelMix builder={builder} />
      </section>
    </div>
  );
}

export function BuilderDetail({
  builder,
  builders,
  sort,
  teamMedianCostPerPr,
  teamSpend,
  dateMeta,
  onSelectBuilder,
  onClose,
}: {
  builder: BuilderInsightBuilder;
  builders: readonly BuilderInsightBuilder[];
  sort: "spend" | "prs" | "efficiency";
  teamMedianCostPerPr: number;
  teamSpend: number;
  dateMeta: string;
  onSelectBuilder: (id: string) => void;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    if (!(await copyToClipboard(builder.markdown, "Markdown copied"))) return;
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };

  return (
    <section
      id="builder-insights-detail"
      className="flex min-h-[34rem] flex-col overflow-hidden rounded-xl border bg-card md:flex-row"
    >
      <BuilderSidebar builders={builders} sort={sort} selectedId={builder.id} onSelect={onSelectBuilder} />
      <div className="min-w-0 flex-1">
        <div className="flex min-h-12 flex-wrap items-center justify-between gap-2 border-b px-4 py-2">
          <span className="font-mono text-xs text-muted-foreground">
            {builder.id} · {dateMeta}
          </span>
          <div className="flex items-center gap-2">
            <Button type="button" variant="outline" size="xs" onClick={() => void copy()}>
              {copied ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
              {copied ? "Copied" : "Copy"}
            </Button>
            <Button type="button" variant="outline" size="xs" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
        <BuilderDocument builder={builder} median={teamMedianCostPerPr} teamSpend={teamSpend} />
      </div>
    </section>
  );
}
