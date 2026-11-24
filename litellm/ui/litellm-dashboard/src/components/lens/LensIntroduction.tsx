import { useState } from "react";
import { ArrowUpRight, ChevronDown, ChevronRight } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/cva.config";
import styles from "./LensIntroduction.module.css";
import { GatewayFlow } from "./GatewayFlow";

const traceSteps = [
  { name: "research_agent", start: 0, duration: 14.2, failed: false },
  { name: "plan", start: 0, duration: 1.7, failed: false },
  { name: "search_docs", start: 1.7, duration: 3.1, failed: false },
  { name: "run_benchmark", start: 4.8, duration: 1.3, failed: true },
  { name: "search_docs", start: 6.1, duration: 3.6, failed: false },
  { name: "answer", start: 9.7, duration: 4.5, failed: false },
];

type EvidenceProps = { highlighted: boolean; onHighlight: () => void };

function TraceExample({ highlighted, onHighlight }: EvidenceProps) {
  const { t } = useTranslation();
  return (
    <section
      aria-labelledby="lens-sample-trace"
      className={cn(styles.sampleCard, "min-w-0 rounded-xl border bg-card p-4")}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <h3 id="lens-sample-trace" className="font-semibold">
          {t("lensWorkspace.introduction.tracing")}
        </h3>
        <p className="text-xs text-muted-foreground">{t("lensWorkspace.introduction.sampleTrace")}</p>
      </div>
      <ol
        className="mt-3 space-y-1.5 text-xs leading-5 sm:text-sm"
        aria-label={t("lensWorkspace.introduction.timeline")}
      >
        {traceSteps.map((item, index) => {
          const linked = item.failed || item.name === "answer";
          const rowClass = cn(
            "grid w-full grid-cols-[minmax(0,1fr)_minmax(48px,1fr)_2.5rem] items-center gap-2 text-left sm:gap-3",
          );
          const content = (
            <>
              <span className={cn("flex min-w-0 items-center gap-1", index > 0 && "pl-2")}>
                {index === 0 ? (
                  <ChevronDown aria-hidden="true" className="size-3 shrink-0" />
                ) : (
                  <ChevronRight aria-hidden="true" className="size-3 shrink-0" />
                )}
                <span className="truncate" title={item.name}>
                  {item.name}
                </span>
                {item.failed && <span className="sr-only">{t("lensWorkspace.introduction.failed")}</span>}
              </span>
              <span aria-hidden="true" className="relative h-2.5 overflow-hidden rounded-sm bg-muted/70">
                <span
                  className={cn(
                    "absolute inset-y-0 rounded-sm",
                    linked && styles.evidenceBar,
                    item.failed ? "bg-rose-400" : "bg-indigo-600 dark:bg-indigo-400",
                  )}
                  style={{
                    left: `${(item.start / 14.2) * 100}%`,
                    width: `${(item.duration / 14.2) * 100}%`,
                  }}
                />
              </span>
              <span className="text-right tabular-nums text-muted-foreground">
                {t("lensWorkspace.introduction.duration", { duration: item.duration })}
              </span>
            </>
          );
          return (
            <li key={`${item.name}-${item.start}`}>
              {linked ? (
                <button
                  type="button"
                  className={cn(rowClass, styles.evidenceLink)}
                  data-evidence-link=""
                  aria-label={t("lensWorkspace.introduction.highlight", { name: item.name, duration: item.duration })}
                  aria-controls="lens-linked-finding"
                  aria-pressed={highlighted}
                  onClick={onHighlight}
                >
                  {content}
                </button>
              ) : (
                <div className={rowClass}>{content}</div>
              )}
            </li>
          );
        })}
      </ol>
      <div className="mt-3 grid gap-3 border-t pt-3 sm:grid-cols-2">
        <div id="lens-benchmark-output" className={cn(styles.evidenceOutput, "rounded-lg bg-muted/60 p-2.5")}>
          <p className="text-xs text-muted-foreground">
            {t("lensWorkspace.introduction.output", { name: "run_benchmark" })}
          </p>
          <p className="mt-1.5 text-sm leading-4.5">{t("lensWorkspace.introduction.benchmarkError")}</p>
        </div>
        <div id="lens-answer-output" className={cn(styles.evidenceOutput, "rounded-lg bg-muted/60 p-2.5")}>
          <p className="text-xs text-muted-foreground">{t("lensWorkspace.introduction.output", { name: "answer" })}</p>
          <p className="mt-1.5 text-sm leading-4.5">{t("lensWorkspace.introduction.sampleAnswer")}</p>
        </div>
      </div>
    </section>
  );
}

function FindingExamples({ highlighted, onHighlight }: EvidenceProps) {
  const { t } = useTranslation();
  return (
    <section
      aria-labelledby="lens-sample-findings"
      className={cn(styles.sampleCard, "min-w-0 rounded-xl border bg-card p-4")}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <h3 id="lens-sample-findings" className="font-semibold">
          {t("lensWorkspace.introduction.findings")}
        </h3>
        <p className="text-xs text-muted-foreground">{t("lensWorkspace.introduction.sampleRuns")}</p>
      </div>
      <div className="mt-3 divide-y border-t">
        <article id="lens-linked-finding" className={cn(styles.evidenceFinding, "py-3")}>
          <h4 className="text-sm font-semibold">
            <button
              type="button"
              className={cn(styles.evidenceLink, "flex w-full items-start gap-2 text-left")}
              data-evidence-link=""
              aria-controls="lens-benchmark-output lens-answer-output"
              aria-pressed={highlighted}
              onClick={onHighlight}
            >
              <span aria-hidden="true" className="mt-1.5 size-2 shrink-0 rounded-full bg-rose-500" />
              {t("lensWorkspace.introduction.claimTitle")}
            </button>
          </h4>
          <div className="ml-4 mt-1.5 space-y-1.5 text-sm leading-4.5">
            <p className="text-muted-foreground">{t("lensWorkspace.introduction.claimBody")}</p>
            <blockquote className="border-l-2 pl-3 italic">{t("lensWorkspace.introduction.claimQuote")}</blockquote>
            <p>
              <span className="text-muted-foreground">{t("lensWorkspace.introduction.nextStep")} </span>
              {t("lensWorkspace.introduction.claimAction")}
            </p>
            <p className="text-xs text-muted-foreground">{t("lensWorkspace.introduction.claimPriority")}</p>
          </div>
        </article>
        <article className="pt-3">
          <h4 className="flex items-start gap-2 text-sm font-semibold">
            <span aria-hidden="true" className="mt-1.5 size-2 shrink-0 rounded-full bg-amber-500" />
            {t("lensWorkspace.introduction.followupTitle")}
          </h4>
          <div className="ml-4 mt-1.5 space-y-1.5 text-sm leading-4.5 text-muted-foreground">
            <p>{t("lensWorkspace.introduction.followupBody")}</p>
            <p className="text-xs">{t("lensWorkspace.introduction.followupPriority")}</p>
          </div>
        </article>
      </div>
    </section>
  );
}

export function LensIntroduction({ onRetry }: { onRetry: () => void }) {
  const { t } = useTranslation();
  const [highlighted, setHighlighted] = useState(false);
  const toggleEvidence = () => setHighlighted((current) => !current);
  return (
    <section
      aria-labelledby="lens-introduction"
      className="@container mx-auto w-full max-w-6xl px-5 py-8 sm:px-8 sm:py-12 lg:px-10"
    >
      <h1 id="lens-introduction" className="max-w-3xl text-3xl leading-tight font-semibold tracking-tight sm:text-4xl">
        {t("lensWorkspace.introduction.title")}
      </h1>
      <p className="mt-3 max-w-[780px] text-sm leading-6 text-muted-foreground">
        {t("lensWorkspace.introduction.description")}
      </p>
      <div className="mt-5 flex flex-wrap items-center gap-3">
        <a className={buttonVariants()} href="https://docs.litellm.ai/docs/proxy/lens" target="_blank" rel="noreferrer">
          {t("lensWorkspace.introduction.setup")} <ArrowUpRight aria-hidden="true" className="size-4" />
        </a>
        <Button variant="outline" onClick={onRetry}>
          {t("lensWorkspace.retry")}
        </Button>
      </div>
      <GatewayFlow />
      <div
        className={cn(styles.examples, "mt-6 grid items-stretch gap-5 @2xl:grid-cols-2")}
        data-evidence-active={highlighted}
      >
        <TraceExample highlighted={highlighted} onHighlight={toggleEvidence} />
        <FindingExamples highlighted={highlighted} onHighlight={toggleEvidence} />
      </div>
    </section>
  );
}
