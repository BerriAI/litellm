import { useId } from "react";
import { ArrowRight, ArrowUpRight, ChevronDown, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

const swarmColors = ["fill-violet-400", "fill-sky-400", "fill-amber-400", "fill-rose-400", "fill-indigo-400"];

function flowDot(index: number) {
  const column = index % 100;
  const row = Math.floor(index / 100);
  const wave = 2.5 + 2.6 * Math.sin((column + 5) / 3.8);
  if (Math.abs(row - wave) > 0.9) return null;
  return {
    x: column * 10 + 5,
    y: row * 10 + 5,
    color: column < 61 ? swarmColors[(column + row) % swarmColors.length] : "fill-indigo-600 dark:fill-indigo-400",
  };
}

const flowDots = Array.from({ length: 600 }, (_, index) => flowDot(index)).filter((dot) => dot !== null);

function GatewayFlow() {
  const patternId = useId();
  return (
    <div className="mt-10 sm:mt-12">
      <div className="grid grid-cols-3 gap-3 text-xs sm:text-sm">
        <div>
          <p className="font-semibold">Agent swarms</p>
          <p className="mt-1 leading-5 text-muted-foreground">Every run, every recorded step</p>
        </div>
        <div className="text-center">
          <p className="font-semibold">LiteLLM gateway</p>
          <p className="mt-1 leading-5 text-muted-foreground">One place, your infrastructure</p>
        </div>
        <div className="text-right">
          <p className="font-semibold">Lens</p>
          <p className="mt-1 leading-5 text-muted-foreground">Findings to improve your agents</p>
        </div>
      </div>
      <svg aria-hidden="true" viewBox="0 0 1000 60" className="mt-3 h-auto w-full" focusable="false">
        <defs>
          <pattern id={patternId} width="10" height="10" patternUnits="userSpaceOnUse">
            <circle cx="5" cy="5" r="2.3" className="fill-muted-foreground/10" />
          </pattern>
        </defs>
        <rect width="1000" height="60" fill={`url(#${patternId})`} />
        {flowDots.map((dot) => (
          <circle key={`${dot.x}-${dot.y}`} cx={dot.x} cy={dot.y} r="2.3" className={dot.color} />
        ))}
        <path d="M 615 1 H 609 V 59 H 615" fill="none" stroke="currentColor" strokeWidth="1.8" />
      </svg>
    </div>
  );
}

const traceSteps = [
  { name: "research_agent", start: 0, duration: 14.2, failed: false },
  { name: "plan", start: 0, duration: 1.7, failed: false },
  { name: "search_docs", start: 1.7, duration: 3.1, failed: false },
  { name: "run_benchmark", start: 4.8, duration: 1.3, failed: true },
  { name: "search_docs", start: 6.1, duration: 3.6, failed: false },
  { name: "answer", start: 9.7, duration: 4.5, failed: false },
];

const sampleAnswer = "The column store is 40% faster than the row store for your workload.";

function TraceExample() {
  return (
    <section aria-labelledby="lens-sample-trace" className="min-w-0 rounded-2xl border bg-card p-4 sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <h3 id="lens-sample-trace" className="font-semibold">
          Tracing
        </h3>
        <p className="text-xs text-muted-foreground">Sample trace · 14.2s</p>
      </div>
      <ol className="mt-5 space-y-3.5 text-xs sm:text-[13px]" aria-label="Sample trace timeline">
        {traceSteps.map((item, index) => (
          <li
            key={`${item.name}-${item.start}`}
            className="grid grid-cols-[minmax(0,1fr)_minmax(48px,1fr)_2.5rem] items-center gap-2 sm:gap-3"
          >
            <span className={`flex min-w-0 items-center gap-1 ${index > 0 ? "pl-2" : ""}`}>
              {index === 0 ? (
                <ChevronDown aria-hidden="true" className="size-3 shrink-0" />
              ) : (
                <ChevronRight aria-hidden="true" className="size-3 shrink-0" />
              )}
              <span className="truncate" title={item.name}>
                {item.name}
              </span>
              {item.failed && <span className="sr-only">Failed</span>}
            </span>
            <span aria-hidden="true" className="relative h-2.5 overflow-hidden rounded-sm bg-muted/70">
              <span
                className={`absolute inset-y-0 rounded-sm ${item.failed ? "bg-rose-400" : "bg-indigo-600 dark:bg-indigo-400"}`}
                style={{ left: `${(item.start / 14.2) * 100}%`, width: `${(item.duration / 14.2) * 100}%` }}
              />
            </span>
            <span className="text-right tabular-nums text-muted-foreground">{item.duration}s</span>
          </li>
        ))}
      </ol>
      <div className="mt-5 grid gap-3 border-t pt-4 sm:grid-cols-2">
        <div className="rounded-xl bg-muted/60 p-3.5">
          <p className="text-xs text-muted-foreground">run_benchmark · output</p>
          <p className="mt-2 text-[13px] leading-5">Error: benchmark runner unavailable (503)</p>
        </div>
        <div className="rounded-xl bg-muted/60 p-3.5">
          <p className="text-xs text-muted-foreground">answer · output</p>
          <p className="mt-2 text-[13px] leading-5">“{sampleAnswer}”</p>
        </div>
      </div>
    </section>
  );
}

function FindingExamples() {
  return (
    <section aria-labelledby="lens-sample-findings" className="min-w-0 rounded-2xl border bg-card p-4 sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <h3 id="lens-sample-findings" className="font-semibold">
          Lens findings
        </h3>
        <p className="text-xs text-muted-foreground">500 sample runs reviewed</p>
      </div>
      <div className="mt-4 divide-y border-t">
        <article className="py-4">
          <h4 className="flex items-start gap-2 text-sm font-semibold">
            <span aria-hidden="true" className="mt-1.5 size-2 shrink-0 rounded-full bg-rose-500" />
            Claims numbers it never measured
          </h4>
          <div className="ml-4 mt-2 space-y-2 text-[13px] leading-5">
            <p className="text-muted-foreground">
              When the benchmark tool fails, the agent still reports a speedup in its answer.
            </p>
            <blockquote className="border-l-2 pl-3 italic">“{sampleAnswer}”</blockquote>
            <p>
              <span className="text-muted-foreground">Next step: </span>Tell the agent to report the failed benchmark
              instead of estimating.
            </p>
            <p className="text-xs text-muted-foreground">38 linked runs · High priority</p>
          </div>
        </article>
        <article className="pt-4">
          <h4 className="flex items-start gap-2 text-sm font-semibold">
            <span aria-hidden="true" className="mt-1.5 size-2 shrink-0 rounded-full bg-amber-500" />
            Promises to follow up, then stops
          </h4>
          <div className="ml-4 mt-2 space-y-2 text-[13px] leading-5 text-muted-foreground">
            <p>After the order lookup times out, the support agent replies “I will check” and ends the run.</p>
            <p className="text-xs">12 linked runs · Medium priority</p>
          </div>
        </article>
      </div>
    </section>
  );
}

export function LensIntroduction({ onStart, onDemo }: { onStart: () => void; onDemo?: () => void }) {
  return (
    <section
      aria-labelledby="lens-introduction"
      className="rounded-2xl border bg-card p-5 sm:p-8 xl:rounded-3xl xl:p-10"
    >
      <h2
        id="lens-introduction"
        className="max-w-[800px] text-3xl leading-[1.12] font-semibold tracking-[-0.035em] sm:text-4xl xl:text-[42px]"
      >
        The gateway that helps your agents improve
      </h2>
      <p className="mt-5 max-w-[780px] text-base leading-7 text-muted-foreground">
        Lens turns recorded agent runs into findings tied to the exact steps that caused them, so you can see what needs
        to change without reading every trace by hand.
      </p>
      <div className="mt-7 flex flex-wrap items-center gap-3">
        {onDemo && (
          <Button size="lg" className="h-11 rounded-lg px-5" onClick={onDemo}>
            Explore with sample data <ArrowRight aria-hidden="true" className="size-4" />
          </Button>
        )}
        <Button size="lg" variant={onDemo ? "outline" : "default"} className="h-11 rounded-lg px-5" onClick={onStart}>
          Set up Lens
        </Button>
        <a
          href="https://docs.litellm.ai/docs/proxy/lens"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 px-3 text-sm font-medium text-muted-foreground underline-offset-4 hover:underline"
        >
          Docs <ArrowUpRight aria-hidden="true" className="size-4" />
        </a>
      </div>
      <GatewayFlow />
      <div className="mt-7 grid items-stretch gap-4 lg:grid-cols-2">
        <TraceExample />
        <FindingExamples />
      </div>
    </section>
  );
}
