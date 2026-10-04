import { Aperture, ArrowRight, ArrowUpRight, GitBranch } from "lucide-react";
import { Button } from "@/components/ui/button";
import { scenarios } from "../demo/scenarios";

export function LensIntroduction({ onStart, onDemo }: { onStart: () => void; onDemo?: () => void }) {
  const example = scenarios[1];
  return (
    <section aria-labelledby="lens-introduction" className="rounded-lg border bg-card p-5 sm:p-8">
      <h2 id="lens-introduction" className="max-w-2xl text-2xl font-semibold tracking-tight sm:text-3xl">
        The gateway that helps your agents improve
      </h2>
      <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">
        Lens reviews your agents’ recorded activity and finds behavior worth fixing. Each finding links to the original
        steps, so you can see what happened and what to change next.
      </p>
      <div className="mt-6 flex flex-wrap items-center gap-3">
        {onDemo && (
          <Button onClick={onDemo}>
            Explore with sample data <ArrowRight aria-hidden="true" className="size-4" />
          </Button>
        )}
        <Button variant={onDemo ? "outline" : "default"} onClick={onStart}>
          Set up Lens
        </Button>
        <a
          href="https://docs.litellm.ai/docs/proxy/lens"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 px-2 text-sm text-muted-foreground underline-offset-4 hover:underline"
        >
          Docs <ArrowUpRight aria-hidden="true" className="size-3.5" />
        </a>
      </div>
      <div className="mt-8 grid overflow-hidden rounded-lg border lg:grid-cols-2">
        <div className="border-b bg-muted/20 p-5 lg:border-r lg:border-b-0">
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <h3 className="flex items-center gap-2 font-medium">
              <GitBranch aria-hidden="true" className="size-4" />
              Agent trace
            </h3>
            <span className="text-xs text-muted-foreground">Sample · {example.agent}</span>
          </div>
          <dl className="mt-5 space-y-4 text-sm">
            <div>
              <dt className="font-mono text-xs text-muted-foreground">{example.tool}</dt>
              <dd className="mt-1.5 leading-6">{example.result}</dd>
            </div>
            <div className="border-t pt-4">
              <dt className="text-xs text-muted-foreground">Final answer</dt>
              <dd className="mt-1.5 leading-6">“{example.answer}”</dd>
            </div>
          </dl>
        </div>
        <div className="p-5">
          <h3 className="flex items-center gap-2 text-sm font-medium">
            <Aperture aria-hidden="true" className="size-4" />
            Lens finding
          </h3>
          <h4 className="mt-5 text-base font-medium">Performance claim has no supporting benchmark</h4>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            The answer claims a 40% improvement, but the retrieved documentation contains no comparative benchmark.
          </p>
          <p className="mt-4 border-t pt-4 text-sm leading-6">
            <span className="font-medium">Suggested change: </span>Require a benchmark source for numerical performance
            claims, or remove the comparison.
          </p>
        </div>
      </div>
    </section>
  );
}
