import { useState } from "react";
import { Check } from "lucide-react";
import { cn } from "@/lib/cva.config";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { problemsOverview, type AgentSummary, type Problem } from "../model/problems";
import { runTime } from "../model/format";
import type { Lens } from "../model/types";
import { DotFlow, dotColors } from "../DotFlow";

function AgentTile({
  label,
  detail,
  selected,
  onSelect,
}: {
  label: string;
  detail: string;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onSelect}
      className={cn(
        "relative flex h-[4.5rem] flex-col justify-center rounded-xl px-3.5 text-left transition-transform active:scale-[0.97] motion-reduce:transition-none",
        selected
          ? "bg-background ring-[1.5px] ring-inset ring-foreground"
          : "bg-muted/60 text-muted-foreground hover:bg-muted",
      )}
    >
      <span className="truncate pr-5 text-sm font-medium">{label}</span>
      <span className="mt-0.5 text-xs text-muted-foreground">{detail}</span>
      {selected && (
        <Check
          aria-hidden="true"
          className="absolute top-2.5 right-2.5 size-3.5 animate-in fade-in zoom-in-50 duration-200 motion-reduce:animate-none"
        />
      )}
    </button>
  );
}

function Frequency({ problem, onOpen }: { problem: Problem; onOpen: (executionId: string) => void }) {
  return (
    <div className="flex items-center gap-3">
      <TooltipProvider>
        <div className="flex flex-wrap items-center gap-1" aria-hidden="true">
          {problem.runs.map((run) => (
            <Tooltip key={run.executionId}>
              <TooltipTrigger
                render={
                  <button
                    type="button"
                    tabIndex={-1}
                    onClick={(e) => {
                      e.stopPropagation();
                      onOpen(run.executionId);
                    }}
                    className={cn(
                      "size-2 rounded-full transition-transform hover:scale-150",
                      run.hit ? "bg-foreground" : "bg-muted-foreground/25",
                    )}
                  />
                }
              />
              <TooltipContent className="whitespace-normal">
                <span>
                  <span className="block">{runTime(run.startTime)}</span>
                  {run.hit && run.quote && <span className="mt-1 block opacity-80">&ldquo;{run.quote}&rdquo;</span>}
                </span>
              </TooltipContent>
            </Tooltip>
          ))}
        </div>
      </TooltipProvider>
      <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
        {problem.hits} of {problem.runs.length || problem.hits}
      </span>
    </div>
  );
}

export function ProblemsTable({
  lenses,
  onOpenProblem,
  onOpenRun,
}: {
  lenses: Lens[];
  onOpenProblem: (problem: Problem) => void;
  onOpenRun: (lensId: string, executionId: string) => void;
}) {
  const [agent, setAgent] = useState<string | null>(null);
  const overview = problemsOverview(lenses);
  if (!overview.runs) return null;
  const shown = overview.problems.filter((p) => !agent || p.agent === agent);
  const tiles: readonly (AgentSummary & { key: string | null })[] = [
    { key: null, agent: "all", runs: overview.runs, affected: overview.affected },
    ...overview.agents.map((a) => ({ ...a, key: a.agent })),
  ];
  const scope = agent ? overview.agents.find((a) => a.agent === agent) : undefined;
  const clean = (scope?.runs ?? overview.runs) - (scope?.affected ?? overview.affected);
  const agentColors = overview.agents.map((a, i) => ({ agent: a.agent, color: dotColors[i % dotColors.length] }));
  const flowColors = agentColors.filter((a) => !agent || a.agent === agent).map((a) => a.color);
  return (
    <section aria-labelledby="lens-problems" className="space-y-5">
      <DotFlow active={flowColors} />
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="lens-problems" className="text-lg font-semibold">
          Problems
        </h2>
        <p className="text-xs text-muted-foreground">
          {overview.problems.length} {overview.problems.length === 1 ? "problem" : "problems"} in {overview.affected} of{" "}
          {overview.runs} runs
        </p>
      </div>
      <div role="group" aria-label="Filter by agent" className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
        {tiles.map((tile) => (
          <AgentTile
            key={tile.agent}
            label={tile.agent}
            detail={tile.key === null ? `${tile.runs} runs` : `${tile.affected} of ${tile.runs} runs`}
            selected={agent === tile.key}
            onSelect={() => setAgent(tile.key)}
          />
        ))}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th className="py-2 pr-4 font-medium">Problem</th>
              <th className="w-24 py-2 pr-4 font-medium">Severity</th>
              <th className="w-[34%] py-2 pr-4 font-medium">Frequency</th>
              <th className="w-40 py-2 font-medium">Agent</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {shown.map((problem) => (
              <tr
                key={`${problem.lensId}:${problem.findingId}`}
                onClick={() => onOpenProblem(problem)}
                className="cursor-pointer align-top hover:bg-muted/30"
              >
                <td className="py-3.5 pr-4">
                  <button
                    type="button"
                    className="text-left font-medium focus-visible:outline-2 focus-visible:outline-ring"
                    onClick={(e) => {
                      e.stopPropagation();
                      onOpenProblem(problem);
                    }}
                  >
                    {problem.title}
                  </button>
                  <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">{problem.quote}</p>
                </td>
                <td className="py-3.5 pr-4 text-muted-foreground">{problem.priority}</td>
                <td className="py-3.5 pr-4">
                  <Frequency problem={problem} onOpen={(id) => onOpenRun(problem.lensId, id)} />
                </td>
                <td className="truncate py-3.5 text-muted-foreground">{problem.agent}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!shown.length && (
          <p className="border-b py-10 text-center text-sm text-muted-foreground">
            No open problems{agent ? ` for ${agent}` : ""} in the latest investigations.
          </p>
        )}
      </div>
      {clean > 0 && (
        <p className="text-xs text-muted-foreground">
          {clean} {clean === 1 ? "run" : "runs"} had no problems
        </p>
      )}
    </section>
  );
}
