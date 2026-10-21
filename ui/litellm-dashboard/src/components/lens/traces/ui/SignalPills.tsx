import { cn } from "@/lib/cva.config";

import type { SignalFlag } from "../types";

const percent = (score: number): string => `${Math.round(score * 100)}%`;

export const signalSummary = (flags: readonly SignalFlag[]): string =>
  flags.map((flag) => `${flag.name} (${percent(flag.score)})`).join(", ");

function SignalPill({ flag, showScore }: { flag: SignalFlag; showScore: boolean }) {
  return (
    <span className="inline-flex max-w-full min-w-0 items-center gap-1 rounded-full bg-destructive/10 px-1.5 py-0.5 text-xs leading-none font-medium text-destructive">
      <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-destructive" />
      <span className="truncate">{flag.name}</span>
      {showScore && <span className="shrink-0 font-normal tabular-nums opacity-80">{percent(flag.score)}</span>}
    </span>
  );
}

export function SignalPills({
  flags,
  showScore = false,
  className,
}: {
  flags: readonly SignalFlag[];
  showScore?: boolean;
  className?: string;
}) {
  return (
    <span
      role="list"
      aria-label="Signals"
      title={`Signals: ${signalSummary(flags)}`}
      className={cn("inline-flex max-w-full min-w-0 items-center gap-1", className)}
    >
      {flags.map((flag) => (
        <span role="listitem" key={flag.signal_id} className="min-w-0">
          <SignalPill flag={flag} showScore={showScore} />
        </span>
      ))}
    </span>
  );
}
