"use client";

import type { KeyBudgetEntry } from "@/app/(dashboard)/hooks/keys/useKeyBudgets";
import { CellTooltip } from "@/components/shared/table_cells";
import { cn } from "@/lib/cva.config";
import { formatNumberWithCommas } from "@/utils/dataUtils";

import { budgetThresholdRule, cannotTrip, isBlockingRow, scopeLabel } from "./KeyBudgetsTableColumns";

/**
 * Fraction of its limit a row has spent, or null when the row has no limit or no reading.
 *
 * A row without both numbers cannot be drawn at all, and drawing it at zero would read as untouched
 * headroom on a budget that may be exhausted.
 */
export const utilization = (entry: KeyBudgetEntry): number | null => {
  if (entry.max_budget == null || entry.max_budget <= 0 || entry.spend == null) return null;
  return entry.spend / entry.max_budget;
};

/** Rows worth plotting: a real limit, a real reading, and the ability to reject a request. */
export const plottable = (entries: readonly KeyBudgetEntry[]): readonly KeyBudgetEntry[] =>
  [...entries]
    .filter((entry) => utilization(entry) != null && !cannotTrip(entry))
    .sort((a, b) => (utilization(b) ?? 0) - (utilization(a) ?? 0));

const BAND_BOUNDS = { comfortable: 0.7, tight: 0.9 } as const;

const measureTone = (fraction: number, blocking: boolean): string => {
  if (blocking) return "bg-red-500";
  if (fraction >= BAND_BOUNDS.tight) return "bg-amber-500";
  return "bg-sky-500";
};

const percentLabel = (fraction: number): string => {
  const percent = fraction * 100;
  if (percent > 0 && percent < 1) return "<1%";
  return `${formatNumberWithCommas(percent, percent >= 10 ? 0 : 1)}%`;
};

/**
 * One bullet graph row: qualitative bands behind a measure bar, with the limit as the track's end.
 *
 * The bands are what carry "how close is close", which is the question a spend column next to a
 * limit column makes the reader answer by subtraction.
 */
function BulletRow({ entry }: { entry: KeyBudgetEntry }) {
  const fraction = utilization(entry) ?? 0;
  const blocking = isBlockingRow(entry);
  const rule = budgetThresholdRule(entry);
  return (
    <div className="grid grid-cols-[10rem_1fr_9rem_3.5rem] items-center gap-3 text-xs">
      <span className="truncate font-medium text-foreground" title={scopeLabel(entry)}>
        {scopeLabel(entry)}
      </span>
      <CellTooltip
        content={rule ?? undefined}
        trigger={
          <div className="relative h-3 w-full overflow-hidden rounded-sm bg-muted">
            <div
              className="absolute inset-y-0 left-0 bg-amber-100"
              style={{ left: `${BAND_BOUNDS.comfortable * 100}%`, right: `${(1 - BAND_BOUNDS.tight) * 100}%` }}
            />
            <div
              className="absolute inset-y-0 right-0 bg-red-100"
              style={{ width: `${(1 - BAND_BOUNDS.tight) * 100}%` }}
            />
            <div
              className={cn("absolute inset-y-0.5 left-0 rounded-sm", measureTone(fraction, blocking))}
              style={{ width: `${Math.min(fraction, 1) * 100}%` }}
              data-testid={blocking ? "key-budget-bullet-blocking" : "key-budget-bullet"}
            />
            <div className="absolute inset-y-0 right-0 w-0.5 bg-foreground/70" />
          </div>
        }
      />
      <span className="truncate tabular-nums text-muted-foreground">
        ${formatNumberWithCommas(entry.spend ?? 0, 2)} of ${formatNumberWithCommas(entry.max_budget ?? 0, 2)}
      </span>
      <span className={cn("text-right tabular-nums", blocking ? "font-medium text-red-600" : "text-muted-foreground")}>
        {percentLabel(fraction)}
      </span>
    </div>
  );
}

/**
 * The one-line answer to "what stopped my request", stated before the evidence under it.
 *
 * An unreadable scope is named here rather than left to the table, because a verdict that ignores
 * the rows nobody could read is a verdict that rules them out.
 */
function Verdict({ budgets }: { budgets: readonly KeyBudgetEntry[] }) {
  const blocking = budgets.filter(isBlockingRow);
  const unknown = budgets.filter((entry) => entry.status === "unknown");
  const closest = plottable(budgets).find((entry) => !isBlockingRow(entry));
  const closestFraction = closest ? utilization(closest) : null;

  if (blocking.length > 0) {
    return (
      <p className="text-sm font-medium text-red-600" data-testid="key-budgets-verdict">
        Blocked by {blocking.map(scopeLabel).join(", ")}
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-0.5" data-testid="key-budgets-verdict">
      <p className="text-sm font-medium text-foreground">Nothing is blocking this key.</p>
      {closest && closestFraction != null && (
        <p className="text-xs text-muted-foreground">
          Closest to its limit: {scopeLabel(closest)}, {percentLabel(closestFraction)} used.
        </p>
      )}
      {unknown.length > 0 && (
        <p className="text-xs text-amber-600">
          {unknown.length} {unknown.length === 1 ? "scope" : "scopes"} could not be read, so nothing on{" "}
          {unknown.length === 1 ? "it" : "them"} can be ruled out: {unknown.map(scopeLabel).join(", ")}.
        </p>
      )}
    </div>
  );
}

export function KeyBudgetsBulletChart({ budgets }: { budgets: readonly KeyBudgetEntry[] }) {
  const rows = plottable(budgets);
  return (
    <div className="flex flex-col gap-3 rounded-md border border-border bg-card p-4" data-testid="key-budgets-chart">
      <Verdict budgets={budgets} />
      {rows.length > 0 && (
        <div className="flex flex-col gap-2">
          {rows.map((entry) => (
            <BulletRow key={`${entry.scope}:${entry.entity_id ?? ""}`} entry={entry} />
          ))}
        </div>
      )}
    </div>
  );
}
