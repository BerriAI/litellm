"use client";

import type { ColumnDef } from "@tanstack/react-table";

import type {
  KeyBudgetEnforcement,
  KeyBudgetEntry,
  KeyBudgetNote,
  KeyBudgetNoteCode,
} from "@/app/(dashboard)/hooks/keys/useKeyBudgets";
import {
  CellTooltip,
  DateCell,
  MoneyCell,
  SpendBudgetCell,
  StatusBadge,
  type StatusTone,
} from "@/components/shared/table_cells";
import { formatNumberWithCommas } from "@/utils/dataUtils";

const SCOPE_LABELS: Record<string, string> = {
  proxy: "Proxy",
  key: "Key",
  key_window: "Key window",
  key_model: "Key per-model",
  team: "Team",
  team_window: "Team window",
  team_member: "Team member",
  user: "User",
  organization: "Organization",
  project: "Project",
  tag: "Tag",
  end_user: "End user",
  end_user_model: "End user per-model",
};

const isAlertOnly = (entry: KeyBudgetEntry): boolean => entry.enforcement === "soft";

/**
 * Whether a note means the row is dead: it cannot reject a request no matter what the numbers say.
 * Only a permanent property of this row counts. `custom_auth_skips_read_time_checks` is deliberately
 * not dead even though it says budgets go unchecked, because the reservation layer still enforces
 * the scopes it covers, so which rows survive depends on scope rather than on the code. Encoding
 * that split here would duplicate the resolver's coverage set and drift from it.
 *
 * Deadness is a property of the code alone. Severity does not imply it in either direction, since
 * dead codes appear under both values, so an unclassified code is assumed live: calling a row dead
 * when it is not invites dismissing the budget that actually stopped the request, which is the one
 * failure this table exists to prevent. Exhaustive over the union, so a code added server-side
 * fails this build until someone classifies it.
 */
const CODE_KILLS_ROW: Readonly<Record<KeyBudgetNoteCode, boolean>> = {
  alert_only: false,
  custom_auth_may_override_end_user_cap: false,
  custom_auth_skips_read_time_checks: false,
  end_user_route_only: false,
  per_model_counters: false,
  project_spend_not_tracked: true,
  request_tags_add_budgets: false,
  reservation_blocks_at_limit: false,
  rolling_window: false,
  throttled_instead_of_blocked: false,
  user_budget_not_applied_to_team_key: true,
};

const noteKillsRow = (note: KeyBudgetNote): boolean => {
  const classified: boolean | undefined = CODE_KILLS_ROW[note.code];
  return classified ?? false;
};

export const cannotTrip = (entry: KeyBudgetEntry): boolean => entry.notes.some(noteKillsRow);

/** Exceeding a throttled budget slows requests rather than rejecting them, so it never denies one. */
export const isThrottled = (entry: KeyBudgetEntry): boolean => entry.enforcement === "throttled";

/** Whether going over this budget rejects a request, as opposed to alerting, throttling or nothing. */
const canDeny = (entry: KeyBudgetEntry): boolean => !isAlertOnly(entry) && !cannotTrip(entry) && !isThrottled(entry);

export const isBlockingRow = (entry: KeyBudgetEntry): boolean => entry.status === "exceeded" && canDeny(entry);

/** Ascending relevance to "what stopped my request", so a row that cannot answer it sorts last. */
export const rowRank = (entry: KeyBudgetEntry): number => {
  if (entry.status === "unlimited") return 3;
  if (cannotTrip(entry)) return 4;
  if (isBlockingRow(entry)) return 0;
  if (entry.status === "exceeded") return 1;
  return 2;
};

/**
 * Only `live` and `no_counter` carry a number worth drawing. Any other state, including one the
 * server adds after this ships, is rendered as unknown rather than as a confident zero: overstating
 * a spend is the failure that matters here, and a new state is by definition not the normal one.
 */
const spendIsReadable = (entry: KeyBudgetEntry): boolean =>
  entry.spend_state === "live" || entry.spend_state === "no_counter";

const COMPARISON_GLYPH: Record<string, string> = { ">=": "≥", ">": ">" };

/**
 * Scopes disagree on whether hitting the limit exactly is already over it, and a scope's operator
 * is not fixed: budget reservation tightens some scopes to `>=`, and disabling it relaxes them back
 * to `>`. So this reads `comparison` off each row rather than assuming a constant per scope. Two
 * rows can show identical numbers and opposite statuses, so state the threshold each one enforces.
 */
const thresholdVerb = (entry: KeyBudgetEntry): string => {
  if (isAlertOnly(entry)) return "Alerts";
  return isThrottled(entry) ? "Throttles" : "Blocks";
};

export const budgetThresholdRule = (entry: KeyBudgetEntry): string | null => {
  if (entry.max_budget == null) return null;
  const threshold = `${COMPARISON_GLYPH[entry.comparison] ?? entry.comparison} $${formatNumberWithCommas(entry.max_budget, 2)}`;
  return `${thresholdVerb(entry)} at ${threshold}`;
};

const statusPresentation = (entry: KeyBudgetEntry): { tone: StatusTone; label: string } => {
  if (entry.status === "unlimited") return { tone: "neutral", label: "Unlimited" };
  if (cannotTrip(entry)) return { tone: "neutral", label: "Cannot trip" };
  if (entry.status !== "exceeded") return { tone: "success", label: "Within budget" };
  if (isAlertOnly(entry)) return { tone: "warning", label: "Exceeded (alert only)" };
  return isThrottled(entry)
    ? { tone: "warning", label: "Exceeded (throttling)" }
    : { tone: "error", label: "Exceeded" };
};

function ScopeCell({ entry }: { entry: KeyBudgetEntry }) {
  const entity = entry.entity_label || entry.entity_id;
  // Per-model rows split one cap across every request model that routes onto it, so `entity_id` is
  // what tells two rows apart while `entity_label` repeats the cap. Showing only the label would
  // render them as duplicates.
  const measured = entry.entity_label && entry.entity_id !== entry.entity_label ? entry.entity_id : null;
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <CellTooltip
        content={`Limit source: ${entry.source}`}
        trigger={
          <span className="w-fit truncate text-sm font-medium text-foreground">
            {SCOPE_LABELS[entry.scope] ?? entry.scope}
          </span>
        }
      />
      {entity && (
        <span className="truncate font-mono text-xs text-muted-foreground" title={entity}>
          {entity}
        </span>
      )}
      {measured && (
        <span className="truncate font-mono text-xs text-muted-foreground/70" title={measured}>
          {measured}
        </span>
      )}
      {entry.notes.map((note) => (
        <span
          key={note.code}
          className={
            note.severity === "warning" ? "text-xs italic text-amber-600" : "text-xs italic text-muted-foreground"
          }
        >
          {note.text}
        </span>
      ))}
    </div>
  );
}

/** Exhaustive, so a fourth enforcement mode fails this build rather than defaulting to a claim. */
const ENFORCEMENT_BADGE: Readonly<Record<KeyBudgetEnforcement, { tone: StatusTone; label: string; tooltip: string }>> =
  {
    soft: {
      tone: "neutral",
      label: "Alert only",
      tooltip: "Soft budget. Going over raises an alert and never rejects a request.",
    },
    throttled: {
      tone: "warning",
      label: "Throttles requests",
      tooltip: "This key opted into throttle_on_budget_exceeded, so going over reduces its rate limits.",
    },
    hard: { tone: "info", label: "Blocks requests", tooltip: "Going over this budget rejects requests on this key." },
  };

function EnforcementCell({ entry }: { entry: KeyBudgetEntry }) {
  const { tone, label, tooltip } = ENFORCEMENT_BADGE[entry.enforcement];
  return <StatusBadge tone={tone} label={label} tooltip={tooltip} />;
}

function SpendCell({ entry }: { entry: KeyBudgetEntry }) {
  const rule = budgetThresholdRule(entry);
  return (
    <div className="flex flex-col gap-0.5">
      {spendIsReadable(entry) ? (
        <SpendBudgetCell spend={entry.spend} maxBudget={entry.max_budget} budgetDecimals={2} />
      ) : (
        <span className="whitespace-nowrap text-xs">
          <span className="font-medium text-amber-600">Unknown</span>{" "}
          <span className="text-muted-foreground">
            {entry.max_budget == null ? "· Unlimited" : `of $${formatNumberWithCommas(entry.max_budget, 2)}`}
          </span>
        </span>
      )}
      {rule && <span className="whitespace-nowrap text-xs tabular-nums text-muted-foreground">{rule}</span>}
    </div>
  );
}

function RemainingCell({ entry }: { entry: KeyBudgetEntry }) {
  const unlimited = entry.max_budget == null;
  return (
    <MoneyCell
      value={unlimited ? null : entry.remaining}
      decimals={2}
      emptyText={unlimited ? "Unlimited" : "-"}
      showZero
    />
  );
}

export const getKeyBudgetsTableColumns = (): ColumnDef<KeyBudgetEntry>[] => [
  {
    id: "scope",
    meta: { title: "Scope" },
    header: "Scope",
    size: 300,
    enableSorting: false,
    cell: ({ row }) => <ScopeCell entry={row.original} />,
  },
  {
    id: "enforcement",
    meta: { title: "Enforcement" },
    header: "Enforcement",
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <EnforcementCell entry={row.original} />,
  },
  {
    id: "spend",
    meta: { title: "Spend / Limit" },
    header: "Spend / Limit",
    size: 200,
    enableSorting: false,
    cell: ({ row }) => <SpendCell entry={row.original} />,
  },
  {
    id: "remaining",
    meta: { title: "Remaining", className: "text-right", headerClassName: "text-right" },
    header: "Remaining",
    size: 120,
    enableSorting: false,
    cell: ({ row }) => <RemainingCell entry={row.original} />,
  },
  {
    id: "status",
    meta: { title: "Status" },
    header: "Status",
    size: 170,
    enableSorting: false,
    cell: ({ row }) => {
      const { tone, label } = statusPresentation(row.original);
      return (
        <StatusBadge
          tone={tone}
          label={label}
          tooltip={budgetThresholdRule(row.original)}
          dataTestId={isBlockingRow(row.original) ? "key-budget-blocking" : undefined}
        />
      );
    },
  },
  {
    id: "resets",
    meta: { title: "Resets" },
    header: "Resets",
    size: 150,
    enableSorting: false,
    cell: ({ row }) => (
      <div className="flex flex-col gap-0.5">
        <DateCell value={row.original.budget_reset_at} precision="date" fallback="Never" />
        {row.original.budget_duration && (
          <span className="text-xs text-muted-foreground">Every {row.original.budget_duration}</span>
        )}
      </div>
    ),
  },
];
