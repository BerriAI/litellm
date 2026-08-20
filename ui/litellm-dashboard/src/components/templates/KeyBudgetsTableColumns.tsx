"use client";

import type { ColumnDef } from "@tanstack/react-table";

import type { KeyBudgetEntry, KeyBudgetNote, KeyBudgetNoteCode } from "@/app/(dashboard)/hooks/keys/useKeyBudgets";
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
 * Keyed by `code` rather than by `severity` because severity does not track this reliably, and
 * exhaustive over the union so a code added server-side fails this build until it is classified.
 * `alert_only` is not dead, it restates the `enforcement` column and a soft budget that is over
 * still outranks healthy rows. `end_user_route_only` is not dead either, it scopes which requests
 * the row applies to.
 */
const CODE_KILLS_ROW: Readonly<Record<KeyBudgetNoteCode, boolean>> = {
  alert_only: false,
  custom_auth_may_override_end_user_cap: false,
  end_user_route_only: false,
  model_budget_fails_open: true,
  per_model_counters: false,
  project_spend_not_tracked: true,
  request_tags_add_budgets: false,
  reservation_blocks_at_limit: false,
  rolling_window: false,
  throttled_instead_of_blocked: false,
  user_budget_not_applied_to_team_key: true,
};

/** Severity is the fallback for a code this build predates, never the primary signal. */
const noteKillsRow = (note: KeyBudgetNote): boolean => {
  const classified: boolean | undefined = CODE_KILLS_ROW[note.code];
  return classified ?? note.severity === "info";
};

export const cannotTrip = (entry: KeyBudgetEntry): boolean => entry.notes.some(noteKillsRow);

export const isBlockingRow = (entry: KeyBudgetEntry): boolean =>
  entry.status === "exceeded" && !isAlertOnly(entry) && !cannotTrip(entry);

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
export const budgetThresholdRule = (entry: KeyBudgetEntry): string | null => {
  if (entry.max_budget == null) return null;
  const threshold = `${COMPARISON_GLYPH[entry.comparison] ?? entry.comparison} $${formatNumberWithCommas(entry.max_budget, 2)}`;
  return isAlertOnly(entry) ? `Alerts at ${threshold}` : `Blocks at ${threshold}`;
};

const statusPresentation = (entry: KeyBudgetEntry): { tone: StatusTone; label: string } => {
  if (entry.status === "unlimited") return { tone: "neutral", label: "Unlimited" };
  if (cannotTrip(entry)) return { tone: "neutral", label: "Cannot trip" };
  if (entry.status !== "exceeded") return { tone: "success", label: "Within budget" };
  return isAlertOnly(entry)
    ? { tone: "warning", label: "Exceeded (alert only)" }
    : { tone: "error", label: "Exceeded" };
};

function ScopeCell({ entry }: { entry: KeyBudgetEntry }) {
  const entity = entry.entity_label || entry.entity_id;
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

function EnforcementCell({ entry }: { entry: KeyBudgetEntry }) {
  return isAlertOnly(entry) ? (
    <StatusBadge
      tone="neutral"
      label="Alert only"
      tooltip="Soft budget. Going over raises an alert and never rejects a request."
    />
  ) : (
    <StatusBadge tone="info" label="Blocks requests" tooltip="Going over this budget rejects requests on this key." />
  );
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
