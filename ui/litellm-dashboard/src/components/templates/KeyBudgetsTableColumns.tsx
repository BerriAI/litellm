"use client";

import type { ColumnDef } from "@tanstack/react-table";

import type { KeyBudgetEntry } from "@/app/(dashboard)/hooks/keys/useKeyBudgets";
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

export const isBlockingRow = (entry: KeyBudgetEntry): boolean => entry.status === "exceeded" && !isAlertOnly(entry);

const STATUS_ORDER: Record<string, number> = { exceeded: 0, ok: 1, unlimited: 2 };

export const severityRank = (entry: KeyBudgetEntry): number =>
  (STATUS_ORDER[entry.status] ?? STATUS_ORDER.unlimited) * 2 + (isAlertOnly(entry) ? 1 : 0);

const COMPARISON_GLYPH: Record<string, string> = { ">=": "≥", ">": ">" };

/**
 * Scopes disagree on whether hitting the limit exactly is over it: team_member enforces `>=`
 * so 50 of 50 is already denied, while team enforces `>` so 300 of 300 still passes. Two rows
 * can therefore show identical numbers and opposite statuses, so state each row's real threshold.
 */
export const budgetThresholdRule = (entry: KeyBudgetEntry): string | null => {
  if (entry.max_budget == null) return null;
  const threshold = `${COMPARISON_GLYPH[entry.comparison] ?? entry.comparison} $${formatNumberWithCommas(entry.max_budget, 2)}`;
  return isAlertOnly(entry) ? `Alerts at ${threshold}` : `Blocks at ${threshold}`;
};

const statusPresentation = (entry: KeyBudgetEntry): { tone: StatusTone; label: string } => {
  if (entry.status === "unlimited") return { tone: "neutral", label: "Unlimited" };
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
      {entity && <span className="truncate font-mono text-xs text-muted-foreground">{entity}</span>}
      {entry.note && (
        <span className="truncate text-xs text-muted-foreground italic" title={entry.note}>
          {entry.note}
        </span>
      )}
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
    size: 240,
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
    cell: ({ row }) => {
      const rule = budgetThresholdRule(row.original);
      return (
        <div className="flex flex-col gap-0.5">
          <SpendBudgetCell spend={row.original.spend} maxBudget={row.original.max_budget} budgetDecimals={2} />
          {rule && <span className="whitespace-nowrap text-xs tabular-nums text-muted-foreground">{rule}</span>}
        </div>
      );
    },
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
