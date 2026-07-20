import { InfoCircleOutlined } from "@ant-design/icons";
import { Tooltip, Typography } from "antd";
import React from "react";

import { formatNumberWithCommas } from "@/utils/dataUtils";
import { KeyResponse } from "./key_list";
import {
  formatModelMaxBudgetTimePeriod,
  modelMaxBudgetEntriesFromValue,
  ModelMaxBudgetValue,
  normalizeModelMaxBudgetConfig,
} from "./ModelMaxBudgetEditor";

export type ModelMaxBudgetMembership = {
  user_id: string;
  litellm_budget_table?: {
    model_max_budget?: ModelMaxBudgetValue | Record<string, unknown> | null;
  } | null;
};

export type ModelMaxBudgetOverrideRow = {
  id: string;
  label: string;
  kind: "member" | "key";
  entries: ReturnType<typeof modelMaxBudgetEntriesFromValue>;
};

export function parseModelMaxBudgetValue(
  raw: ModelMaxBudgetValue | Record<string, number> | Record<string, unknown> | null | undefined,
): ModelMaxBudgetValue | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const entries = Object.entries(raw);
  if (entries.length === 0) {
    return null;
  }

  const parsed = Object.fromEntries(
    entries.map(([model, config]) => {
      if (typeof config === "number") {
        return [model, { budget_limit: config, time_period: "1d" }];
      }
      return [model, normalizeModelMaxBudgetConfig(config as Record<string, unknown>)];
    }),
  ) as ModelMaxBudgetValue;

  return modelMaxBudgetEntriesFromValue(parsed).length > 0 ? parsed : null;
}

export function collectModelMaxBudgetOverrides(
  memberships: ModelMaxBudgetMembership[],
  keys: KeyResponse[],
): ModelMaxBudgetOverrideRow[] {
  const memberOverrides = memberships.flatMap((membership) => {
    const parsed = parseModelMaxBudgetValue(membership.litellm_budget_table?.model_max_budget);
    const entries = modelMaxBudgetEntriesFromValue(parsed);
    if (entries.length === 0) {
      return [];
    }
    return [
      {
        id: membership.user_id,
        label: membership.user_id,
        kind: "member" as const,
        entries,
      },
    ];
  });

  const keyOverrides = keys.flatMap((key) => {
    const parsed = parseModelMaxBudgetValue(key.model_max_budget);
    const entries = modelMaxBudgetEntriesFromValue(parsed);
    if (entries.length === 0) {
      return [];
    }
    return [
      {
        id: key.token_id,
        label: key.key_alias || key.key_name || key.token_id,
        kind: "key" as const,
        entries,
      },
    ];
  });

  return [...memberOverrides, ...keyOverrides];
}

function formatBudgetLine(budgetLimit: number, timePeriod: string): string {
  return `$${formatNumberWithCommas(budgetLimit, 2)} / ${formatModelMaxBudgetTimePeriod(timePeriod)}`;
}

function BudgetEntryList({
  entries,
  className,
}: {
  entries: ReturnType<typeof modelMaxBudgetEntriesFromValue>;
  className?: string;
}) {
  return (
    <div className={className}>
      {entries.map((entry) => (
        <Typography.Text key={entry.model} className="block text-xs">
          {entry.model}: {formatBudgetLine(entry.budget_limit ?? 0, entry.time_period)}
        </Typography.Text>
      ))}
    </div>
  );
}

interface ModelMaxBudgetOverviewProps {
  teamModelMaxBudget?: ModelMaxBudgetValue | null;
  memberships?: ModelMaxBudgetMembership[];
  keys?: KeyResponse[];
  variant?: "card" | "inline";
}

export function ModelMaxBudgetOverview({
  teamModelMaxBudget,
  memberships = [],
  keys = [],
  variant = "card",
}: ModelMaxBudgetOverviewProps) {
  const teamEntries = modelMaxBudgetEntriesFromValue(parseModelMaxBudgetValue(teamModelMaxBudget));
  const overrides = collectModelMaxBudgetOverrides(memberships, keys);
  const memberOverrides = overrides.filter((row) => row.kind === "member");
  const keyOverrides = overrides.filter((row) => row.kind === "key");

  const hasAnyBudgets = teamEntries.length > 0 || overrides.length > 0;

  const enforcementNote = (
    <Tooltip title="Team defaults apply to every member. Members and virtual keys can set higher per-model limits. Spend is tracked separately at team, member, and key level.">
      <InfoCircleOutlined className="ml-1 text-gray-400" />
    </Tooltip>
  );

  const content = (
    <>
      {variant === "card" && (
        <Typography.Text className="text-gray-500 text-xs">
          Enforced per member and virtual key{enforcementNote}
        </Typography.Text>
      )}
      {!hasAnyBudgets ? (
        <Typography.Text type="secondary" className={variant === "card" ? "mt-2 block" : undefined}>
          No per-model budgets configured
        </Typography.Text>
      ) : (
        <div className={variant === "card" ? "mt-2 space-y-3" : "space-y-2"}>
          {teamEntries.length > 0 && (
            <div>
              <Typography.Text className={variant === "card" ? "text-gray-500" : "font-medium"}>
                Team defaults
              </Typography.Text>
              <BudgetEntryList entries={teamEntries} className="mt-1" />
            </div>
          )}
          {memberOverrides.length > 0 && (
            <div>
              <Typography.Text className={variant === "card" ? "text-gray-500" : "font-medium"}>
                Member overrides ({memberOverrides.length})
              </Typography.Text>
              {memberOverrides.map((row) => (
                <div key={row.id} className="mt-1">
                  <Typography.Text className="block text-xs font-medium">{row.label}</Typography.Text>
                  <BudgetEntryList entries={row.entries} className="ml-2" />
                </div>
              ))}
            </div>
          )}
          {keyOverrides.length > 0 && (
            <div>
              <Typography.Text className={variant === "card" ? "text-gray-500" : "font-medium"}>
                Virtual key overrides ({keyOverrides.length})
              </Typography.Text>
              {keyOverrides.map((row) => (
                <div key={row.id} className="mt-1">
                  <Typography.Text className="block text-xs font-medium">{row.label}</Typography.Text>
                  <BudgetEntryList entries={row.entries} className="ml-2" />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </>
  );

  if (variant === "inline") {
    return (
      <div>
        <Typography.Text className="font-medium">
          Per-Model Budgets
          {enforcementNote}
        </Typography.Text>
        {content}
      </div>
    );
  }

  return content;
}
