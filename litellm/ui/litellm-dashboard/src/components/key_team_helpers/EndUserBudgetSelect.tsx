"use client";

import type { ParseKeys, TFunction } from "i18next";
import React from "react";
import { useTranslation } from "react-i18next";

import { useBudgetOptions } from "@/app/(dashboard)/hooks/budgets/useBudgetOptions";
import type { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";

export const END_USER_BUDGET_HINT: ParseKeys = "keyTeamHelpers.endUserBudgetSelect.hint";

interface EndUserBudgetSelectProps {
  readonly id?: string;
  readonly accessToken: string | null;
  readonly value: string | null;
  readonly onChange: (next: string | null) => void;
  readonly canEdit: boolean;
}

const budgetSublabel = (budget: budgetItem, t: TFunction): string | undefined => {
  const parts = [
    budget.max_budget != null ? `$${budget.max_budget}` : null,
    budget.budget_duration
      ? t("keyTeamHelpers.endUserBudgetSelect.resets", { duration: budget.budget_duration })
      : null,
  ].filter((part): part is string => part !== null);
  return parts.length > 0 ? parts.join(", ") : undefined;
};

export const EndUserBudgetSelect: React.FC<EndUserBudgetSelectProps> = ({
  id,
  accessToken,
  value,
  onChange,
  canEdit,
}) => {
  const { t } = useTranslation();
  const { data: budgets } = useBudgetOptions(accessToken, canEdit);
  const options: SearchSelectOption[] = (budgets ?? []).map((budget) => ({
    label: budget.budget_id,
    value: budget.budget_id,
    sublabel: budgetSublabel(budget, t),
  }));

  return (
    <SearchSelect
      inputId={id}
      aria-label={t("keyTeamHelpers.endUserBudgetSelect.ariaLabel")}
      placeholder={t("keyTeamHelpers.endUserBudgetSelect.placeholder")}
      emptyText={t("keyTeamHelpers.endUserBudgetSelect.empty")}
      options={options}
      value={value}
      onValueChange={onChange}
      disabled={!canEdit}
    />
  );
};
