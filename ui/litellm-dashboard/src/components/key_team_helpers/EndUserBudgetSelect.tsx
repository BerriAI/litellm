"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import React from "react";

import { useBudgetOptions } from "@/app/(dashboard)/hooks/budgets/useBudgetOptions";
import type { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import { SearchSelect, type SearchSelectOption } from "@/components/shared/SearchSelect";

export const END_USER_BUDGET_HINT =
  "Reusable budget applied to every new customer (end user) this key creates via `user` or x-litellm-end-user-id. " +
  "Overrides the proxy-wide max_end_user_budget_id; customers that already have their own budget keep it.";

interface EndUserBudgetSelectProps {
  readonly id?: string;
  readonly accessToken: string | null;
  readonly value: string | null;
  readonly onChange: (next: string | null) => void;
  readonly canEdit: boolean;
}

const budgetSublabel = (budget: budgetItem, t: ReturnType<typeof useLanguage>["t"]): string | undefined => {
  const parts = [
    budget.max_budget != null ? `$${budget.max_budget}` : null,
    budget.budget_duration ? t("resets {duration}", { duration: budget.budget_duration }) : null,
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
  const { t } = useLanguage();
  const { data: budgets } = useBudgetOptions(accessToken, canEdit);
  const options: SearchSelectOption[] = (budgets ?? []).map((budget) => ({
    label: budget.budget_id,
    value: budget.budget_id,
    sublabel: budgetSublabel(budget, t),
  }));

  return (
    <SearchSelect
      inputId={id}
      aria-label={t("Default Customer Budget")}
      placeholder={t("No default budget")}
      emptyText={t("No budgets found. Create one under Budgets.")}
      options={options}
      value={value}
      onValueChange={onChange}
      disabled={!canEdit}
    />
  );
};
