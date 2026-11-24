import { useTranslation } from "react-i18next";
import { Fragment, type ComponentProps, type ReactNode } from "react";
import { CustomerBudgetFields } from "./CustomerBudgetFields";
import CustomerBudgetCompactFields from "./CustomerBudgetCompactFields";
import type { BudgetFields } from "./customerBudgetForm";
import { rateFields } from "./CustomerRateLimits";
import { Select, SelectContent, SelectItem, SelectSeparator, SelectTrigger, SelectValue } from "@/components/ui/select";
import { customerBudgetLimitLabel } from "./customerBudgetUsage";
import type { Organization } from "./customerHierarchyForms";
import type { CustomerOrganizationFields } from "./customerOrganizationForm";

export type CustomerLimits = Pick<
  CustomerOrganizationFields,
  "budgetMode" | "budgetId" | "budgetAmount" | "budgetDuration" | "tpm" | "rpm" | "softBudget" | "allowedModels"
>;

const currencyOptions = { style: "currency", minimumFractionDigits: 0, maximumFractionDigits: 2 } as const;

function LimitOptionContent({ label, summary }: { label: string; summary: string }) {
  return (
    <span className="flex min-w-0 flex-1 flex-col items-start gap-1 whitespace-normal sm:flex-row sm:items-center sm:justify-between sm:gap-4">
      <span className="font-medium">{label}</span>
      <span className="text-sm text-muted-foreground sm:text-right">{summary}</span>
    </span>
  );
}

export function CustomerBudgetSourceSelect({
  profiles,
  value,
  onChange,
  title,
  disabled,
  loading = false,
  inheritedOption,
  noTemplate = false,
  custom = true,
  unavailableSelection,
}: {
  profiles: Organization["default_budgets"];
  value: string;
  onChange: (value: string | null) => void;
  title: string;
  disabled: boolean;
  loading?: boolean;
  inheritedOption?: { label: string; summary: string };
  noTemplate?: boolean;
  custom?: boolean;
  unavailableSelection?: { value: string; label: string; summary: string };
}) {
  const { t } = useTranslation();
  const profileOptions = profiles.map((group) => {
    const spendLimits = [
      ...(group.max_budget == null
        ? []
        : [{ max_budget: group.max_budget, budget_duration: group.budget_duration ?? null }]),
      ...(group.budget_limits ?? []),
    ];
    const spendSummary = spendLimits.map((limit) => {
      const amount = t("format.number", { val: limit.max_budget, ...currencyOptions });
      const label = customerBudgetLimitLabel(amount, limit.budget_duration ?? null, t);
      return limit.budget_duration ? label : `${label} · ${t("customerPortal.analytics.noAutomaticReset")}`;
    });
    const rateSummary = rateFields.map((field) =>
      group[`${field}_limit`] == null
        ? t(`customerPortal.createDialog.${field}NoLimit`)
        : t(`customerPortal.detail.${field}`, {
            val: t("format.compactNumber", { val: group[`${field}_limit`] }),
          }),
    );
    return {
      value: `profile:${group.budget_id}`,
      budgetId: group.budget_id,
      label: group.name,
      summary: [
        ...(spendSummary.length ? spendSummary : [t("customerPortal.rateLimits.noSpendLimit")]),
        ...rateSummary,
        ...(group.allowed_models?.length
          ? [`${t("customerPortal.hierarchy.allowedModels")}: ${group.allowed_models.join(", ")}`]
          : []),
      ].join(" · "),
    };
  });
  const customOption = {
    value: "custom",
    label: t("customerPortal.createDialog.customLimits"),
    summary: t("customerPortal.createDialog.customLimitsHelp"),
  };
  const noTemplateOption = {
    value: "none",
    label: t("customerPortal.hierarchy.noTemplate"),
    summary: t("customerPortal.hierarchy.noTemplateHelp"),
  };
  const options = [
    ...(inheritedOption ? [{ ...inheritedOption, value: "inherit" }] : []),
    ...profileOptions,
    ...(noTemplate ? [noTemplateOption] : []),
    ...(custom ? [customOption] : []),
  ];
  const selection =
    options.find((option) => option.value === value) ??
    (unavailableSelection?.value === value ? unavailableSelection : undefined) ??
    (value.startsWith("profile:")
      ? { label: t("customerPortal.hierarchy.unknownTemplate", { id: value.slice("profile:".length) }), summary: "" }
      : undefined);
  return (
    <Select value={value} disabled={disabled} onValueChange={onChange}>
      <SelectTrigger aria-label={title} className="w-full gap-4 px-3 py-3 data-[size=default]:h-auto">
        <SelectValue className="min-w-0" placeholder={t("customerPortal.createDialog.profilePlaceholder")}>
          {selection && <LimitOptionContent label={selection.label} summary={selection.summary} />}
        </SelectValue>
      </SelectTrigger>
      <SelectContent className="p-1">
        {unavailableSelection?.value === value && (
          <SelectItem value={unavailableSelection.value} disabled className="py-3">
            <LimitOptionContent label={unavailableSelection.label} summary={unavailableSelection.summary} />
          </SelectItem>
        )}
        {loading && (
          <SelectItem value="loading" disabled>
            {t("customerPortal.common.loading")}
          </SelectItem>
        )}
        {options.map((option) => (
          <Fragment key={option.value}>
            {option.value === "custom" && profileOptions.length > 0 && <SelectSeparator />}
            <SelectItem
              value={option.value}
              label={option.value === "custom" ? t("customerPortal.createDialog.enterCustomLimits") : option.label}
              className="py-3"
            >
              {option.value === "custom" ? (
                t("customerPortal.createDialog.enterCustomLimits")
              ) : (
                <LimitOptionContent label={option.label} summary={option.summary} />
              )}
            </SelectItem>
          </Fragment>
        ))}
      </SelectContent>
    </Select>
  );
}

export function CustomerBudgetEditor({
  title,
  description,
  id,
  source,
  error,
  children,
  budget,
  className = "space-y-4",
}: {
  title?: string;
  description?: string;
  id: string;
  source?: ComponentProps<typeof CustomerBudgetSourceSelect>;
  error?: ReactNode;
  className?: string;
} & (
  | { children: ReactNode; budget?: never }
  | {
      budget: (Omit<ComponentProps<typeof CustomerBudgetCompactFields>, "id"> & { id?: string }) | null;
      children?: never;
    }
)) {
  return (
    <section className={className} aria-labelledby={title ? `${id}-limits-heading` : undefined}>
      {title && (
        <div className="space-y-1">
          <h3 id={`${id}-limits-heading`} className="font-semibold">
            {title}
          </h3>
          {description && <p className="text-sm text-muted-foreground">{description}</p>}
        </div>
      )}
      {source && <CustomerBudgetSourceSelect {...source} />}
      {error}
      {(!source || source.value === "custom") &&
        (budget ? <CustomerBudgetCompactFields id={id} {...budget} /> : children)}
    </section>
  );
}

export default function CustomerLimitsFields({
  fields,
  disabled,
  onChange,
  profiles,
  title,
  description,
  idPrefix,
  loading = false,
  inheritedOption,
}: {
  fields: CustomerLimits;
  disabled: boolean;
  onChange: (patch: Partial<CustomerLimits>) => void;
  profiles?: Organization["default_budgets"];
  title: string;
  description: string;
  idPrefix: string;
  loading?: boolean;
  inheritedOption?: { label: string; summary: string };
}) {
  const selectSource = (value: string | null) => {
    if (value === "inherit") return onChange({ budgetMode: "inherit" });
    if (value === "custom") return onChange({ budgetMode: "custom" });
    const profile = profiles?.find((group) => `profile:${group.budget_id}` === value);
    if (profile) onChange({ budgetMode: "profile", budgetId: profile.budget_id });
  };
  const changeLimit = <K extends keyof BudgetFields>(key: K, value: BudgetFields[K]) =>
    onChange({ [key === "maxBudget" ? "budgetAmount" : key]: value });
  return (
    <CustomerBudgetEditor
      id={idPrefix}
      title={title}
      description={description}
      source={
        profiles === undefined
          ? undefined
          : {
              profiles,
              value: fields.budgetMode === "profile" ? `profile:${fields.budgetId}` : fields.budgetMode,
              title,
              disabled,
              loading,
              onChange: selectSource,
              inheritedOption: fields.budgetMode === "inherit" ? inheritedOption : undefined,
            }
      }
    >
      <CustomerBudgetFields
        fields={{
          maxBudget: fields.budgetAmount,
          budgetDuration: fields.budgetDuration,
          tpm: fields.tpm,
          rpm: fields.rpm,
          softBudget: fields.softBudget,
          allowedModels: fields.allowedModels,
        }}
        disabled={disabled}
        idPrefix={idPrefix}
        onChange={changeLimit}
      />
    </CustomerBudgetEditor>
  );
}
