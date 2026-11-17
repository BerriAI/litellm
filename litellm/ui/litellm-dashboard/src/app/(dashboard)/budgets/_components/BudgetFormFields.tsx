import React, { useId } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { Controller, useWatch } from "react-hook-form";
import { z } from "zod";
import { FormField } from "@/components/shared/form/FormField";
import { Badge } from "@/components/ui/badge";
import { ButtonGroup } from "@/components/ui/button-group";
import {
  Field,
  FieldDescription,
  FieldError,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSeparator,
  FieldSet,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText } from "@/components/ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { getBudgetDurationLabel } from "@/components/common_components/budget_duration_dropdown";
import i18n from "@/i18n/i18n";
import { useZodForm } from "@/lib/forms/useZodForm";
import { COMBINED_BUDGET_DURATION } from "./budgetLimits";

const budgetShape = {
  budget_id: z.string().min(1, { error: () => i18n.t("budgets.form.errors.budgetIdRequired") }),
  tpm_limit: z.number().nullish(),
  rpm_limit: z.number().nullish(),
  tpd_limit: z.number().nullish(),
  max_budget: z.number().nullish(),
  budget_duration: z.string().nullish(),
  annual_budget: z.number().nullish(),
  additional_max_budget: z.number().nullish(),
  additional_budget_duration: z.string().nullish(),
};

const budgetSchema = z.object(budgetShape).superRefine((values, context) => {
  if (values.budget_duration !== COMBINED_BUDGET_DURATION) return;
  for (const field of ["max_budget", "annual_budget"] as const) {
    if (values[field] == null || values[field] < 0) {
      context.addIssue({ code: "custom", path: [field], message: i18n.t("budgets.form.errors.invalidBudgetLimit") });
    }
  }
});

export type BudgetFormValues = z.output<typeof budgetSchema>;

export const useBudgetForm = (defaultValues: z.input<typeof budgetSchema>) =>
  useZodForm(budgetSchema, { defaultValues });

type BudgetForm = ReturnType<typeof useBudgetForm>;
type NumberFieldName = "tpm_limit" | "rpm_limit" | "tpd_limit" | "max_budget";

const NO_RESET = "never";

const RESET_OPTIONS: readonly { value: string; labelKey: ParseKeys }[] = [
  { value: NO_RESET, labelKey: "budgets.form.resetOptions.inTotal" },
  { value: "24h", labelKey: "budgets.form.resetOptions.perDay" },
  { value: "7d", labelKey: "budgets.form.resetOptions.perWeek" },
  { value: "30d", labelKey: "budgets.form.resetOptions.perMonth" },
  { value: "12mo", labelKey: "budgets.form.resetOptions.perYear" },
  { value: COMBINED_BUDGET_DURATION, labelKey: "budgets.form.resetOptions.perMonthAndYear" },
];

const numberProps = <T extends { value?: number | null; onChange: (value: number | null) => void }>({
  value,
  onChange,
  ...field
}: T) => ({
  ...field,
  type: "number",
  value: value ?? "",
  placeholder: i18n.t("budgets.form.noLimit"),
  onChange: (event: React.ChangeEvent<HTMLInputElement>) =>
    onChange(event.target.value === "" ? null : event.target.valueAsNumber),
});

const OptionalBadge = () => {
  const { t } = useTranslation();
  return <Badge variant="secondary">{t("budgets.form.optional")}</Badge>;
};

function SpendLimitField({ form, additional = false }: { form: BudgetForm; additional?: boolean }) {
  const { t } = useTranslation();
  const amountId = useId();
  const amountField = additional ? "additional_max_budget" : "max_budget";
  const durationField = additional ? "additional_budget_duration" : "budget_duration";
  const duration = useWatch({ control: form.control, name: durationField });
  const mixed = useWatch({ control: form.control, name: "additional_max_budget" }) !== undefined;
  const combined = duration === COMBINED_BUDGET_DURATION;
  const options = RESET_OPTIONS.filter(({ value }) => {
    if (additional) return value !== COMBINED_BUDGET_DURATION;
    const calendar = value === "30d" || value === "12mo" || value === COMBINED_BUDGET_DURATION;
    return !mixed || calendar;
  }).map(({ value, labelKey }) => ({ value, label: t(labelKey) }));
  const resetItems =
    duration && !options.some(({ value }) => value === duration)
      ? [...options, { value: duration, label: getBudgetDurationLabel(duration, t) }]
      : options;
  const primaryLabel = combined ? "budgets.form.monthlyLimit" : "budgets.form.spendLimit";
  return (
    <Field className="gap-2">
      <FieldLabel htmlFor={amountId}>
        {t(additional ? "budgets.form.additionalSpendLimit" : primaryLabel)}
        {!combined && <OptionalBadge />}
      </FieldLabel>
      <ButtonGroup className="w-full">
        <Controller
          control={form.control}
          name={amountField}
          render={({ field, fieldState }) => (
            <InputGroup>
              <InputGroupAddon>
                <InputGroupText>$</InputGroupText>
              </InputGroupAddon>
              <InputGroupInput
                {...numberProps(field)}
                id={amountId}
                aria-invalid={fieldState.invalid || undefined}
                step={combined ? "any" : 0.01}
              />
            </InputGroup>
          )}
        />
        <Controller
          control={form.control}
          name={durationField}
          render={({ field }) => (
            <Select
              items={resetItems}
              value={field.value ?? NO_RESET}
              onValueChange={(next) => {
                if (combined && next === "12mo") {
                  form.setValue("max_budget", form.getValues("annual_budget"), { shouldDirty: true });
                } else if (next === COMBINED_BUDGET_DURATION && field.value === "12mo") {
                  form.setValue("annual_budget", form.getValues("max_budget"), { shouldDirty: true });
                  form.setValue("max_budget", null, { shouldDirty: true });
                }
                field.onChange(next === NO_RESET ? null : next);
              }}
            >
              <SelectTrigger
                aria-label={t(additional ? "budgets.form.additionalResetPeriod" : "budgets.form.resetPeriod")}
                className="w-44"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {resetItems.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        />
      </ButtonGroup>
      {additional && <FieldDescription>{t("budgets.form.additionalLimitHint")}</FieldDescription>}
      <FieldError errors={[form.formState.errors[amountField]]} />
    </Field>
  );
}

function RateLimitField({ form, name, label }: { form: BudgetForm; name: NumberFieldName; label: string }) {
  return (
    <FormField
      control={form.control}
      name={name}
      label={label}
      orientation="horizontal"
      className="justify-between *:data-[slot=field-label]:flex-none"
    >
      {(control) => <Input {...numberProps(control)} className="w-36" />}
    </FormField>
  );
}

export function BudgetFormFields({ form, isEditing }: { form: BudgetForm; isEditing: boolean }) {
  const { t } = useTranslation();
  const combined = useWatch({ control: form.control, name: "budget_duration" }) === COMBINED_BUDGET_DURATION;
  const mixed = useWatch({ control: form.control, name: "additional_max_budget" }) !== undefined;
  return (
    <FieldGroup className="gap-4">
      <FormField
        control={form.control}
        name="budget_id"
        label={t("budgets.form.budgetId")}
        description={isEditing ? t("budgets.form.budgetIdLocked") : undefined}
        className="gap-2"
      >
        {({ ref, ...field }) => <Input {...field} ref={ref} value={field.value ?? ""} disabled={isEditing} />}
      </FormField>
      <SpendLimitField form={form} />
      {combined && (
        <FormField
          control={form.control}
          name="annual_budget"
          label={t("budgets.form.annualLimit")}
          description={t("budgets.form.calendarLimitsHint")}
        >
          {(field) => <Input {...numberProps(field)} min={0} step="any" />}
        </FormField>
      )}
      {mixed && <SpendLimitField form={form} additional />}
      <FieldSeparator />
      <FieldSet className="gap-3">
        <FieldLegend className="mb-4 flex items-center gap-2 text-[15px] leading-[22px] font-semibold">
          {t("budgets.form.rateLimits")}
          <OptionalBadge />
        </FieldLegend>
        <RateLimitField form={form} name="tpm_limit" label={t("budgets.form.tpmLimit")} />
        <RateLimitField form={form} name="rpm_limit" label={t("budgets.form.rpmLimit")} />
        <RateLimitField form={form} name="tpd_limit" label={t("budgets.form.tpdLimit")} />
        <FieldDescription>{t("budgets.form.batchNote")}</FieldDescription>
      </FieldSet>
    </FieldGroup>
  );
}
