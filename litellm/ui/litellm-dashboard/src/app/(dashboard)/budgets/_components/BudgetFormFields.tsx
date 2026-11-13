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
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSeparator,
  FieldSet,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupInput, InputGroupText } from "@/components/ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import i18n from "@/i18n/i18n";
import { useZodForm } from "@/lib/forms/useZodForm";

const budgetShape = {
  budget_id: z.string().min(1, { error: () => i18n.t("budgets.form.errors.budgetIdRequired") }),
  tpm_limit: z.number().nullish(),
  rpm_limit: z.number().nullish(),
  tpd_limit: z.number().nullish(),
  max_budget: z.number().nullish(),
  budget_duration: z.string().nullish(),
  is_profile: z.boolean().optional(),
  revision: z.string().optional(),
  monthly: z.number().nullish(),
  annual: z.number().nullish(),
  cap_policy: z.enum(["pause_allow_increase", "pause_until_reset"]).nullish(),
};

const budgetSchema = z.object(budgetShape).superRefine((values, context) => {
  if (!values.is_profile) return;
  if (values.monthly == null && values.annual == null) {
    context.addIssue({ code: "custom", message: i18n.t("budgetProfiles.limitRequired"), path: ["monthly"] });
  }
  for (const field of ["monthly", "annual"] as const) {
    if (values[field] != null && values[field] < 0) {
      context.addIssue({ code: "custom", message: i18n.t("budgetProfiles.nonnegative"), path: [field] });
    }
  }
});

export type BudgetFormValues = z.output<typeof budgetSchema>;

export const useBudgetForm = (defaultValues: z.input<typeof budgetSchema>) =>
  useZodForm(budgetSchema, { defaultValues });

type BudgetForm = ReturnType<typeof useBudgetForm>;
type NumberFieldName = "tpm_limit" | "rpm_limit" | "tpd_limit" | "max_budget" | "monthly" | "annual";

const NO_RESET = "never";

const RESET_OPTIONS: readonly { value: string; labelKey: ParseKeys }[] = [
  { value: NO_RESET, labelKey: "budgets.form.resetOptions.inTotal" },
  { value: "24h", labelKey: "budgets.form.resetOptions.perDay" },
  { value: "7d", labelKey: "budgets.form.resetOptions.perWeek" },
  { value: "30d", labelKey: "budgets.form.resetOptions.perMonth" },
  { value: "12mo", labelKey: "budgets.form.resetOptions.perYear" },
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

function SpendLimitField({ form }: { form: BudgetForm }) {
  const { t } = useTranslation();
  const amountId = useId();
  const resetItems = RESET_OPTIONS.map(({ value, labelKey }) => ({ value, label: t(labelKey) }));
  return (
    <Field className="gap-2">
      <FieldLabel htmlFor={amountId}>
        {t("budgets.form.spendLimit")}
        <OptionalBadge />
      </FieldLabel>
      <ButtonGroup className="w-full">
        <Controller
          control={form.control}
          name="max_budget"
          render={({ field }) => (
            <InputGroup>
              <InputGroupAddon>
                <InputGroupText>$</InputGroupText>
              </InputGroupAddon>
              <InputGroupInput {...numberProps(field)} id={amountId} step={0.01} />
            </InputGroup>
          )}
        />
        <Controller
          control={form.control}
          name="budget_duration"
          render={({ field }) => (
            <Select
              items={resetItems}
              value={field.value ?? NO_RESET}
              onValueChange={(next) => field.onChange(next === NO_RESET ? null : next)}
            >
              <SelectTrigger aria-label={t("budgets.form.resetPeriod")} className="w-32">
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
  const isProfile = useWatch({ control: form.control, name: "is_profile" });
  return (
    <FieldGroup className="gap-4">
      <FormField control={form.control} name="is_profile" label={t("budgetProfiles.kind")}>
        {({ id, value, onChange }) => (
          <Select
            value={value ? "profile" : "shared"}
            onValueChange={(next) => onChange(next === "profile")}
            disabled={isEditing}
            items={[
              { value: "shared", label: t("budgetProfiles.shared") },
              { value: "profile", label: t("budgetProfiles.profile") },
            ]}
          >
            <SelectTrigger id={id}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="shared">{t("budgetProfiles.shared")}</SelectItem>
              <SelectItem value="profile">{t("budgetProfiles.profile")}</SelectItem>
            </SelectContent>
          </Select>
        )}
      </FormField>
      <FormField
        control={form.control}
        name="budget_id"
        label={t("budgets.form.budgetId")}
        description={isEditing ? t("budgets.form.budgetIdLocked") : undefined}
        className="gap-2"
      >
        {({ ref, ...field }) => <Input {...field} ref={ref} value={field.value ?? ""} disabled={isEditing} />}
      </FormField>
      {isProfile ? (
        <FieldSet className="gap-3">
          <FieldDescription>{t("budgetProfiles.profileHelp")}</FieldDescription>
          <RateLimitField form={form} name="monthly" label={t("budgetProfiles.monthlyLimit")} />
          <RateLimitField form={form} name="annual" label={t("budgetProfiles.annualLimit")} />
          <FieldDescription>{t("budgetProfiles.independentHelp")}</FieldDescription>
          <FormField
            control={form.control}
            name="cap_policy"
            label={t("budgetProfiles.capPolicy")}
            description={t("budgetProfiles.capPolicyHelp")}
          >
            {({ id, value, onChange }) => (
              <Select
                value={value ?? "pause_until_reset"}
                onValueChange={onChange}
                items={[
                  { value: "pause_until_reset", label: t("budgetProfiles.pauseUntilReset") },
                  { value: "pause_allow_increase", label: t("budgetProfiles.pauseAllowIncrease") },
                ]}
              >
                <SelectTrigger id={id}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pause_until_reset">{t("budgetProfiles.pauseUntilReset")}</SelectItem>
                  <SelectItem value="pause_allow_increase">{t("budgetProfiles.pauseAllowIncrease")}</SelectItem>
                </SelectContent>
              </Select>
            )}
          </FormField>
        </FieldSet>
      ) : (
        <>
          <SpendLimitField form={form} />
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
        </>
      )}
    </FieldGroup>
  );
}
