import React, { useId } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { Controller } from "react-hook-form";
import { z } from "zod";
import { FormField } from "@/components/shared/form/FormField";
import { BudgetCapPolicySelect } from "@/components/shared/BudgetCapPolicySelect";
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
  monthly_budget: z.number().nonnegative().nullish(),
  annual_budget: z.number().nonnegative().nullish(),
  cap_policy: z.enum(["pause_allow_increase", "pause_until_reset"]).nullish(),
};

const budgetSchema = z
  .object(budgetShape)
  .refine((value) => !value.is_profile || value.monthly_budget != null || value.annual_budget != null, {
    message: i18n.t("budgetProfiles.limitRequired"),
    path: ["monthly_budget"],
  });

export type BudgetFormValues = z.output<typeof budgetSchema>;

export function budgetFormPayload(values: BudgetFormValues, changed: Partial<BudgetFormValues> = values) {
  const { monthly_budget: _monthly, annual_budget: _annual, ...rest } = changed;
  if (!values.is_profile) return rest;
  return {
    ...rest,
    cap_policy: values.cap_policy ?? "pause_until_reset",
    ...("monthly_budget" in changed || "annual_budget" in changed
      ? {
          budget_limits: [
            ...(values.monthly_budget == null ? [] : [{ budget_duration: "1mo", max_budget: values.monthly_budget }]),
            ...(values.annual_budget == null ? [] : [{ budget_duration: "12mo", max_budget: values.annual_budget }]),
          ],
        }
      : {}),
    max_budget: null,
    budget_duration: null,
    tpm_limit: null,
    rpm_limit: null,
    tpd_limit: null,
  };
}

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
  const isProfile = form.watch("is_profile");
  return (
    <FieldGroup className="gap-4">
      <FormField control={form.control} name="is_profile" label={t("budgetProfiles.budgetType")}>
        {(field) => (
          <Select
            value={field.value ? "profile" : "budget"}
            onValueChange={(value) => field.onChange(value === "profile")}
            disabled={isEditing}
            items={[
              { value: "budget", label: t("budgetProfiles.budget") },
              { value: "profile", label: t("budgetProfiles.profile") },
            ]}
          >
            <SelectTrigger id={field.id}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="budget">{t("budgetProfiles.budget")}</SelectItem>
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
        <>
          <FieldDescription>{t("budgetProfiles.profileHelp")}</FieldDescription>
          <FormField control={form.control} name="monthly_budget" label={t("budgetProfiles.profileMonthly")}>
            {(field) => <Input {...numberProps(field)} min={0} step={0.01} />}
          </FormField>
          <FormField control={form.control} name="annual_budget" label={t("budgetProfiles.profileAnnual")}>
            {(field) => <Input {...numberProps(field)} min={0} step={0.01} />}
          </FormField>
          <FormField
            control={form.control}
            name="cap_policy"
            label={t("budgetProfiles.cap")}
            description={t(
              form.watch("cap_policy") === "pause_allow_increase"
                ? "budgetProfiles.allowIncreaseHelp"
                : "budgetProfiles.waitHelp",
            )}
          >
            {(field) => (
              <BudgetCapPolicySelect
                id={field.id}
                aria-describedby={field["aria-describedby"]}
                value={field.value ?? "pause_until_reset"}
                onValueChange={field.onChange}
              />
            )}
          </FormField>
          <FieldDescription>{t("budgetProfiles.customerRequestHelp")}</FieldDescription>
          <FieldDescription>{t("budgetProfiles.resetHelp")}</FieldDescription>
        </>
      ) : (
        <SpendLimitField form={form} />
      )}
      {!isProfile && (
        <>
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
